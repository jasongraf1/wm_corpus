#!/usr/bin/env python
"""
03_merge_speakers.py

Step 4 of the raw-media pipeline: merge Whisper word timestamps with
pyannote diarization to produce a speaker-labelled word stream.

For every ``<rec_id>_words.csv`` found (recursively) under the
transcription directory, the matching diarization output is located
under the diarization directory (same relative sub-path, same
``rec_id``) and each word is assigned to a speaker.

Assignment strategy
--------------------
If an "exclusive" diarization CSV is available (pyannote community-1 /
4.x, where exactly one speaker is active at any instant), each word is
assigned by simple interval lookup on its midpoint -- fast and
unambiguous, since there is nothing to disambiguate.

If only "regular" diarization is available (may contain overlapping
segments, e.g. 3.1), each word is assigned to the speaker with maximum
temporal overlap against the word's span; a midpoint lookup is the
tiebreak of last resort.

Either way, words whose midpoint falls in a gap between diarization
segments (silence the diarizer didn't attribute to anyone) are assigned
to the nearer neighbouring segment if it is within --gap-tolerance-s;
otherwise the word is left unassigned (speaker = UNASSIGNED) and
flagged for manual attention -- this should be rare and worth checking
if it isn't.

Independently of which diarization source was used for assignment, the
REGULAR diarization (when available) is always used to detect genuine
overlap regions (time spans where 2+ distinct speakers are
simultaneously active, e.g. backchannels). Every word is flagged
in_overlap_region=True/False with the other speaker(s) present, so you
can see at the EAF/manual-correction stage which words were spoken
during cross-talk even when the automatic assignment gave a single
best-guess speaker.

Outputs (per recording, mirrored directory structure):

    merged/<relative/path>/<rec_id>_merged.csv
        word, start_s, end_s, probability, segment_id, speaker,
        assignment_method, assignment_distance_s, in_overlap_region,
        overlap_with_speakers

    merged/<relative/path>/<rec_id>_merge_manifest.json
        diarization source used, parameters, per-method word counts,
        per-speaker word/time totals, unassigned and overlap counts.

Usage
-----
    conda activate socio
    python 03_merge_speakers.py <transcription_dir> <diarization_dir> <output_dir> [options]

    python 03_merge_speakers.py D:/corpus/transcription D:/corpus/diarization D:/corpus/merged
    python 03_merge_speakers.py IN_T IN_D OUT --gap-tolerance-s 0.5

No third-party dependencies.
"""

import argparse
import bisect
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

WORDS_SUFFIX = "_words.csv"
EXCLUSIVE_SUFFIX = "_exclusive.csv"
SEGMENTS_SUFFIX = "_segments.csv"
UNASSIGNED_LABEL = "UNASSIGNED"

# How far back (in seconds) the overlap-based scan looks for candidate
# regular-diarization segments before giving up on a word. Generous
# relative to expected turn lengths in interview data.
OVERLAP_SCAN_SAFETY_MARGIN_S = 5.0


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_words_csvs(transcription_dir: Path):
    return sorted(p for p in transcription_dir.rglob(f"*{WORDS_SUFFIX}")
                  if p.is_file())


def rec_id_from_words_path(words_path: Path) -> str:
    name = words_path.name
    assert name.endswith(WORDS_SUFFIX)
    return name[: -len(WORDS_SUFFIX)]


def load_words_csv(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "word": r["word"],
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "probability": float(r["probability"]),
                "segment_id": r["segment_id"],
            })
    return rows


def load_segments_csv(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "speaker": r["speaker"],
            })
    rows.sort(key=lambda s: s["start_s"])
    return rows


# ---------------------------------------------------------------------------
# Overlap-region detection (from regular/raw diarization)
# ---------------------------------------------------------------------------

def compute_overlap_intervals(regular_rows):
    """
    Sweep-line over regular diarization segments to find time spans
    where 2+ distinct speakers are simultaneously active.

    Returns a list of (start_s, end_s, frozenset(speakers)) sorted by
    start_s, covering only the overlapping spans (gaps between them are
    implicit -- not returned).
    """
    if not regular_rows:
        return []

    # Event type 0 = segment end, 1 = segment start. Sorting ends
    # before starts at an identical timestamp avoids flagging a false
    # overlap when one turn ends exactly as the next begins.
    events = []
    for r in regular_rows:
        events.append((r["start_s"], 1, r["speaker"]))
        events.append((r["end_s"], 0, r["speaker"]))
    events.sort(key=lambda e: (e[0], e[1]))

    active = {}          # speaker -> concurrent segment count
    intervals = []
    prev_time = None

    for time, kind, speaker in events:
        if prev_time is not None and time > prev_time:
            active_speakers = frozenset(s for s, c in active.items() if c > 0)
            if len(active_speakers) >= 2:
                intervals.append((prev_time, time, active_speakers))
        if kind == 1:
            active[speaker] = active.get(speaker, 0) + 1
        else:
            active[speaker] = active.get(speaker, 0) - 1
            if active[speaker] <= 0:
                del active[speaker]
        prev_time = time

    return intervals


def find_overlap_speakers(midpoint, overlap_intervals, overlap_starts):
    idx = bisect.bisect_right(overlap_starts, midpoint) - 1
    if idx >= 0:
        start, end, speakers = overlap_intervals[idx]
        if start <= midpoint < end:
            return speakers
    return None


# ---------------------------------------------------------------------------
# Speaker assignment
# ---------------------------------------------------------------------------

def assign_by_midpoint(midpoint, segs, starts, gap_tolerance_s):
    """
    Interval lookup: which segment contains this midpoint? Used
    directly for exclusive diarization, and as the gap-filling fallback
    for regular diarization when overlap-based assignment finds
    nothing.
    """
    idx = bisect.bisect_right(starts, midpoint) - 1
    if idx >= 0 and segs[idx]["start_s"] <= midpoint < segs[idx]["end_s"]:
        return segs[idx]["speaker"], "midpoint_lookup", 0.0

    candidates = []
    if idx >= 0:
        dist = midpoint - segs[idx]["end_s"]
        if dist >= 0:
            candidates.append((segs[idx]["speaker"], dist))
    if idx + 1 < len(segs):
        dist = segs[idx + 1]["start_s"] - midpoint
        if dist >= 0:
            candidates.append((segs[idx + 1]["speaker"], dist))

    if candidates:
        speaker, dist = min(candidates, key=lambda c: c[1])
        if dist <= gap_tolerance_s:
            return speaker, "nearest_gap", round(dist, 4)

    return UNASSIGNED_LABEL, "unassigned", None


def assign_by_max_overlap(word_start, word_end, segs, starts):
    """
    For possibly-overlapping (regular) diarization: assign to whichever
    segment shares the most time with the word span.
    """
    idx_hi = bisect.bisect_right(starts, word_end) - 1
    best_speaker, best_overlap = None, 0.0
    i = idx_hi
    while i >= 0:
        seg = segs[i]
        if seg["end_s"] <= word_start - OVERLAP_SCAN_SAFETY_MARGIN_S:
            break
        overlap = min(seg["end_s"], word_end) - max(seg["start_s"], word_start)
        if overlap > best_overlap:
            best_overlap, best_speaker = overlap, seg["speaker"]
        i -= 1

    if best_speaker is not None and best_overlap > 0:
        return best_speaker, "max_overlap", round(best_overlap, 4)
    return None, None, None


# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------

def process_recording(words_path, diar_dir, rec_id, gap_tolerance_s):
    words = load_words_csv(words_path)

    exclusive_path = diar_dir / f"{rec_id}{EXCLUSIVE_SUFFIX}"
    regular_path = diar_dir / f"{rec_id}{SEGMENTS_SUFFIX}"

    regular_rows = load_segments_csv(regular_path) if regular_path.exists() else []
    overlap_intervals = compute_overlap_intervals(regular_rows)
    overlap_starts = [iv[0] for iv in overlap_intervals]

    if exclusive_path.exists():
        segs = load_segments_csv(exclusive_path)
        source = "exclusive"
    elif regular_rows:
        segs = regular_rows
        source = "regular_fallback"
    else:
        raise FileNotFoundError(
            f"no diarization found for {rec_id} "
            f"(looked for {exclusive_path.name} and {regular_path.name})")

    starts = [s["start_s"] for s in segs]

    merged_rows = []
    method_counts = {}
    speaker_word_counts = {}
    n_in_overlap = 0

    for w in words:
        midpoint = (w["start_s"] + w["end_s"]) / 2.0

        if source == "exclusive":
            speaker, method, dist = assign_by_midpoint(
                midpoint, segs, starts, gap_tolerance_s)
        else:
            speaker, method, dist = assign_by_max_overlap(
                w["start_s"], w["end_s"], segs, starts)
            if speaker is None:
                speaker, method, dist = assign_by_midpoint(
                    midpoint, segs, starts, gap_tolerance_s)

        in_overlap = False
        overlap_with = ""
        if overlap_intervals:
            present = find_overlap_speakers(midpoint, overlap_intervals,
                                            overlap_starts)
            if present is not None:
                in_overlap = True
                n_in_overlap += 1
                others = sorted(s for s in present if s != speaker)
                overlap_with = ";".join(others)

        method_counts[method] = method_counts.get(method, 0) + 1
        speaker_word_counts.setdefault(speaker, 0)
        speaker_word_counts[speaker] += 1

        merged_rows.append({
            "word": w["word"],
            "start_s": w["start_s"],
            "end_s": w["end_s"],
            "probability": w["probability"],
            "segment_id": w["segment_id"],
            "speaker": speaker,
            "assignment_method": method,
            "assignment_distance_s": dist if dist is not None else "",
            "in_overlap_region": "yes" if in_overlap else "no",
            "overlap_with_speakers": overlap_with,
        })

    manifest = {
        "recording_id": rec_id,
        "diarization_source": source,
        "gap_tolerance_s": gap_tolerance_s,
        "overlap_detection_available": bool(regular_rows),
        "n_words": len(merged_rows),
        "n_unassigned": method_counts.get("unassigned", 0),
        "pct_unassigned": round(
            100 * method_counts.get("unassigned", 0) / len(merged_rows), 2)
            if merged_rows else 0.0,
        "n_in_overlap_region": n_in_overlap,
        "pct_in_overlap_region": round(100 * n_in_overlap / len(merged_rows), 2)
            if merged_rows else 0.0,
        "method_counts": method_counts,
        "speaker_word_counts": speaker_word_counts,
    }
    return merged_rows, manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Assign Whisper words to speakers using pyannote "
                    "diarization output.")
    ap.add_argument("transcription_dir", type=Path,
                    help="Directory of *_words.csv files (searched recursively)")
    ap.add_argument("diarization_dir", type=Path,
                    help="Directory of diarization outputs (mirrors "
                         "transcription_dir's relative structure)")
    ap.add_argument("output_dir", type=Path,
                    help="Directory for merged output")
    ap.add_argument("--gap-tolerance-s", type=float, default=0.3,
                    help="Max distance (s) to the nearest diarization "
                         "segment for a word in a diarization gap to "
                         "still be assigned (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="Re-merge even if outputs already exist")
    args = ap.parse_args()

    transcription_dir = args.transcription_dir.resolve()
    diarization_dir = args.diarization_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not transcription_dir.is_dir():
        sys.exit(f"ERROR: transcription directory not found: {transcription_dir}")
    if not diarization_dir.is_dir():
        sys.exit(f"ERROR: diarization directory not found: {diarization_dir}")

    words_csvs = find_words_csvs(transcription_dir)
    if not words_csvs:
        sys.exit(f"ERROR: no *{WORDS_SUFFIX} files found under {transcription_dir}")

    print(f"Found {len(words_csvs)} transcription file(s) under {transcription_dir}\n")

    n_done = n_skipped = n_failed = 0
    errors = []
    warn_threshold_pct = 2.0   # flag recordings with unusually high unassigned rate
    flagged = []

    for i, words_path in enumerate(words_csvs, 1):
        rec_id = rec_id_from_words_path(words_path)
        rel_parent = words_path.parent.relative_to(transcription_dir)
        diar_dir = diarization_dir / rel_parent
        out_parent = output_dir / rel_parent
        merged_path = out_parent / f"{rec_id}_merged.csv"
        manifest_path = out_parent / f"{rec_id}_merge_manifest.json"

        prefix = f"[{i}/{len(words_csvs)}] {rec_id}"

        if merged_path.exists() and manifest_path.exists() and not args.force:
            n_skipped += 1
            print(f"{prefix} ... skipped (outputs exist)")
            continue

        try:
            merged_rows, manifest = process_recording(
                words_path, diar_dir, rec_id, args.gap_tolerance_s)

            out_parent.mkdir(parents=True, exist_ok=True)
            with open(merged_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "word", "start_s", "end_s", "probability", "segment_id",
                    "speaker", "assignment_method", "assignment_distance_s",
                    "in_overlap_region", "overlap_with_speakers",
                ])
                writer.writeheader()
                writer.writerows(merged_rows)

            manifest["timestamp_utc"] = datetime.now(timezone.utc) \
                .isoformat(timespec="seconds")
            manifest["words_csv"] = str(words_path)
            manifest["outputs"] = {
                "merged_csv": str(merged_path),
                "manifest": str(manifest_path),
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            n_done += 1
            note = ""
            if manifest["pct_unassigned"] > warn_threshold_pct:
                note = f"  ** {manifest['pct_unassigned']}% unassigned -- check gap tolerance / diarization **"
                flagged.append((rec_id, manifest["pct_unassigned"]))
            print(f"{prefix} ... done "
                  f"(source={manifest['diarization_source']}, "
                  f"{manifest['n_words']} words, "
                  f"{manifest['pct_unassigned']}% unassigned, "
                  f"{manifest['pct_in_overlap_region']}% in overlap){note}")

        except Exception as e:
            n_failed += 1
            errors.append((rec_id, str(e)))
            print(f"{prefix} ... FAILED: {e}")

    print(f"\nSummary: {n_done} merged, {n_skipped} skipped, {n_failed} failed.")
    if flagged:
        print("Recordings with elevated unassigned rate:")
        for rec_id, pct in flagged:
            print(f"  {rec_id}: {pct}%")
    if errors:
        err_path = output_dir / "merge_errors.csv"
        new_file = not err_path.exists()
        with open(err_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["timestamp_utc", "recording_id", "error"])
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for rec_id, msg in errors:
                writer.writerow([ts, rec_id, msg])
        print(f"Errors logged to {err_path}")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()