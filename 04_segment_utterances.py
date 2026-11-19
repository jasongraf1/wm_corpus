#!/usr/bin/env python
"""
04_segment_utterances.py

Step 5 of the raw-media pipeline: group the speaker-labelled word
stream (output of 03_merge_speakers.py) into utterance-level annotation
units, ready to become ELAN annotations (one per speaker tier).

Whisper's own segment boundaries are NOT used for this -- they ignore
speaker identity entirely. Utterances are rebuilt from scratch by
walking the word stream in time order and starting a new utterance
whenever:

  (a) the assigned speaker changes from the previous word, or
  (b) the pause since the previous word's end exceeds --pause-threshold-s

UNASSIGNED words (from 03_merge_speakers.py, words the diarizer
couldn't place) are treated as their own pseudo-speaker for segmentation
purposes -- they naturally form their own (usually short) utterance(s)
rather than being silently folded into a neighbour, so they stay
visible for manual attention in ELAN.

Utterances are reported, not force-split, when they run long: pauses
and speaker changes are the only segmentation triggers by default,
since a linguistically arbitrary cut is worse for later MFA alignment
than a long utterance. Pass --max-utterance-s to additionally force a
hard split at the pause-of-least-resistance once an utterance would
exceed that duration (rare in interview data, but a hedge against an
uninterrupted monologue).

Outputs (per recording, mirrored directory structure):

    utterances/<relative/path>/<rec_id>_utterances.csv
        utterance_id, speaker, start_s, end_s, duration_s, n_words,
        text, mean_word_probability, min_word_probability,
        contains_overlap, overlap_word_count, unassigned_word_count,
        gap_before_s

    utterances/<relative/path>/<rec_id>_segment_manifest.json
        parameters, utterance/speaker counts and durations, long-
        utterance and overlap/unassigned summary stats.

Usage
-----
    conda activate socio
    python 04_segment_utterances.py <merged_dir> <output_dir> [options]

    python 04_segment_utterances.py D:/corpus/merged D:/corpus/utterances
    python 04_segment_utterances.py IN OUT --pause-threshold-s 0.6
    python 04_segment_utterances.py IN OUT --max-utterance-s 15

No third-party dependencies.
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

MERGED_SUFFIX = "_merged.csv"
UNASSIGNED_LABEL = "UNASSIGNED"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_merged_csvs(merged_dir: Path):
    return sorted(p for p in merged_dir.rglob(f"*{MERGED_SUFFIX}") if p.is_file())


def rec_id_from_merged_path(merged_path: Path) -> str:
    name = merged_path.name
    assert name.endswith(MERGED_SUFFIX)
    return name[: -len(MERGED_SUFFIX)]


def load_merged_csv(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "word": r["word"],
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "probability": float(r["probability"]),
                "speaker": r["speaker"],
                "in_overlap_region": r["in_overlap_region"] == "yes",
            })
    rows.sort(key=lambda w: w["start_s"])
    return rows


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def new_utterance(word, gap_before_s):
    return {
        "speaker": word["speaker"],
        "start_s": word["start_s"],
        "end_s": word["end_s"],
        "words": [word],
        "gap_before_s": gap_before_s,
    }


def finalize_utterance(utt, utt_index):
    words = utt["words"]
    probs = [w["probability"] for w in words]
    overlap_flags = [w["in_overlap_region"] for w in words]
    unassigned_count = sum(1 for w in words if w["speaker"] == UNASSIGNED_LABEL)
    return {
        "utterance_id": f"u{utt_index:04d}",
        "speaker": utt["speaker"],
        "start_s": round(utt["start_s"], 4),
        "end_s": round(utt["end_s"], 4),
        "duration_s": round(utt["end_s"] - utt["start_s"], 4),
        "n_words": len(words),
        "text": " ".join(w["word"] for w in words),
        "mean_word_probability": round(sum(probs) / len(probs), 4),
        "min_word_probability": round(min(probs), 4),
        "contains_overlap": "yes" if any(overlap_flags) else "no",
        "overlap_word_count": sum(overlap_flags),
        "unassigned_word_count": unassigned_count,
        "gap_before_s": (round(utt["gap_before_s"], 4)
                        if utt["gap_before_s"] is not None else ""),
    }


def segment_words(words, pause_threshold_s, max_utterance_s):
    utterances = []
    current = None

    for word in words:
        if current is None:
            current = new_utterance(word, gap_before_s=None)
            continue

        gap = word["start_s"] - current["end_s"]
        speaker_changed = word["speaker"] != current["speaker"]
        would_exceed_max = (
            max_utterance_s is not None
            and (word["end_s"] - current["start_s"]) > max_utterance_s
        )

        if speaker_changed or gap > pause_threshold_s or would_exceed_max:
            utterances.append(current)
            current = new_utterance(word, gap_before_s=max(gap, 0.0))
        else:
            current["words"].append(word)
            current["end_s"] = word["end_s"]

    if current is not None:
        utterances.append(current)

    return [finalize_utterance(u, i) for i, u in enumerate(utterances, 1)]


# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------

def process_recording(merged_path, pause_threshold_s, max_utterance_s,
                      report_long_s):
    words = load_merged_csv(merged_path)
    if not words:
        return [], {
            "n_utterances": 0, "n_words": 0,
            "warning": "no words found in merged CSV",
        }

    utterances = segment_words(words, pause_threshold_s, max_utterance_s)

    speaker_stats = {}
    for u in utterances:
        s = speaker_stats.setdefault(u["speaker"], {
            "n_utterances": 0, "speech_time_s": 0.0, "n_words": 0})
        s["n_utterances"] += 1
        s["speech_time_s"] = round(s["speech_time_s"] + u["duration_s"], 4)
        s["n_words"] += u["n_words"]

    durations = [u["duration_s"] for u in utterances]
    long_utterances = [u["utterance_id"] for u in utterances
                       if u["duration_s"] > report_long_s]
    overlap_utterances = sum(1 for u in utterances if u["contains_overlap"] == "yes")
    unassigned_utterances = sum(1 for u in utterances
                                if u["speaker"] == UNASSIGNED_LABEL)

    manifest = {
        "pause_threshold_s": pause_threshold_s,
        "max_utterance_s": max_utterance_s,
        "report_long_utterance_s": report_long_s,
        "n_words": len(words),
        "n_utterances": len(utterances),
        "mean_utterance_duration_s": round(sum(durations) / len(durations), 4)
            if durations else 0.0,
        "max_utterance_duration_s": round(max(durations), 4) if durations else 0.0,
        "n_long_utterances": len(long_utterances),
        "long_utterance_ids": long_utterances,
        "n_overlap_utterances": overlap_utterances,
        "n_unassigned_utterances": unassigned_utterances,
        "speaker_stats": speaker_stats,
    }
    return utterances, manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Group speaker-labelled words into utterance-level "
                    "annotation units.")
    ap.add_argument("merged_dir", type=Path,
                    help="Directory of *_merged.csv files (searched recursively)")
    ap.add_argument("output_dir", type=Path,
                    help="Directory for utterance-level output")
    ap.add_argument("--pause-threshold-s", type=float, default=0.75,
                    help="Pause duration (s) since the previous word that "
                         "triggers a new utterance (default: %(default)s)")
    ap.add_argument("--max-utterance-s", type=float, default=None,
                    help="If set, force a hard split once an utterance "
                         "would exceed this duration, even without a "
                         "speaker change or long pause (default: disabled)")
    ap.add_argument("--report-long-utterance-s", type=float, default=15.0,
                    help="Utterances longer than this are listed in the "
                         "manifest for review (default: %(default)s); "
                         "purely informational unless --max-utterance-s "
                         "is also set")
    ap.add_argument("--force", action="store_true",
                    help="Re-segment even if outputs already exist")
    args = ap.parse_args()

    merged_dir = args.merged_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not merged_dir.is_dir():
        sys.exit(f"ERROR: merged directory not found: {merged_dir}")

    merged_csvs = find_merged_csvs(merged_dir)
    if not merged_csvs:
        sys.exit(f"ERROR: no *{MERGED_SUFFIX} files found under {merged_dir}")

    print(f"Found {len(merged_csvs)} merged file(s) under {merged_dir}")
    print(f"pause_threshold_s={args.pause_threshold_s}  "
          f"max_utterance_s={args.max_utterance_s}\n")

    n_done = n_skipped = n_failed = 0
    errors = []
    flagged = []

    for i, merged_path in enumerate(merged_csvs, 1):
        rec_id = rec_id_from_merged_path(merged_path)
        rel_parent = merged_path.parent.relative_to(merged_dir)
        out_parent = output_dir / rel_parent
        utt_path = out_parent / f"{rec_id}_utterances.csv"
        manifest_path = out_parent / f"{rec_id}_segment_manifest.json"

        prefix = f"[{i}/{len(merged_csvs)}] {rec_id}"

        if utt_path.exists() and manifest_path.exists() and not args.force:
            n_skipped += 1
            print(f"{prefix} ... skipped (outputs exist)")
            continue

        try:
            utterances, manifest = process_recording(
                merged_path, args.pause_threshold_s, args.max_utterance_s,
                args.report_long_utterance_s)

            out_parent.mkdir(parents=True, exist_ok=True)
            with open(utt_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "utterance_id", "speaker", "start_s", "end_s",
                    "duration_s", "n_words", "text",
                    "mean_word_probability", "min_word_probability",
                    "contains_overlap", "overlap_word_count",
                    "unassigned_word_count", "gap_before_s",
                ])
                writer.writeheader()
                writer.writerows(utterances)

            manifest["recording_id"] = rec_id
            manifest["timestamp_utc"] = datetime.now(timezone.utc) \
                .isoformat(timespec="seconds")
            manifest["merged_csv"] = str(merged_path)
            manifest["outputs"] = {
                "utterances_csv": str(utt_path),
                "manifest": str(manifest_path),
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            n_done += 1
            note = ""
            if manifest.get("n_long_utterances"):
                note = (f"  ** {manifest['n_long_utterances']} utterance(s) "
                        f"over {args.report_long_utterance_s}s **")
                flagged.append((rec_id, manifest["n_long_utterances"]))
            print(f"{prefix} ... done "
                  f"({manifest['n_utterances']} utterances, "
                  f"mean {manifest['mean_utterance_duration_s']}s, "
                  f"max {manifest['max_utterance_duration_s']}s){note}")

        except Exception as e:
            n_failed += 1
            errors.append((rec_id, str(e)))
            print(f"{prefix} ... FAILED: {e}")

    print(f"\nSummary: {n_done} segmented, {n_skipped} skipped, "
          f"{n_failed} failed.")
    if flagged:
        print("Recordings with long utterances (check pause threshold / "
              "consider --max-utterance-s):")
        for rec_id, n in flagged:
            print(f"  {rec_id}: {n} long utterance(s)")
    if errors:
        err_path = output_dir / "segment_errors.csv"
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