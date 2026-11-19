#!/usr/bin/env python
"""
05_generate_eaf.py

Step 6 (final) of the raw-media pipeline: turn utterance-level CSVs
(output of 04_segment_utterances.py) into ELAN .eaf files, one tier per
speaker, with the archival audio (and optionally the source video)
linked as media, ready for manual correction.

Tier naming
-----------
There is no way to automatically know which diarized voice belongs to
which real participant -- that's a fact recoverable only from consent
forms / fieldwork records, not from the audio. So by default, tiers are
named after the raw diarization labels (SPEAKER_00, SPEAKER_01, ...).

If you *have* worked out the real identity for some recordings, pass
--speaker-map pointing at a CSV with columns:

    recording_id,diarized_speaker,tier_name

One row per (recording, diarized label) you want renamed. Recordings or
speakers not listed simply keep their raw diarized label as the tier
name -- there is no penalty for leaving this file partial or empty, and
no requirement to fill it in before running this script. Add rows
whenever you've verified a recording; re-run with --force to relabel.

Media linking
-------------
Link video, audio, or both -- at least one of --video-dir / --audio-dir
is required. If both are given, audio (the archival, native-rate WAV --
never the 16 kHz working copy) is linked first as the primary
descriptor and video second; ELAN treats the first descriptor as
primary for waveform generation. Linking the archival WAV alongside
video buys uncompressed audio (video containers are usually lossy AAC)
and typically snappier waveform scrubbing, at the cost of a second
media path to keep valid -- video-only is a perfectly reasonable
choice if that trade-off isn't worth it for your workflow.

Media descriptors are stored BOTH as an absolute file:// URI (MEDIA_URL)
and, when possible, as a path relative to the .eaf file's own location
(RELATIVE_MEDIA_URL). ELAN tries the relative path first, so the corpus
can be moved or synced (e.g. via Dropbox onto a different machine)
without breaking media links, as long as the relative layout between
the .eaf and its media is preserved.

Outputs (per recording, mirrored directory structure):

    eaf/<relative/path>/<rec_id>_init.eaf
    eaf/<relative/path>/<rec_id>_eaf_manifest.json
        tiers created, tier-name source (raw/mapped), media linked,
        n_annotations per tier, any skipped (zero-duration) annotations.

Usage
-----
    conda activate socio
    python 05_generate_eaf.py <utterances_dir> <output_dir> [options]

    # video only
    python 05_generate_eaf.py D:/corpus/utterances D:/corpus/eaf --video-dir D:/corpus/video

    # audio only
    python 05_generate_eaf.py D:/corpus/utterances D:/corpus/eaf --audio-dir D:/corpus/audio/audio_archival

    # both (audio linked first/primary)
    python 05_generate_eaf.py IN_U OUT --audio-dir D:/corpus/audio/audio_archival --video-dir D:/corpus/video

    python 05_generate_eaf.py IN_U OUT --video-dir D:/corpus/video --speaker-map speaker_map.csv

Requires: pip install pympi-ling
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

UTTERANCES_SUFFIX = "_utterances.csv"

VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".wmv",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".webm", ".flv", ".3gp",
}

# pympi only auto-guesses wav/mpg/mpeg/mp4/xml; supply the rest explicitly.
MIME_TYPES = {
    ".wav": "audio/x-wav",
    ".mp4": "video/mp4",
    ".m4v": "video/x-m4v",
    ".mov": "video/quicktime",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".wmv": "video/x-ms-wmv",
    ".mts": "video/mp2t",
    ".m2ts": "video/mp2t",
    ".mpg": "video/mpeg",
    ".mpeg": "video/mpeg",
    ".webm": "video/webm",
    ".flv": "video/x-flv",
    ".3gp": "video/3gpp",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def find_utterance_csvs(utterances_dir: Path):
    return sorted(p for p in utterances_dir.rglob(f"*{UTTERANCES_SUFFIX}")
                  if p.is_file())


def rec_id_from_utterances_path(path: Path) -> str:
    name = path.name
    assert name.endswith(UTTERANCES_SUFFIX)
    return name[: -len(UTTERANCES_SUFFIX)]


def load_utterances_csv(path: Path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "utterance_id": r["utterance_id"],
                "speaker": r["speaker"],
                "start_s": float(r["start_s"]),
                "end_s": float(r["end_s"]),
                "text": r["text"],
            })
    rows.sort(key=lambda u: u["start_s"])
    return rows


def load_speaker_map(path: Path):
    """Returns {(recording_id, diarized_speaker): tier_name}."""
    mapping = {}
    if path is None:
        return mapping
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["recording_id"], r["diarized_speaker"])
            mapping[key] = r["tier_name"]
    return mapping


# ---------------------------------------------------------------------------
# Media discovery
# ---------------------------------------------------------------------------

def find_video_file(video_dir: Path, rel_parent: Path, rec_id: str):
    if video_dir is None:
        return None
    search_dir = video_dir / rel_parent
    if not search_dir.is_dir():
        return None
    matches = [p for p in search_dir.glob(f"{rec_id}.*")
              if p.suffix.lower() in VIDEO_EXTENSIONS]
    return matches[0] if matches else None


def relative_media_url(media_path: Path, eaf_dir: Path):
    """Forward-slash relative path from the .eaf's directory to the media."""
    try:
        import os
        rel = os.path.relpath(media_path, start=eaf_dir)
        return rel.replace("\\", "/")
    except ValueError:
        # Different drive on Windows -- no relative path possible.
        return None


# ---------------------------------------------------------------------------
# Per-recording processing
# ---------------------------------------------------------------------------

def process_recording(utterances_path, audio_dir, video_dir, output_dir,
                      utterances_dir, speaker_map, rec_id):
    from pympi.Elan import Eaf

    utterances = load_utterances_csv(utterances_path)
    rel_parent = utterances_path.parent.relative_to(utterances_dir)

    eaf = Eaf()
    eaf.remove_tier("default")

    # --- media ---
    eaf_out_dir = output_dir / rel_parent
    media_notes = {"audio_requested": audio_dir is not None,
                  "audio_linked": False, "video_linked": False}

    if audio_dir is not None:
        audio_path = audio_dir / rel_parent / f"{rec_id}.wav"
        if audio_path.is_file():
            abs_uri = audio_path.resolve().as_uri()
            rel_uri = relative_media_url(audio_path.resolve(), eaf_out_dir)
            eaf.add_linked_file(abs_uri, relpath=rel_uri,
                                mimetype=MIME_TYPES[".wav"])
            media_notes["audio_linked"] = True
            media_notes["audio_path"] = str(audio_path)
        else:
            media_notes["audio_path"] = str(audio_path) + " (NOT FOUND)"

    video_path = find_video_file(video_dir, rel_parent, rec_id)
    if video_path is not None:
        abs_uri = video_path.resolve().as_uri()
        rel_uri = relative_media_url(video_path.resolve(), eaf_out_dir)
        eaf.add_linked_file(abs_uri, relpath=rel_uri,
                            mimetype=MIME_TYPES[video_path.suffix.lower()])
        media_notes["video_linked"] = True
        media_notes["video_path"] = str(video_path)

    # --- tiers ---
    raw_speakers = sorted({u["speaker"] for u in utterances})
    tier_name_of = {}
    tier_source = {}
    for raw in raw_speakers:
        mapped = speaker_map.get((rec_id, raw))
        tier_name_of[raw] = mapped if mapped else raw
        tier_source[raw] = "mapped" if mapped else "raw_diarized_label"

    # Guard against two raw labels resolving to the same tier name.
    seen_names = {}
    for raw, name in tier_name_of.items():
        seen_names.setdefault(name, []).append(raw)
    collisions = {name: raws for name, raws in seen_names.items() if len(raws) > 1}
    if collisions:
        raise ValueError(
            f"speaker_map collision for {rec_id}: multiple diarized "
            f"labels map to the same tier name: {collisions}")

    for raw, name in tier_name_of.items():
        eaf.add_tier(name, part=name)

    # --- annotations ---
    n_annotations = {name: 0 for name in tier_name_of.values()}
    n_skipped_zero_duration = 0

    for u in utterances:
        tier = tier_name_of[u["speaker"]]
        start_ms = round(u["start_s"] * 1000)
        end_ms = round(u["end_s"] * 1000)
        if end_ms <= start_ms:
            n_skipped_zero_duration += 1
            continue
        eaf.add_annotation(tier, start_ms, end_ms, value=u["text"])
        n_annotations[tier] += 1

    manifest = {
        "recording_id": rec_id,
        "tiers": [
            {"tier_name": tier_name_of[raw], "diarized_label": raw,
            "source": tier_source[raw], "n_annotations": n_annotations[tier_name_of[raw]]}
            for raw in raw_speakers
        ],
        "n_utterances_total": len(utterances),
        "n_skipped_zero_duration": n_skipped_zero_duration,
        "media": media_notes,
    }
    return eaf, manifest


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate one-tier-per-speaker ELAN .eaf files from "
                    "utterance-level CSVs.")
    ap.add_argument("utterances_dir", type=Path,
                    help="Directory of *_utterances.csv files (recursive)")
    ap.add_argument("output_dir", type=Path, help="Directory for .eaf output")
    ap.add_argument("--audio-dir", type=Path, default=None,
                    help="Archival (native-rate) audio directory, mirroring "
                         "the same relative structure. Linked first/primary "
                         "if --video-dir is also given.")
    ap.add_argument("--video-dir", type=Path, default=None,
                    help="Source-video directory to link as a media "
                         "descriptor. At least one of --audio-dir / "
                         "--video-dir is required.")
    ap.add_argument("--speaker-map", type=Path, default=None,
                    help="Optional CSV with columns "
                         "recording_id,diarized_speaker,tier_name")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if outputs already exist")
    args = ap.parse_args()

    utterances_dir = args.utterances_dir.resolve()
    output_dir = args.output_dir.resolve()
    audio_dir = args.audio_dir.resolve() if args.audio_dir else None
    video_dir = args.video_dir.resolve() if args.video_dir else None

    if not utterances_dir.is_dir():
        sys.exit(f"ERROR: utterances directory not found: {utterances_dir}")
    if audio_dir is None and video_dir is None:
        sys.exit("ERROR: at least one of --audio-dir / --video-dir is required")
    if audio_dir is not None and not audio_dir.is_dir():
        sys.exit(f"ERROR: audio directory not found: {audio_dir}")
    if video_dir is not None and not video_dir.is_dir():
        sys.exit(f"ERROR: video directory not found: {video_dir}")

    try:
        import pympi  # noqa: F401
    except ImportError:
        sys.exit("ERROR: pympi-ling is not installed. "
                 "Run: pip install pympi-ling --break-system-packages")

    speaker_map = load_speaker_map(args.speaker_map)
    if args.speaker_map:
        print(f"Loaded {len(speaker_map)} speaker mapping(s) from "
              f"{args.speaker_map}")

    utterance_csvs = find_utterance_csvs(utterances_dir)
    if not utterance_csvs:
        sys.exit(f"ERROR: no *{UTTERANCES_SUFFIX} files found under "
                 f"{utterances_dir}")

    print(f"Found {len(utterance_csvs)} utterance file(s) under {utterances_dir}\n")

    n_done = n_skipped = n_failed = 0
    errors = []
    missing_audio = []

    for i, utterances_path in enumerate(utterance_csvs, 1):
        rec_id = rec_id_from_utterances_path(utterances_path)
        rel_parent = utterances_path.parent.relative_to(utterances_dir)
        out_parent = output_dir / rel_parent
        eaf_path = out_parent / f"{rec_id}_init.eaf"
        manifest_path = out_parent / f"{rec_id}_eaf_manifest.json"

        prefix = f"[{i}/{len(utterance_csvs)}] {rec_id}"

        if eaf_path.exists() and manifest_path.exists() and not args.force:
            n_skipped += 1
            print(f"{prefix} ... skipped (outputs exist)")
            continue

        try:
            eaf, manifest = process_recording(
                utterances_path, audio_dir, video_dir, output_dir,
                utterances_dir, speaker_map, rec_id)

            out_parent.mkdir(parents=True, exist_ok=True)
            eaf.to_file(str(eaf_path))

            manifest["timestamp_utc"] = datetime.now(timezone.utc) \
                .isoformat(timespec="seconds")
            manifest["utterances_csv"] = str(utterances_path)
            manifest["outputs"] = {
                "eaf": str(eaf_path), "manifest": str(manifest_path)}
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            n_done += 1
            audio_missing = (manifest["media"]["audio_requested"]
                             and not manifest["media"]["audio_linked"])
            if audio_missing:
                missing_audio.append(rec_id)
            tier_summary = ", ".join(
                f"{t['tier_name']} ({t['n_annotations']})" for t in manifest["tiers"])
            note = "  ** NO AUDIO LINKED (requested but not found) **" if audio_missing else ""
            print(f"{prefix} ... done -- tiers: {tier_summary}{note}")

        except Exception as e:
            n_failed += 1
            errors.append((rec_id, str(e)))
            print(f"{prefix} ... FAILED: {e}")

    print(f"\nSummary: {n_done} generated, {n_skipped} skipped, {n_failed} failed.")
    if missing_audio:
        print(f"Recordings with no archival audio found ({len(missing_audio)}):")
        for rec_id in missing_audio:
            print(f"  {rec_id}")
    if errors:
        err_path = output_dir / "eaf_errors.csv"
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
