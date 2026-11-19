#!/usr/bin/env python
"""
00_extract_audio.py

Step 1 of the raw-media pipeline: extract audio from video files, and
assemble a per-recording corpus folder.

For every video found (recursively) under the input directory, this
produces THREE colocated files in a single output folder named after
the recording (mirroring the video's own containing folder):

    <output_dir>/<relative/path>/<stem>.wav        -- archival copy:
        native sample rate and channel layout, 16-bit PCM. Never
        downsampled; this is what later acoustic analysis (formants
        etc.) reads.
    <output_dir>/<relative/path>/<stem>_16k.wav    -- ASR/diarization
        working copy: mono, 16 kHz, 16-bit PCM.
    <output_dir>/<relative/path>/<stem>.mp4 (etc.) -- the ORIGINAL
        video file, MOVED (not copied) into the same folder.

Every later pipeline stage (01_diarize.py .. 05_generate_eaf.py)
already writes its own output into "<output_dir>/<relative/path>/..."
with no extra subfolder of its own. So pointing every stage at this
SAME output directory means all artefacts for a recording -- video,
both WAVs, diarization, transcription, merged/utterance CSVs, the
final .eaf, and every stage's manifest -- accumulate in one flat,
self-contained, shareable folder per recording. That's the whole
trick; no separate "consolidation" step is needed.

The video is MOVED, not copied: the output folder becomes the single
canonical location for that recording's video from this point on. Run
this only once per video for that reason -- there is no second copy
left behind in the input tree to reprocess.

Because the video is moved, idempotency is checked directly against
the two WAVs and the moved video, not against the (increasingly empty)
input tree:
  * If both WAVs and the moved video already exist, the recording is
    skipped entirely.
  * If the WAVs exist but a previous run was interrupted before the
    move completed, only the move is (re)attempted -- ffmpeg is not
    re-run.
  * --force re-extracts the WAVs even if they exist. If the source
    video has already been moved into the output folder from a prior
    run, --force reads from its new location automatically.

Output layout (single tree, one folder per recording):

    <output_dir>/
        <relative/path>/<stem>.wav
        <relative/path>/<stem>_16k.wav
        <relative/path>/<stem>.mp4              (moved original video)
        extraction_manifest.csv                 (corpus-level rollup)
        extraction_errors.csv                   (only if there were errors)

Design notes
------------
* Replicability: every successful extraction is logged to the
  corpus-level manifest with the source file's SHA-256 hash,
  ffprobe-reported source properties, the exact ffmpeg commands run,
  the ffmpeg version, and a UTC timestamp. Hashing large videos takes
  a moment; disable with --no-hash if you are iterating.
* Timing is never altered: no silence trimming, no loudness
  processing. The 16 kHz copy therefore stays sample-accurate in time
  with the archival copy (modulo resampling).
* Only the FIRST audio stream (-map 0:a:0) is extracted. If a source
  file has multiple audio streams (e.g. camera + lapel mic tracks),
  this is flagged in the manifest so you can handle it deliberately.

Usage
-----
    python 00_extract_audio.py <input_dir> <output_dir> [options]

    python 00_extract_audio.py D:/corpus/video D:/2025_corpus
    python 00_extract_audio.py D:/corpus/video D:/2025_corpus --force
    python 00_extract_audio.py D:/corpus/video D:/2025_corpus --no-hash
    python 00_extract_audio.py D:/corpus/video D:/2025_corpus --move move

Requires: ffmpeg and ffprobe on PATH. No third-party Python packages.
"""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

VIDEO_EXTENSIONS = {
    ".mp4", ".m4v", ".mov", ".mkv", ".avi", ".wmv",
    ".mts", ".m2ts", ".mpg", ".mpeg", ".webm", ".flv", ".3gp",
}

ASR_SUFFIX = "_16k"
ASR_SAMPLE_RATE = 16000

MANIFEST_NAME = "extraction_manifest.csv"
ERRORS_NAME = "extraction_errors.csv"

MANIFEST_FIELDS = [
    "timestamp_utc",
    "video_original_path",     # where the source video was found
    "video_path",              # final (moved) location, colocated with WAVs
    "source_sha256",
    "source_duration_s",
    "source_codec",
    "source_sample_rate",
    "source_channels",
    "n_audio_streams",
    "multiple_audio_streams",  # "yes" flags files needing manual attention
    "archival_path",
    "archival_sample_rate",
    "asr_path",
    "asr_sample_rate",
    "ffmpeg_version",
    "ffmpeg_cmd_archival",
    "ffmpeg_cmd_asr",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd):
    """Run a subprocess, returning (returncode, stdout, stderr) as text."""
    proc = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    return proc.returncode, proc.stdout, proc.stderr


def ffmpeg_version():
    rc, out, _ = run(["ffmpeg", "-version"])
    if rc == 0 and out:
        return out.splitlines()[0].strip()
    return "unknown"


def ffprobe_audio_info(path):
    """Return (info_dict, n_audio_streams) for the first audio stream."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries",
        "stream=index,codec_name,sample_rate,channels:format=duration",
        "-of", "json",
        str(path),
    ]
    rc, out, err = run(cmd)
    if rc != 0:
        raise RuntimeError(f"ffprobe failed: {err.strip()}")
    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("no audio stream found in file")
    first = streams[0]
    info = {
        "codec": first.get("codec_name", ""),
        "sample_rate": first.get("sample_rate", ""),
        "channels": first.get("channels", ""),
        "duration": data.get("format", {}).get("duration", ""),
    }
    return info, len(streams)


def sha256_of_file(path, chunk_size=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def wav_sample_rate(path):
    """Sample rate of an output WAV (for manifest verification)."""
    try:
        info, _ = ffprobe_audio_info(path)
        return info["sample_rate"]
    except Exception:
        return ""


def discover_recordings(input_dir, output_dir):
    """
    Build the list of recordings to process, whether their video is still
    sitting in the input tree (not yet moved) or has already been moved
    into the output tree by a previous run (needed so --force can still
    find and re-extract from it).

    Returns a sorted list of (rel_parent, stem, suffix, source_path).
    Identity is (relative parent folder, filename stem, extension); if a
    video happens to exist in both trees under the same identity, the
    input-tree copy wins (it's the one that hasn't been moved yet).
    """
    recordings = {}

    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
            rel_parent = p.relative_to(input_dir).parent
            key = (str(rel_parent), p.stem, p.suffix.lower())
            recordings[key] = p

    if output_dir.is_dir():
        for p in sorted(output_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS:
                rel_parent = p.relative_to(output_dir).parent
                key = (str(rel_parent), p.stem, p.suffix.lower())
                recordings.setdefault(key, p)   # don't override a pending input-tree copy

    return sorted(
        (Path(rel_parent_str), stem, suffix, source_path)
        for (rel_parent_str, stem, suffix), source_path in recordings.items()
    )


def extract(cmd_desc, cmd, out_path):
    """Run an ffmpeg extraction; raise with stderr detail on failure."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rc, _, err = run(cmd)
    if rc != 0:
        if out_path.exists():
            out_path.unlink()
        raise RuntimeError(f"{cmd_desc} failed: {err.strip()[-500:]}")


# ---------------------------------------------------------------------------
# Main per-file processing
# ---------------------------------------------------------------------------

def process_video(rel_parent, stem, suffix, source, output_dir,
                  do_hash, force, ff_ver, move):
    target_dir = output_dir / rel_parent

    archival_path = target_dir / (stem + ".wav")
    asr_path = target_dir / (stem + ASR_SUFFIX + ".wav")
    video_dest = target_dir / (stem + suffix)

    wavs_exist = archival_path.exists() and asr_path.exists()
    video_moved = video_dest.exists()

    if wavs_exist and video_moved and not force:
        return "skipped", None

    if not source.exists():
        raise FileNotFoundError(f"source video not found: {source}")

    info = {"codec": "", "sample_rate": "", "channels": "", "duration": ""}
    n_streams = 0
    src_hash = "not_computed"
    cmd_archival = cmd_asr = None

    if (not wavs_exist) or force:
        info, n_streams = ffprobe_audio_info(source)
        src_hash = sha256_of_file(source) if do_hash else "not_computed"

        cmd_archival = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-map", "0:a:0",
            "-vn",
            "-acodec", "pcm_s16le",
            str(archival_path),
        ]
        cmd_asr = [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(source),
            "-map", "0:a:0",
            "-vn",
            "-ac", "1",
            "-ar", str(ASR_SAMPLE_RATE),
            "-acodec", "pcm_s16le",
            str(asr_path),
        ]
        extract("archival extraction", cmd_archival, archival_path)
        extract("16 kHz extraction", cmd_asr, asr_path)

    # Move the video in now that both WAVs are confirmed present.
    if source != video_dest:
        if move == 'move':
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(video_dest))
        elif move == 'copy':
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(source), str(video_dest))

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_original_path": str(source),
        "video_path": str(video_dest),
        "source_sha256": src_hash,
        "source_duration_s": info["duration"],
        "source_codec": info["codec"],
        "source_sample_rate": info["sample_rate"],
        "source_channels": info["channels"],
        "n_audio_streams": n_streams,
        "multiple_audio_streams": "yes" if n_streams > 1 else "no",
        "archival_path": str(archival_path),
        "archival_sample_rate": wav_sample_rate(archival_path),
        "asr_path": str(asr_path),
        "asr_sample_rate": wav_sample_rate(asr_path),
        "ffmpeg_version": ff_ver,
        "ffmpeg_cmd_archival": " ".join(cmd_archival) if cmd_archival else "(not re-run; WAVs already existed)",
        "ffmpeg_cmd_asr": " ".join(cmd_asr) if cmd_asr else "(not re-run; WAVs already existed)",
    }
    return "done", row


# ---------------------------------------------------------------------------
# Manifest / error CSV handling
# ---------------------------------------------------------------------------

def append_rows(csv_path, fieldnames, rows):
    if not rows:
        return
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Extract archival + 16 kHz mono WAVs from all videos "
                    "under a directory, and move each video into a "
                    "per-recording corpus folder alongside its WAVs.")
    ap.add_argument("input_dir", type=Path, help="Directory containing video files")
    ap.add_argument("output_dir", type=Path,
                    help="Corpus root directory (one subfolder per recording "
                         "will be created here)")
    ap.add_argument("--force", action="store_true",
                    help="Re-extract WAVs even if outputs already exist")
    ap.add_argument("--no-hash", action="store_true",
                    help="Skip SHA-256 hashing of source videos (faster)")
    ap.add_argument("--move", choices=['move', 'copy', 'no'], default='copy',
                help="Move video files to output directory? ('move', 'copy', 'no', default: %(default)s)")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        sys.exit(f"ERROR: input directory not found: {input_dir}")

    recordings = discover_recordings(input_dir, output_dir)
    if not recordings:
        sys.exit(f"ERROR: no video files found under {input_dir} or "
                 f"{output_dir} (looked for: "
                 f"{', '.join(sorted(VIDEO_EXTENSIONS))})")

    output_dir.mkdir(parents=True, exist_ok=True)
    ff_ver = ffmpeg_version()

    print(f"Found {len(recordings)} recording(s) across {input_dir} "
          f"(pending) and {output_dir} (already moved)")
    print(f"Writing per-recording folders under {output_dir}")
    if args.move == 'move':
        print("NOTE: source videos will be MOVED, not copied.\n")
    elif args.move == 'copy':
        print("NOTE: source videos will be COPIED.\n")
    else:
        print("NOTE: source videos will remain in source directory.\n")

    manifest_rows, error_rows = [], []
    n_done = n_skipped = n_failed = 0

    for i, (rel_parent, stem, suffix, source) in enumerate(recordings, 1):
        prefix = f"[{i}/{len(recordings)}] {rel_parent / (stem + suffix)}"
        try:
            status, row = process_video(
                rel_parent, stem, suffix, source, output_dir,
                do_hash=not args.no_hash, force=args.force, ff_ver=ff_ver, move=args.move,
            )
            if status == "skipped":
                n_skipped += 1
                print(f"{prefix} ... skipped (outputs exist)")
            else:
                n_done += 1
                manifest_rows.append(row)
                note = ""
                if row["multiple_audio_streams"] == "yes":
                    note = (f"  ** {row['n_audio_streams']} audio streams -- "
                            f"only the first was extracted **")
                print(f"{prefix} ... done{note}")
        except Exception as e:
            n_failed += 1
            error_rows.append({
                "timestamp_utc": datetime.now(timezone.utc)
                                 .isoformat(timespec="seconds"),
                "source_path": str(source),
                "error": str(e),
            })
            print(f"{prefix} ... FAILED: {e}")

    append_rows(output_dir / MANIFEST_NAME, MANIFEST_FIELDS, manifest_rows)
    append_rows(output_dir / ERRORS_NAME,
                ["timestamp_utc", "source_path", "error"], error_rows)

    print(f"\nSummary: {n_done} extracted, {n_skipped} skipped, "
          f"{n_failed} failed.")
    if manifest_rows:
        print(f"Manifest: {output_dir / MANIFEST_NAME}")
    if error_rows:
        print(f"Errors:   {output_dir / ERRORS_NAME}")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()