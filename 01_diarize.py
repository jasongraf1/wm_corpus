#!/usr/bin/env python
"""
01_diarize.py

Step 2 of the raw-media pipeline: speaker diarization with pyannote.audio.

Runs a pretrained pyannote diarization pipeline over every 16 kHz WAV
found (recursively) under the input directory (the ``audio_16k/`` tree
produced by ``00_extract_audio.py``) and saves, per recording:

    diarization/<relative/path>/<recording_id>.rttm
        Standard RTTM (regular diarization, may contain overlaps).
        Kept as the canonical raw artefact.

    diarization/<relative/path>/<recording_id>_segments.csv
        start_s, end_s, duration_s, speaker  -- regular diarization.

    diarization/<relative/path>/<recording_id>_exclusive.csv
        Same columns -- "exclusive" diarization (pyannote 4.x /
        community-1 only), in which exactly one speaker is active at a
        time. This is the recommended input for assigning ASR word
        timestamps to speakers in the merge step.

    diarization/<relative/path>/<recording_id>_diarize_manifest.json
        Parameters, library/model versions, device, runtime, and
        per-speaker speech-time stats for this run.

Compatibility
-------------
Works with both pyannote.audio 3.x (pipeline call returns an
``Annotation``) and 4.x (returns an output object exposing
``.speaker_diarization`` and ``.exclusive_speaker_diarization``).
Default model is ``pyannote/speaker-diarization-community-1`` (requires
pyannote.audio >= 4.0); pass ``--model pyannote/speaker-diarization-3.1``
to use the legacy pipeline with a 3.x install.

Both models are gated on Hugging Face: accept the user conditions on the
model page once, then supply a token via ``--hf-token`` or the
``HF_TOKEN`` environment variable.

Usage
-----
    conda activate socio
    python 01_diarize.py <audio_16k_dir> <output_dir> [options]

    # dyadic interviews (the default: exactly 2 speakers)
    python 01_diarize.py D:/corpus/audio/audio_16k D:/corpus/diarization

    # let the model decide within bounds instead
    python 01_diarize.py IN OUT --min-speakers 2 --max-speakers 3

    # legacy pipeline
    python 01_diarize.py IN OUT --model pyannote/speaker-diarization-3.1

Notes
-----
* Idempotent: a recording is skipped if its RTTM and manifest already
  exist (override with --force).
* --num-speakers defaults to 2, which is the single biggest accuracy
  win for dyadic interview data. Set --num-speakers 0 to disable the
  constraint (or use --min/--max-speakers).
* Speaker labels (SPEAKER_00, ...) are arbitrary per recording; mapping
  them to real speaker IDs happens later (merge/EAF stage or manually).
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ASR_SUFFIX = "_16k"          # stripped from WAV stems to form recording IDs
MANIFEST_SUFFIX = "_diarize_manifest.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recording_id(wav_path: Path) -> str:
    stem = wav_path.stem
    if stem.endswith(ASR_SUFFIX):
        stem = stem[: -len(ASR_SUFFIX)]
    return stem


def find_wavs(input_dir: Path):
    """
    Find 16 kHz working-copy WAVs specifically (suffix '_16k.wav') -- NOT
    a bare '*.wav' glob. Since 00_extract_audio.py now writes the archival
    WAV and the 16 kHz WAV into the SAME per-recording folder, a bare
    '*.wav' would pick up both and attempt to diarize the archival copy
    too. Only the 16 kHz copy should be fed to pyannote.
    """
    return [p for p in sorted(input_dir.rglob(f"*{ASR_SUFFIX}.wav")) if p.is_file()]


def annotation_to_rows(annotation):
    """Flatten a pyannote Annotation into sorted segment rows."""
    rows = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        rows.append({
            "start_s": round(segment.start, 4),
            "end_s": round(segment.end, 4),
            "duration_s": round(segment.end - segment.start, 4),
            "speaker": str(label),
        })
    rows.sort(key=lambda r: (r["start_s"], r["speaker"]))
    return rows


def write_segments_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["start_s", "end_s", "duration_s", "speaker"])
        writer.writeheader()
        writer.writerows(rows)


def speaker_stats(rows):
    stats = {}
    for r in rows:
        s = stats.setdefault(r["speaker"], {"n_segments": 0, "speech_time_s": 0.0})
        s["n_segments"] += 1
        s["speech_time_s"] = round(s["speech_time_s"] + r["duration_s"], 4)
    return stats


def get_annotations(pipeline_output):
    """
    Return (regular, exclusive) Annotations from a pipeline call,
    handling both pyannote.audio 3.x and 4.x return types.
    """
    if hasattr(pipeline_output, "speaker_diarization"):        # 4.x
        regular = pipeline_output.speaker_diarization
        exclusive = getattr(
            pipeline_output, "exclusive_speaker_diarization", None)
        return regular, exclusive
    return pipeline_output, None                               # 3.x Annotation


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Diarize all WAVs under a directory with pyannote.audio.")
    ap.add_argument("input_dir", type=Path,
                    help="Directory of 16 kHz WAVs (searched recursively)")
    ap.add_argument("output_dir", type=Path,
                    help="Directory for diarization outputs")
    ap.add_argument("--model", default="pyannote/speaker-diarization-community-1",
                    help="Pretrained pipeline name (default: %(default)s)")
    ap.add_argument("--num-speakers", type=int, default=2,
                    help="Exact number of speakers; 0 to disable "
                         "(default: %(default)s)")
    ap.add_argument("--min-speakers", type=int, default=None,
                    help="Lower bound on number of speakers "
                         "(ignored if --num-speakers > 0)")
    ap.add_argument("--max-speakers", type=int, default=None,
                    help="Upper bound on number of speakers "
                         "(ignored if --num-speakers > 0)")
    ap.add_argument("--hf-token", default=None,
                    help="Hugging Face token (default: HF_TOKEN env var)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="Computation device (default: %(default)s)")
    ap.add_argument("--force", action="store_true",
                    help="Re-diarize even if outputs already exist")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        sys.exit(f"ERROR: input directory not found: {input_dir}")

    wavs = find_wavs(input_dir)
    if not wavs:
        sys.exit(f"ERROR: no .wav files found under {input_dir}")

    token = args.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("ERROR: no Hugging Face token. Pass --hf-token or set the "
                 "HF_TOKEN environment variable. (You must also have "
                 f"accepted the gate for {args.model} on huggingface.co.)")

    # Heavy imports deferred until after argument validation.
    import torch
    import pyannote.audio
    from pyannote.audio import Pipeline

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            sys.exit("ERROR: --device cuda requested but CUDA is not available")

    print(f"pyannote.audio {pyannote.audio.__version__} | torch "
          f"{torch.__version__} | device: {device}")
    print(f"Loading pipeline: {args.model} ...")
    pipeline = Pipeline.from_pretrained(args.model, token=token)
    if pipeline is None:
        sys.exit(f"ERROR: could not load {args.model}. Check the model name "
                 "and that you have accepted its user conditions on "
                 "huggingface.co with this token's account.")
    pipeline.to(torch.device(device))

    call_kwargs = {}
    if args.num_speakers and args.num_speakers > 0:
        call_kwargs["num_speakers"] = args.num_speakers
    else:
        if args.min_speakers:
            call_kwargs["min_speakers"] = args.min_speakers
        if args.max_speakers:
            call_kwargs["max_speakers"] = args.max_speakers

    print(f"Found {len(wavs)} WAV(s) under {input_dir}")
    print(f"Speaker constraint: {call_kwargs or 'none'}\n")

    n_done = n_skipped = n_failed = 0
    errors = []

    for i, wav in enumerate(wavs, 1):
        rec_id = recording_id(wav)
        rel_parent = wav.parent.relative_to(input_dir)
        out_parent = output_dir / rel_parent
        rttm_path = out_parent / f"{rec_id}.rttm"
        segments_path = out_parent / f"{rec_id}_segments.csv"
        exclusive_path = out_parent / f"{rec_id}_exclusive.csv"
        manifest_path = out_parent / f"{rec_id}{MANIFEST_SUFFIX}"

        prefix = f"[{i}/{len(wavs)}] {rec_id}"

        if rttm_path.exists() and manifest_path.exists() and not args.force:
            n_skipped += 1
            print(f"{prefix} ... skipped (outputs exist)")
            continue

        try:
            t0 = time.monotonic()
            output = pipeline(str(wav), **call_kwargs)
            runtime = round(time.monotonic() - t0, 1)

            regular, exclusive = get_annotations(output)
            # Name the annotation so the RTTM 'uri' field is the recording ID.
            regular.uri = rec_id

            out_parent.mkdir(parents=True, exist_ok=True)

            with open(rttm_path, "w", encoding="utf-8") as f:
                regular.write_rttm(f)

            rows = annotation_to_rows(regular)
            write_segments_csv(segments_path, rows)

            exclusive_rows = None
            if exclusive is not None:
                exclusive_rows = annotation_to_rows(exclusive)
                write_segments_csv(exclusive_path, exclusive_rows)

            manifest = {
                "timestamp_utc": datetime.now(timezone.utc)
                                 .isoformat(timespec="seconds"),
                "recording_id": rec_id,
                "input_wav": str(wav),
                "model": args.model,
                "pyannote_audio_version": pyannote.audio.__version__,
                "torch_version": torch.__version__,
                "device": device,
                "speaker_constraint": call_kwargs,
                "runtime_s": runtime,
                "n_segments": len(rows),
                "speakers": speaker_stats(rows),
                "exclusive_available": exclusive_rows is not None,
                "outputs": {
                    "rttm": str(rttm_path),
                    "segments_csv": str(segments_path),
                    "exclusive_csv": (str(exclusive_path)
                                      if exclusive_rows is not None else None),
                },
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            n_done += 1
            spk = ", ".join(f"{k}: {v['speech_time_s']:.0f}s"
                            for k, v in sorted(manifest["speakers"].items()))
            print(f"{prefix} ... done in {runtime}s "
                  f"({len(rows)} segments; {spk})")

        except Exception as e:
            n_failed += 1
            errors.append((rec_id, str(e)))
            print(f"{prefix} ... FAILED: {e}")

    print(f"\nSummary: {n_done} diarized, {n_skipped} skipped, "
          f"{n_failed} failed.")
    if errors:
        err_path = output_dir / "diarization_errors.csv"
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