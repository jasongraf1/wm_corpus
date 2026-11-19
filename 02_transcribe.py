#!/usr/bin/env python
"""
02_transcribe.py

Step 3 of the raw-media pipeline: ASR with faster-whisper, with
word-level timestamps for the later speaker-merge step.

Runs faster-whisper over every 16 kHz WAV found (recursively) under the
input directory (the ``audio_16k/`` tree from ``00_extract_audio.py``)
and saves, per recording:

    transcription/<relative/path>/<recording_id>_whisper.json
        Full raw output: every segment with its word list (word, start,
        end, probability), plus detection info. Canonical artefact --
        the merge step reads this.

    transcription/<relative/path>/<recording_id>_words.csv
        Flat word table: word, start_s, end_s, probability, segment_id.
        Convenient for inspection and downstream merging.

    transcription/<relative/path>/<recording_id>_transcribe_manifest.json
        Model, versions, device, all decoding parameters, runtime.

Design decisions (see comments inline):
  * word_timestamps=True     -- required by the merge step.
  * vad_filter=True          -- suppresses Whisper's tendency to
                                hallucinate text in long silences
                                (thinking pauses are common in
                                sociolinguistic interviews).
  * temperature=0, fixed beam -- deterministic decoding for
                                replicability. Whisper's default
                                temperature *fallback* is stochastic;
                                with it disabled you may occasionally
                                see repetition loops, which manual
                                correction will catch.
  * condition_on_previous_text=False (default here)
                             -- prevents a hallucination early in the
                                file from propagating; each ~30 s
                                window is decoded independently. Set
                                --condition-on-previous to re-enable.

Usage
-----
    conda activate socio
    python 02_transcribe.py <audio_16k_dir> <output_dir> [options]

    python 02_transcribe.py D:/corpus/audio/audio_16k D:/corpus/transcription
    python 02_transcribe.py IN OUT --model large-v3 --device cuda
    python 02_transcribe.py IN OUT --prompt-file wm_prompt.txt

Requires: pip install faster-whisper
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ASR_SUFFIX = "_16k"
MANIFEST_SUFFIX = "_transcribe_manifest.json"


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
    '*.wav' would pick up both and attempt to transcribe the archival
    copy too. Only the 16 kHz copy should be fed to faster-whisper.
    """
    return [p for p in sorted(input_dir.rglob(f"*{ASR_SUFFIX}.wav")) if p.is_file()]


def main():
    ap = argparse.ArgumentParser(
        description="Transcribe all WAVs under a directory with "
                    "faster-whisper (word-level timestamps).")
    ap.add_argument("input_dir", type=Path,
                    help="Directory of 16 kHz WAVs (searched recursively)")
    ap.add_argument("output_dir", type=Path,
                    help="Directory for transcription outputs")
    ap.add_argument("--model", default="large-v3",
                    help="Whisper model size or path (default: %(default)s). "
                         "For faster CPU runs try 'medium.en' or "
                         "'distil-large-v3'.")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="Computation device (default: %(default)s)")
    ap.add_argument("--compute-type", default="default",
                    help="ctranslate2 compute type, e.g. float16, int8, "
                         "int8_float16 (default: %(default)s)")
    ap.add_argument("--language", default="en",
                    help="Language code (default: %(default)s)")
    ap.add_argument("--beam-size", type=int, default=5,
                    help="Beam size (default: %(default)s)")
    ap.add_argument("--prompt-file", type=Path, default=None,
                    help="UTF-8 text file whose contents are passed as "
                         "initial_prompt (e.g. local place names/spellings)")
    ap.add_argument("--condition-on-previous", action="store_true",
                    help="Enable condition_on_previous_text (default: off, "
                         "to stop hallucinations propagating)")
    ap.add_argument("--min-silence-ms", type=int, default=500,
                    help="VAD min_silence_duration_ms (default: %(default)s)")
    ap.add_argument("--no-vad", action="store_true",
                    help="Disable the VAD filter (not recommended)")
    ap.add_argument("--force", action="store_true",
                    help="Re-transcribe even if outputs already exist")
    args = ap.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.is_dir():
        sys.exit(f"ERROR: input directory not found: {input_dir}")

    wavs = find_wavs(input_dir)
    if not wavs:
        sys.exit(f"ERROR: no .wav files found under {input_dir}")

    initial_prompt = None
    if args.prompt_file is not None:
        if not args.prompt_file.is_file():
            sys.exit(f"ERROR: prompt file not found: {args.prompt_file}")
        initial_prompt = args.prompt_file.read_text(
            encoding="utf-8-sig").strip() or None

    # Heavy imports deferred until after argument validation.
    import faster_whisper
    from faster_whisper import WhisperModel

    if args.device == "auto":
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device

    print(f"faster-whisper {faster_whisper.__version__} | model: "
          f"{args.model} | device: {device} | compute: {args.compute_type}")
    print("Loading model (first run downloads weights) ...")
    model = WhisperModel(args.model, device=device,
                         compute_type=args.compute_type)

    transcribe_kwargs = {
        "language": args.language,
        "beam_size": args.beam_size,
        "temperature": 0.0,                      # deterministic decoding
        "condition_on_previous_text": args.condition_on_previous,
        "word_timestamps": True,                 # required for merge step
        "initial_prompt": initial_prompt,
        "vad_filter": not args.no_vad,
        "vad_parameters": {"min_silence_duration_ms": args.min_silence_ms},
    }

    print(f"Found {len(wavs)} WAV(s) under {input_dir}")
    print(f"Decoding parameters: { {k: v for k, v in transcribe_kwargs.items() if k != 'initial_prompt'} }")
    if initial_prompt:
        print(f"Initial prompt ({len(initial_prompt)} chars) loaded from "
              f"{args.prompt_file}")
    print()

    n_done = n_skipped = n_failed = 0
    errors = []

    for i, wav in enumerate(wavs, 1):
        rec_id = recording_id(wav)
        rel_parent = wav.parent.relative_to(input_dir)
        out_parent = output_dir / rel_parent
        json_path = out_parent / f"{rec_id}_whisper.json"
        words_path = out_parent / f"{rec_id}_words.csv"
        manifest_path = out_parent / f"{rec_id}{MANIFEST_SUFFIX}"

        prefix = f"[{i}/{len(wavs)}] {rec_id}"

        if json_path.exists() and manifest_path.exists() and not args.force:
            n_skipped += 1
            print(f"{prefix} ... skipped (outputs exist)")
            continue

        try:
            t0 = time.monotonic()
            segments_iter, info = model.transcribe(str(wav),
                                                   **transcribe_kwargs)

            segments = []
            word_rows = []
            for seg_idx, seg in enumerate(segments_iter):
                seg_words = []
                for w in (seg.words or []):
                    word = {
                        "word": w.word.strip(),
                        "start_s": round(w.start, 4),
                        "end_s": round(w.end, 4),
                        "probability": round(w.probability, 4),
                    }
                    seg_words.append(word)
                    word_rows.append({**word, "segment_id": seg_idx})
                segments.append({
                    "segment_id": seg_idx,
                    "start_s": round(seg.start, 4),
                    "end_s": round(seg.end, 4),
                    "text": seg.text.strip(),
                    "avg_logprob": round(seg.avg_logprob, 4),
                    "no_speech_prob": round(seg.no_speech_prob, 4),
                    "words": seg_words,
                })
            runtime = round(time.monotonic() - t0, 1)

            out_parent.mkdir(parents=True, exist_ok=True)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "recording_id": rec_id,
                    "input_wav": str(wav),
                    "language": info.language,
                    "language_probability": round(info.language_probability, 4),
                    "duration_s": round(info.duration, 3),
                    "segments": segments,
                }, f, indent=2, ensure_ascii=False)

            with open(words_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["word", "start_s", "end_s",
                                   "probability", "segment_id"])
                writer.writeheader()
                writer.writerows(word_rows)

            manifest = {
                "timestamp_utc": datetime.now(timezone.utc)
                                 .isoformat(timespec="seconds"),
                "recording_id": rec_id,
                "input_wav": str(wav),
                "model": args.model,
                "faster_whisper_version": faster_whisper.__version__,
                "device": device,
                "compute_type": args.compute_type,
                "parameters": {k: v for k, v in transcribe_kwargs.items()
                               if k != "initial_prompt"},
                "initial_prompt_used": initial_prompt is not None,
                "initial_prompt_file": (str(args.prompt_file)
                                        if initial_prompt else None),
                "runtime_s": runtime,
                "audio_duration_s": round(info.duration, 3),
                "n_segments": len(segments),
                "n_words": len(word_rows),
                "outputs": {
                    "whisper_json": str(json_path),
                    "words_csv": str(words_path),
                },
            }
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            n_done += 1
            speed = (info.duration / runtime) if runtime else 0
            print(f"{prefix} ... done in {runtime}s "
                  f"({len(word_rows)} words, {speed:.1f}x real-time)")

        except Exception as e:
            n_failed += 1
            errors.append((rec_id, str(e)))
            print(f"{prefix} ... FAILED: {e}")

    print(f"\nSummary: {n_done} transcribed, {n_skipped} skipped, "
          f"{n_failed} failed.")
    if errors:
        err_path = output_dir / "transcription_errors.csv"
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