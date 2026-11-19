"""
07_eaf_to_mfa_corpus.py

Convert manually corrected ELAN (.eaf) files into an MFA-ready corpus:
one TextGrid per recording, one interval tier per speaker, tier names
used as speaker IDs. Pairs each TextGrid with its recording's archival
wav.

Expected input layout -- one consolidated corpus root, one folder per
recording, everything colocated (the output of the raw-to-ELAN
pipeline, 00_extract_audio.py through 05_generate_eaf.py):

    2025_corpus/sp-01_sp-02/sp-01_sp-02.eaf         <- hand-corrected
    2025_corpus/sp-01_sp-02/sp-01_sp-02.wav         <- archival (used here)
    2025_corpus/sp-01_sp-02/sp-01_sp-02_16k.wav     <- ASR copy (ignored here)
    2025_corpus/sp-01_sp-02/sp-01_sp-02.mp4         <- video (ignored here)
    2025_corpus/sp-01_sp-02/... (manifests, CSVs -- all ignored here)

The archival wav is used (never the 16 kHz ASR copy), matched by
filename stem in the SAME folder as its .eaf -- no separate audio
directory needed, since 05_generate_eaf.py already writes the .eaf
right next to its wav.

Output layout (MFA corpus structure) is unchanged from before -- a
clean, MFA-only staging copy, so MFA never sees the sibling video/CSV/
manifest files sitting in the recording's corpus folder:

    mfa_corpus/sp-01_sp-02/sp-01_sp-02.wav
    mfa_corpus/sp-01_sp-02/sp-01_sp-02.TextGrid

Usage:
    python 07_eaf_to_mfa_corpus.py --corpus 2025_corpus
    python 07_eaf_to_mfa_corpus.py --corpus 2025_corpus --output mfa_corpus --link

--link attempts a hardlink instead of copying the (potentially large)
archival wav into mfa_corpus/, falling back to a copy automatically if
the two directories are on different volumes (hardlinks can't cross
filesystems). Since mfa_corpus/ and mfa_output/ are fully regenerable
from 2025_corpus/, consider keeping them outside your Dropbox-synced
folder entirely -- on local scratch disk or the HPC -- both to avoid
syncing a duplicate copy of every wav, and to make --link actually
effective (a hardlink into a Dropbox-watched folder gets treated as a
brand new file by Dropbox regardless of what the filesystem is doing).

Then:
    08_run_mfa_elan.bat validate
    08_run_mfa_elan.bat align
"""

import re
import argparse
import shutil
import sys
from pathlib import Path

import pympi
from num2words import num2words

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# Tiers to exclude by name (case-insensitive substring match).
# Add anything that isn't verbatim speech: notes, translations, codes, etc.
EXCLUDE_TIER_PATTERNS = ["note", "comment", "translation", "code", "default"]

# Only include top-level tiers (no parent), i.e. the transcription tiers.
# Set to False if your speaker tiers are child tiers.
TOP_LEVEL_TIERS_ONLY = True

# Annotation cleanup: patterns removed from annotation text before alignment.
# Adjust to match your ELAN transcription conventions.
CLEANUP_PATTERNS = [
    r"\((?:laughs?|laughter|coughs?|sighs?|inaudible|unclear)[^)]*\)",
    r"\[[^\]]*\]",          # anything in square brackets, e.g. [overlap]
    r"\bxxx+\b",            # unintelligible-speech placeholder
]
CLEANUP_RE = [re.compile(p, re.IGNORECASE) for p in CLEANUP_PATTERNS]

# Numeral expansion: faster-whisper often writes "9" instead of "nine".
# Converts standalone digit runs (with optional ordinal suffix) to words.
# Deliberately excludes digits adjacent to ':' or '.' (times, decimals) --
# those need special-casing this doesn't attempt, and are left untouched
# rather than risk producing something like "nine:thirty".
NUMERAL_RE = re.compile(r"(?<![:.\d])\b(\d+)(st|nd|rd|th)?\b(?![:.\d])",
                        re.IGNORECASE)


def expand_numerals(text: str) -> str:
    def _replace(m):
        digits, suffix = m.group(1), m.group(2)
        n = int(digits)
        words = num2words(n, to="ordinal") if suffix else num2words(n)
        return words.replace("-", " ")  # "twenty-one" -> "twenty one":
        # MFA/dictionaries tokenize on whitespace, and "twenty" + "one" are
        # separate dictionary entries -- a hyphenated compound token would
        # itself just become a new OOV.
    return NUMERAL_RE.sub(_replace, text)


def clean_annotation(text: str) -> str:
    for rx in CLEANUP_RE:
        text = rx.sub(" ", text)
    text = expand_numerals(text)
    return re.sub(r"\s+", " ", text).strip()


def include_tier(eaf: pympi.Elan.Eaf, tier_name: str) -> bool:
    lower = tier_name.lower()
    if any(p in lower for p in EXCLUDE_TIER_PATTERNS):
        return False
    if TOP_LEVEL_TIERS_ONLY:
        parent = eaf.tiers[tier_name][2].get("PARENT_REF")
        if parent is not None:
            return False
    return True


def convert_one(eaf_path: Path, wav_path: Path, out_dir: Path, use_link: bool) -> tuple[int, int, int]:
    """Convert one EAF; returns (n_tiers, n_intervals, n_with_leftover_digits)."""
    eaf = pympi.Elan.Eaf(str(eaf_path))

    # Determine total duration from the wav (TextGrid xmax must cover audio).
    import wave
    with wave.open(str(wav_path), "rb") as w:
        duration = w.getnframes() / w.getframerate()

    tg = pympi.Praat.TextGrid(xmax=duration)

    n_tiers = 0
    n_intervals = 0
    n_leftover_digits = 0
    for tier_name in eaf.get_tier_names():
        if not include_tier(eaf, tier_name):
            print(f"    skipping tier: {tier_name}")
            continue
        annotations = eaf.get_annotation_data_for_tier(tier_name)
        if not annotations:
            print(f"    skipping empty tier: {tier_name}")
            continue

        # Sanitise tier name -> speaker ID (no spaces; MFA-friendly).
        speaker = re.sub(r"\s+", "_", tier_name.strip())
        praat_tier = tg.add_tier(speaker)

        for start_ms, end_ms, value in sorted(annotations):
            text = clean_annotation(value or "")
            if not text:
                continue
            if re.search(r"\d", text):
                n_leftover_digits += 1
            start, end = start_ms / 1000.0, end_ms / 1000.0
            if end > duration:
                end = duration
            if end - start <= 0.01:
                continue
            praat_tier.add_interval(start, end, text)
            n_intervals += 1
        n_tiers += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    tg_path = out_dir / f"{wav_path.stem}.TextGrid"
    tg.to_file(str(tg_path))

    dest_wav = out_dir / wav_path.name
    if not dest_wav.exists():
        if use_link:
            try:
                import os
                os.link(wav_path, dest_wav)
            except OSError:
                # Cross-device (common if mfa_corpus and 2025_corpus live on
                # different drives) -- fall back to a real copy silently.
                shutil.copy2(wav_path, dest_wav)
        else:
            shutil.copy2(wav_path, dest_wav)

    return n_tiers, n_intervals, n_leftover_digits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert hand-corrected ELAN files into an MFA-ready corpus")
    parser.add_argument("--corpus", required=True,
                        help="Consolidated corpus root (one folder per "
                             "recording, containing the .eaf and its "
                             "archival .wav together)")
    parser.add_argument("--output", default="mfa_corpus",
                        help="MFA-ready output directory (default: %(default)s)")
    parser.add_argument("--link", action="store_true",
                        help="Hardlink the archival wav into the output "
                             "directory instead of copying it (falls back "
                             "to a copy automatically if --corpus and "
                             "--output are on different drives)")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate the TextGrid even if it already exists")
    args = parser.parse_args()

    CORPUS_DIR = Path(args.corpus)      # 2025_corpus root, one folder per recording
    OUT_DIR = Path(args.output)         # MFA corpus root (one subdir per recording)

    if not CORPUS_DIR.is_dir():
        sys.exit(f"ERROR: corpus directory not found: {CORPUS_DIR.resolve()}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Recordings may be one level deep (as produced by the raw-to-ELAN
    # pipeline) or nested further -- search recursively either way.
    # Only get the manually corrected versions
    eaf_files = sorted([f for f in CORPUS_DIR.rglob("*.eaf") if re.search(r"_corrected", f.stem)])
    if not eaf_files:
        sys.exit(f"No .eaf files found under {CORPUS_DIR.resolve()}. Check if files have '_corrected' in name.")

    failed = []
    for eaf_path in eaf_files:
        # Simplify the stem to match with .wav file
        stem = eaf_path.stem.replace("_corrected", "")
        # The archival wav sits right next to its .eaf, same stem, no
        # suffix (never the "_16k" ASR working copy).
        wav_path = eaf_path.with_name(f"{stem}.wav")
        out_dir = OUT_DIR / eaf_path.parent.stem

        if (out_dir / f"{stem}.TextGrid").exists() and not args.force:
            print(f"[skip] {stem}: TextGrid already exists")
            continue
        if not wav_path.exists():
            print(f"[FAIL] {stem}: no matching archival wav at {wav_path}")
            failed.append(stem)
            continue

        print(f"[conv] {stem}")
        try:
            n_tiers, n_ints, n_digits = convert_one(eaf_path, wav_path, out_dir, args.link)
            print(f"       {n_tiers} speaker tiers, {n_ints} intervals")
            if n_tiers < 2:
                print(f"       WARNING: fewer than 2 speaker tiers found")
            if n_digits:
                print(f"       NOTE: {n_digits} interval(s) still contain digits "
                     f"(likely times/decimals/glued forms like '9am' -- "
                     f"expand_numerals() deliberately leaves these untouched)")
        except Exception as e:
            print(f"[FAIL] {stem}: {e}")
            failed.append(stem)

    print("\nDone.")
    if failed:
        print("Failed recordings:")
        for f in failed:
            print(f"  - {f}")


if __name__ == "__main__":
    main()
