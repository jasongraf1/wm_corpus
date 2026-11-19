#!/usr/bin/env python
"""
07b_eaf_to_txt_transcripts.py

Companion to 07_eaf_to_mfa_corpus.py: exports the SAME verbatim speech
text (same tier-inclusion rules and annotation cleanup) as plain .txt
transcripts, one per recording -- input for a lightweight,
dictionary-based OOV check (00f_check_oov.py) that doesn't depend on
MFA's CLI at all.

Why this exists: MFA's own OOV reporting (mfa validate / mfa find_oovs)
has moved around across MFA versions -- where it writes oovs_found.txt,
whether an --output_directory is required, etc. Checking your
transcripts directly against your dictionary files answers the exact
same question (which words aren't in the dictionary) without needing
MFA's CLI to cooperate.

IMPORTANT -- kept in sync with 07_eaf_to_mfa_corpus.py:
The tier-inclusion rules (EXCLUDE_TIER_PATTERNS, TOP_LEVEL_TIERS_ONLY)
and annotation cleanup (CLEANUP_PATTERNS) below are deliberately
IDENTICAL to those in 07_eaf_to_mfa_corpus.py, so that an OOV check
against these .txt files reflects exactly what MFA will actually see
in the TextGrid. If you ever change the cleanup rules in one script
(e.g. adding a new bracketed-annotation convention), change them in
the other too, or the two will quietly diverge.

Output layout -- colocated with the source .eaf by default:
    2025_corpus/sp-01_sp-02/sp-01_sp-02.eaf   (already there)
    2025_corpus/sp-01_sp-02/sp-01_sp-02.txt   (written by this script)

    All included speaker tiers merged in time order, one cleaned
    utterance per line. No headers, no speaker labels, no
    timestamps -- just the words, so the OOV checker's tokeniser
    sees exactly the same text as MFA's dictionary lookup would.

Since the .txt lands right next to its .eaf, 00f_check_oov.py can be
pointed straight at the corpus root (--transcripts 2025_corpus) --
no separate transcripts directory needed.

Usage:
    python 07b_eaf_to_txt_transcripts.py --corpus 2025_corpus

    # optional: write into one flat directory instead of colocating
    python 07b_eaf_to_txt_transcripts.py --corpus 2025_corpus --output transcripts_txt
"""

import re
import argparse
import sys
from pathlib import Path

import pympi
from num2words import num2words

# --- kept identical to 07_eaf_to_mfa_corpus.py; see note above ---
EXCLUDE_TIER_PATTERNS = ["note", "comment", "translation", "code", "default"]
TOP_LEVEL_TIERS_ONLY = True
CLEANUP_PATTERNS = [
    r"\((?:laughs?|laughter|coughs?|sighs?|inaudible|unclear)[^)]*\)",
    r"\[[^\]]*\]",          # anything in square brackets, e.g. [overlap]
    r"\bxxx+\b",            # unintelligible-speech placeholder
]
CLEANUP_RE = [re.compile(p, re.IGNORECASE) for p in CLEANUP_PATTERNS]

NUMERAL_RE = re.compile(r"(?<![:.\d])\b(\d+)(st|nd|rd|th)?\b(?![:.\d])",
                        re.IGNORECASE)


def expand_numerals(text: str) -> str:
    def _replace(m):
        digits, suffix = m.group(1), m.group(2)
        n = int(digits)
        words = num2words(n, to="ordinal") if suffix else num2words(n)
        return words.replace("-", " ")
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


def convert_one(eaf_path: Path, out_path: Path) -> tuple[int, int]:
    """Write one recording's transcript; returns (n_lines, n_with_leftover_digits)."""
    eaf = pympi.Elan.Eaf(str(eaf_path))

    # Gather (start_ms, text) across all included tiers, merged in time order.
    # Speaker identity isn't needed here -- the OOV check only cares about
    # vocabulary, not who said it -- so tiers are simply pooled.
    lines = []
    n_leftover_digits = 0
    for tier_name in eaf.get_tier_names():
        if not include_tier(eaf, tier_name):
            continue
        for start_ms, end_ms, value in eaf.get_annotation_data_for_tier(tier_name):
            text = clean_annotation(value or "")
            if text:
                if re.search(r"\d", text):
                    n_leftover_digits += 1
                lines.append((start_ms, text))

    lines.sort(key=lambda x: x[0])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, text in lines:
            f.write(text + "\n")

    return len(lines), n_leftover_digits


def main():
    ap = argparse.ArgumentParser(
        description="Export plain-text transcripts from corrected EAF files, "
                    "for OOV checking independent of MFA's CLI.")
    ap.add_argument("--corpus", required=True,
                    help="Consolidated corpus root (one folder per recording, "
                         "containing the corrected .eaf)")
    ap.add_argument("--output", default=None,
                    help="Optional flat output directory. If omitted "
                         "(the default), each .txt is written right next "
                         "to its source .eaf, colocated in the recording's "
                         "own folder.")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the .txt already exists")
    args = ap.parse_args()

    corpus_dir = Path(args.corpus)
    out_dir = Path(args.output) if args.output else None

    if not corpus_dir.is_dir():
        sys.exit(f"ERROR: corpus directory not found: {corpus_dir.resolve()}")

    eaf_files = sorted(corpus_dir.rglob("*.eaf"))
    if not eaf_files:
        sys.exit(f"No .eaf files found under {corpus_dir.resolve()}")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_done = n_skipped = n_failed = 0
    for eaf_path in eaf_files:
        stem = eaf_path.stem
        out_path = (out_dir / f"{stem}.txt") if out_dir is not None \
            else eaf_path.with_name(f"{stem}.txt")

        if out_path.exists() and not args.force:
            print(f"[skip] {stem}: transcript already exists")
            n_skipped += 1
            continue

        try:
            n_lines, n_digits = convert_one(eaf_path, out_path)
            note = f"  ({n_digits} line(s) still contain digits)" if n_digits else ""
            print(f"[conv] {stem}: {n_lines} lines{note}")
            n_done += 1
        except Exception as e:
            print(f"[FAIL] {stem}: {e}")
            n_failed += 1

    print(f"\nSummary: {n_done} written, {n_skipped} skipped, {n_failed} failed.")
    sys.exit(1 if n_failed else 0)


if __name__ == "__main__":
    main()