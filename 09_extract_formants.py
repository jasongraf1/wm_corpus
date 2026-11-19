"""
09_extract_formants.py
-----------------------
Reads MFA-produced TextGrids + corresponding WAVs and extracts vowel
formants (F1, F2, F3) and F0 at 5 timepoints per vowel token.

Designed for MULTI-SPEAKER recordings: MFA names tiers
"<speaker> - words" and "<speaker> - phones", one pair per speaker.
Falls back to plain "words"/"phones" tiers for single-speaker files
(speaker name taken from the filename).

Expected corpus layout (one folder per recording):
    mfa_output/
        SH-03_SH-04/
            SH-03_SH-04.TextGrid
            SH-03_SH-04.wav
        SH-05_SH-06/
            ...

Output:
    vowel_formants.csv  (long format: one row per vowel token per timepoint)

Dependencies:
    pip install praat-parselmouth tgt pandas numpy

Usage:
    python 09_extract_formants.py --corpus_dir 2025_corpus --output_dir output_analysis
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
import tgt

from bisect import bisect_left, bisect_right

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

# Proportional timepoints within each vowel interval.
# 20-80% avoids CV/VC transition zones; equally valid alternative is
# [0.10, 0.30, 0.50, 0.70, 0.90] if you want fuller trajectories.
TIMEPOINTS = [0.20, 0.35, 0.50, 0.65, 0.80]

# Formant ceiling per speaker. Rule of thumb: ~5000 Hz for adult male,
# ~5500 Hz for adult female voices. Anything not listed uses the default.
SPEAKER_MAX_FORMANT = {
    # "SH-03": 5500,
    # "SH-04": 5000,
}
DEFAULT_MAX_FORMANT = 5000

# F0 search range (applied to all speakers; widen rather than narrow)
F0_MIN = 75
F0_MAX = 400

# Minimum duration (s) of a vowel interval to attempt extraction
MIN_PHONE_DURATION = 0.03

# Padding (s) added around each token before acoustic analysis.
# Needs to be >= half the formant window (12.5 ms); 100 ms also gives
# the pitch tracker enough context at F0_MIN = 75 Hz.
TOKEN_PAD = 0.10

# Vowels are identified by character content rather than a hard-coded
# inventory, so this works regardless of which english_uk_mfa symbols
# appear (aj, ej, əw, ʉː, ɒː, ɐ, ɛː, etc.). Any phone containing at
# least one of these base characters counts as a vowel.
VOWEL_CHARS = set("aeiouæɐɑɒɔəɘɵɛɜɪʊʉʌʏyøœɶɨɯɤ")

PHONE_TO_LEXSET = {
    "ɪ":  "KIT",     "ɛ":  "DRESS",   "a":  "TRAP",
    "ɒ":  "LOT",     "ɐ":  "STRUT",   "ʊ":  "FOOT",
    "iː": "FLEECE",  "ʉː": "GOOSE",   "ɜː": "NURSE",
    "ɑː": "START/PALM",
    "ɒː": "THOUGHT/NORTH/FORCE",
    "ej": "FACE",    "aj": "PRICE",   "ɔj": "CHOICE",
    "əw": "GOAT",    "aw": "MOUTH", "ʊ ə": "CURE",
    "ɛː": "SQUARE",  "ɪ ə": "NEAR",
    "ə":  "commA/lettER", 
}

# The low vowels are not well marked so we hard code some
BATH = ["pasty", "abaft", "advance", "advantage", "aft", "after", "aghast", "alas", "Alexander", "Alexandra", "alto", "answer", "ask", "askance", "aunt", "avalanche", "avast", "banal", "banana", "bask", "basket", "Basque", "bastard", "bath", "Belfast", "blanch", "Blanche", "blasphemy", "blast", "branch", "brass", "calf", "calve", "can't", "cask", "casket", "cast", "caste", "caster", "castle", "castor", "chaff", "chance", "chancel", "chancellor", "chancery", "chandler", "chant", "chantry", "chorale", "circumstantial", "clasp", "class", "Cleopatra", "command", "commando", "contralto", "contrast", "corral", "countermand", "craft", "daft", "dance", "demand", "disaster", "disastrous", "draft", "drastic", "draught", "durmast", "elastic", "enchant", "enhance", "enthusiastic", "example", "exasperate", "fast", "fasten", "flabbergast", "Flanders", "flask", "France", "Frances", "Francis", "gasp", "gast", "ghast", "ghastly", "giraffe", "glance", "glasgow", "glass", "graft", "grant", "Grant", "graph", "grasp", "grass", "gymnastic", "haft", "half", "halve", "hance", "hasp", "impasse", "Internationale", "intransigent", "Iran", "Iraq", "kraft", "lance", "lancet", "last", "lath", "lather", "laugh", "laughter", "locale", "masculine", "mask", "masque", "masquerade", "mast", "master", "masturbate", "morale", "moustache", "musicale", "nasty", "paschal", "pass", "past", "pastime", "pastor", "pastoral", "pastorale", "pasture", "path", "planchet", "planchette", "plant", "plaque", "plaster", "plastic", "prance", "raft", "rafter", "rale", "ranch", "rancho", "rascal", "rasp", "raspberry", "rather", "rationale", "remand", "repast", "reprimand", "royale", "salve", "sample", "Sanders", "Sandra", "shaft", "Shaftesbury", "shan't", "slander", "slant", "Slav", "staff", "stance", "stanch", "stanchion", "substantial", "supplant", "task", "trance", "tranche", "transept", "transfer", "transform", "transient", "transit", "transitory", "transport", "udan", "vantage", "vast"] 

PALM = ["palm", "calm", "balm", "psalm", "alms", "father", "bra", "ma", "pa", "mamma", "pappa", "aha", "ah", "hah", "ha", "blah", "hurrah", "baht", "Bach", "facade", "couvade", "roulade", "raj", "taj", "salaam", "Brahms", "Kahn", "Afrikaans", "kraal", "Transvaal", "Mahal", "spa", "Shah", "Pooh-Bah", "Armagh", "schwa", "cantata", "inamorato", "legato", "sonata", "staccato", "pizzicato", "Lusaka", "Karachi", "mafia", "Dada", "bravado", "incommunicado", "Mahdi", "Mikado", "laager", "lager", "Zhivago", "maharajah", "kava", "guava", "Java", "Swazi", "Dali", "Mali", "Guatemala", "Somalia", "Somali", "lama", "llama", "Yokohama", "swami", "Brahmin", "guano", "marijuana", "iguana", "Botswana", "maharani", "ha-ha", "Malawi", "Bahai", "Sumatra", "candelabra", "boa", "bah", "Koran", "khan", "Pakistan", "Shan", "chorale", "rationale", "locale", "khaki", "pasha", "Nazi", "Colorado", "enchilada", "Nevada", "aubade", "lava", "palaver", "plaza", "almond", "drama", "pajama/pyjama", "panorama", "Ghana", "nirvana", "sultana", "soprano", "piranha", "Bali", "finale"]

def is_vowel(phone: str) -> bool:
    return any(ch in VOWEL_CHARS for ch in phone)


# ---------------------------------------------------------------------------
# TIER HANDLING (multi-speaker aware)
# ---------------------------------------------------------------------------

SPEAKER_TIER_RE = re.compile(r"^(?P<spk>.+?)\s*-\s*(?P<kind>words|phones)$",
                             re.IGNORECASE)


def get_speaker_tiers(tg, fallback_speaker: str) -> dict:
    """
    Return {speaker: {"words": tier, "phones": tier}}.

    MFA multi-speaker output names tiers "<speaker> - words" and
    "<speaker> - phones". If no such tiers exist, fall back to plain
    "words"/"phones" tiers under a single speaker name.
    """
    speakers: dict[str, dict] = {}
    for tier in tg.tiers:
        m = SPEAKER_TIER_RE.match(tier.name.strip())
        if m:
            spk = m.group("spk").strip()
            speakers.setdefault(spk, {})[m.group("kind").lower()] = tier

    if speakers:
        return speakers

    # Single-speaker fallback: plain "words" / "phones" tiers
    single: dict = {}
    for tier in tg.tiers:
        name = tier.name.strip().lower()
        if name in ("words", "word"):
            single["words"] = tier
        elif name in ("phones", "phone"):
            single["phones"] = tier
    return {fallback_speaker: single} if single else {}


# def find_word(word_tier, t_mid: float) -> str:
#     """Word whose interval contains the vowel midpoint."""
#     if word_tier is None:
#         return ""
#     for w in word_tier.intervals:
#         if w.start_time <= t_mid <= w.end_time:
#             return w.text.strip()
#     return ""

def find_word_interval(word_tier, t_mid):
    if word_tier is None:
        return None
    for w in word_tier.intervals:
        if w.start_time <= t_mid <= w.end_time:
            return w
    return None

def overlaps_other_speaker(other_word_tiers, start: float, end: float) -> bool:
    """True if any other speaker has non-empty speech overlapping [start, end]."""
    for tier in other_word_tiers:
        for w in tier.intervals:
            if w.text.strip() and w.start_time < end and w.end_time > start:
                return True
    return False


# ---------------------------------------------------------------------------
# ACOUSTICS
# ---------------------------------------------------------------------------

def measure_token(snd: parselmouth.Sound, start: float, end: float,
                  max_hz: float) -> tuple:
    """
    Build Formant and Pitch objects for one padded token.
    Returns (formant, pitch) — query these at absolute times
    (preserve_times=True keeps the original time axis).
    """
    t_min = max(0.0, start - TOKEN_PAD)
    t_max = min(snd.duration, end + TOKEN_PAD)
    sub = snd.extract_part(from_time=t_min, to_time=t_max, preserve_times=True)
    formant = call(sub, "To Formant (burg)", 0.0, 5, max_hz, 0.025, 50)
    try:
        pitch = call(sub, "To Pitch", 0.0, F0_MIN, F0_MAX)
    except parselmouth.PraatError:
        pitch = None
    return formant, pitch


# ---------------------------------------------------------------------------
# EXTRACTION
# ---------------------------------------------------------------------------

def read_textgrid_any_encoding(path: Path):
    """MFA writes UTF-8, but Praat re-saves TextGrids as UTF-16."""
    with open(path, "rb") as f:
        bom = f.read(2)
    if bom in (b"\xfe\xff", b"\xff\xfe"):
        return tgt.read_textgrid(str(path), encoding="utf-16")
    return tgt.read_textgrid(str(path))

def extract_vowels_for_speaker(recording: str, speaker: str, tiers: dict,
                               other_word_tiers: list, snd: parselmouth.Sound
                               ) -> list[dict]:
    phone_tier = tiers.get("phones")
    word_tier = tiers.get("words")
    
    if phone_tier is None:
        print(f"  WARNING: no phones tier for speaker {speaker} — skipping.")
        return []

    vowel_mids = sorted((p.start_time + p.end_time) / 2
                for p in phone_tier.intervals
                if is_vowel(p.text.strip()))
    
    max_hz = SPEAKER_MAX_FORMANT.get(speaker, DEFAULT_MAX_FORMANT)
    rows = []

    token_n = 0

    for interval in phone_tier.intervals:
        phone = interval.text.strip()
        if not phone or not is_vowel(phone):
            continue
        start, end = interval.start_time, interval.end_time
        dur = end - start
        if dur < MIN_PHONE_DURATION:
            continue
        
        token_n += 1
        token_id = f"{recording}_{speaker}_{token_n:04d}"

        t_mid = (start + end) / 2
        # word = find_word(word_tier, t_mid)
        w = find_word_interval(word_tier, t_mid)
        word = w.text.strip() if w else ""
        is_word_final = w is not None and (w.end_time - end) < 0.02
        n_vowels = (bisect_right(vowel_mids, w.end_time)
                    - bisect_left(vowel_mids, w.start_time)) if w else 0
        lexset = classify_lexical_set(phone, word, is_word_final, n_vowels)

        overlap = overlaps_other_speaker(other_word_tiers, start, end)

        try:
            formant, pitch = measure_token(snd, start, end, max_hz)
        except Exception as e:
            print(f"  Warning: analysis failed for {phone} at {start:.3f}s: {e}")
            continue

        for pct in TIMEPOINTS:
            t = start + dur * pct
            f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
            f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
            f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")
            f0 = (call(pitch, "Get value at time", t, "Hertz", "Linear")
                  if pitch is not None else np.nan)

            rows.append({
                "token_id": token_id,
                "recording":  recording,
                "speaker":    speaker,
                "word":       word,
                "lexical_set": lexset,
                "phone":      phone,
                "start":      round(start, 4),
                "end":        round(end, 4),
                "duration":   round(dur, 4),
                "timepoint":  int(pct * 100),
                "t":          round(t, 4),
                "F1":         round(f1, 1) if not np.isnan(f1) else None,
                "F2":         round(f2, 1) if not np.isnan(f2) else None,
                "F3":         round(f3, 1) if not np.isnan(f3) else None,
                "F0":         round(f0, 1) if not np.isnan(f0) else None,
                "overlap":    overlap,
                "word_final": is_word_final,
            })

    return rows


def process_recording(tg_path: Path, wav_path: Path) -> list[dict]:
    recording = tg_path.stem
    print(f"\nProcessing recording: {recording}")
    print(f"  TextGrid: {tg_path}")
    print(f"  WAV:      {wav_path}")

    tg = read_textgrid_any_encoding(tg_path)
    snd = parselmouth.Sound(str(wav_path))

    speakers = get_speaker_tiers(tg, fallback_speaker=recording)
    if not speakers:
        print("  WARNING: no word/phone tiers found — skipping.")
        return []
    print(f"  Speakers: {', '.join(sorted(speakers))}")

    all_rows = []
    for speaker in sorted(speakers):
        other_word_tiers = [t["words"] for s, t in speakers.items()
                            if s != speaker and t.get("words") is not None]
        rows = extract_vowels_for_speaker(recording, speaker,
                                          speakers[speaker],
                                          other_word_tiers, snd)
        n_tokens = len(rows) // len(TIMEPOINTS) if rows else 0
        print(f"    {speaker}: {n_tokens} vowel tokens")
        all_rows.extend(rows)

    return all_rows

# ---------------------------------------------------------------------------
# LEXICAL SETS
# ---------------------------------------------------------------------------

def classify_lexical_set(phone, word, is_word_final, n_vowels_in_word):
    if phone in {"iː", "i", "ɪ"} and is_word_final and n_vowels_in_word >= 2:
        return "happY"
    if phone == "ə" and is_word_final:
        return "lettER" if word.rstrip("'").endswith(("r", "re")) else "commA"
    
    lexset = PHONE_TO_LEXSET.get(phone, "")
    # low vowel sets are tricky so we 
    if phone == "a" and word in BATH:
        return "BATH"
    elif phone == "a" and word in PALM:
        return "START/PALM"
    return PHONE_TO_LEXSET.get(phone, "")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def find_wav_for(tg_path: Path) -> Path | None:
    """WAV with the same stem, in the same folder as the TextGrid."""
    for candidate in [tg_path.with_suffix(".wav")]:
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract vowel formants from MFA TextGrids (multi-speaker)")
    parser.add_argument("--corpus_dir", required=True,
                        help="Root directory containing one folder per "
                             "recording, each with a .TextGrid and .wav")
    parser.add_argument("--output_dir", default="output_analysis",
                        help="Output CSV directory")
    args = parser.parse_args()

    corpus_dir = Path(args.corpus_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tg_files = sorted(corpus_dir.rglob("*.TextGrid"))
    if not tg_files:
        print(f"No TextGrid files found under {corpus_dir}")
        return

    all_rows: list[dict] = []
    for tg_path in tg_files:
        wav_path = find_wav_for(tg_path)
        if wav_path is None:
            print(f"\nWARNING: no WAV found for {tg_path} — skipping.")
            continue
        all_rows.extend(process_recording(tg_path, wav_path))

    if not all_rows:
        print("\nNo vowel tokens extracted.")
        return

    df = pd.DataFrame(all_rows)
    out_path = output_dir / "vowel_formants.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    n_tokens = len(df) // len(TIMEPOINTS)
    print(f"\nWrote {out_path}")
    print(f"  {len(df)} rows ({n_tokens} vowel tokens x {len(TIMEPOINTS)} timepoints)")
    print(f"  Speakers: {', '.join(sorted(df['speaker'].unique()))}")
    print(f"  Tokens in overlapping speech: "
          f"{df[df['overlap']]['phone'].count() // len(TIMEPOINTS)} "
          f"(consider excluding from formant analysis)")

    print("\nNext steps:")
    print("  1. Spot-check formant tracks in Praat, especially low back vowels.")
    print("  2. Set per-speaker formant ceilings in SPEAKER_MAX_FORMANT if any")
    print("     speaker's F1/F2 values look systematically off.")
    print("  3. Run 05_normalise_vowels.py to Lobanov-normalise.")


if __name__ == "__main__":
    main()
