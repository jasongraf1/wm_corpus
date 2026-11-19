"""
00d_make_mfa_dict.py

Strip comments and blank lines from the human-readable pronunciation
master to produce the MFA-ready dictionary file. UTF-8 in, UTF-8 (no
BOM) out, so IPA symbols survive intact.

Run after any edit to the master, then:
    mfa model add_words english_uk_mfa custom_pronunciations_mfa.txt
"""

import sys
from pathlib import Path

SRC = Path("custom_pronunciations_master.txt")
DST = Path("custom_pronunciations_mfa.txt")

lines = []
for raw in SRC.read_text(encoding="utf-8-sig").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if len(line.split(maxsplit=1)) < 2:
        sys.exit(f"Bad entry (word without pronunciation): {line!r}")
    lines.append(line)

DST.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"{len(lines)} entries -> {DST}")
