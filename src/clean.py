"""Clean raw Wikipedia text: data/raw/{lang}.txt -> data/clean/{lang}.txt.

Pipeline, applied line by line in this order, with counts after every step:
  1. NFC Unicode normalisation
  2. junk removal (URLs, extra whitespace, lines shorter than MIN_CHARS)
  3. script filter (>= SCRIPT_RATIO of letters must be in the language's script)
  4. exact-duplicate removal (hash of each line)

Articles in the raw file are separated by a blank line. That structure is kept
in the output (an article that loses all its lines is dropped) so split.py can
split by article.
"""
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CLEAN_DIR = ROOT / "data" / "clean"
STATS_PATH = ROOT / "data" / "stats.json"

MIN_CHARS = 20
SCRIPT_RATIO = 0.70
# Unicode blocks: Telugu U+0C00-U+0C7F, Devanagari (Nepali) U+0900-U+097F
SCRIPT_RANGES = {"te": (0x0C00, 0x0C7F), "ne": (0x0900, 0x097F)}

URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
SPACE_RE = re.compile(r"\s+")
STEPS = ["raw", "nfc", "junk_removed", "script_filtered", "deduplicated"]


def is_letter(ch: str) -> bool:
    # Vowel signs (matras) are combining marks (category M*), not isalpha(),
    # but they are part of the word, so count them as letters too.
    return ch.isalpha() or unicodedata.category(ch).startswith("M")


def script_ratio(line: str, lo: int, hi: int) -> float:
    letters = [c for c in line if is_letter(c)]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def read_articles(path: Path):
    """Yield each article as a list of non-empty raw lines."""
    article = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip():
                article.append(line)
            elif article:
                yield article
                article = []
    if article:
        yield article


class Counter:
    def __init__(self):
        self.lines = self.chars = self.bytes = 0

    def add(self, line: str):
        self.lines += 1
        self.chars += len(line)
        self.bytes += len(line.encode("utf-8"))

    def as_dict(self):
        return {"lines": self.lines, "chars": self.chars, "bytes": self.bytes}


def clean_language(lang: str) -> dict:
    lo, hi = SCRIPT_RANGES[lang]
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    counts = {s: Counter() for s in STEPS}
    seen = set()  # hashes of lines already kept
    articles_in = articles_out = 0

    with open(CLEAN_DIR / f"{lang}.txt", "w", encoding="utf-8", newline="\n") as fout:
        for article in read_articles(RAW_DIR / f"{lang}.txt"):
            articles_in += 1
            kept = []
            for line in article:
                counts["raw"].add(line)

                line = unicodedata.normalize("NFC", line)
                counts["nfc"].add(line)

                line = URL_RE.sub(" ", line)
                line = SPACE_RE.sub(" ", line).strip()
                if len(line) < MIN_CHARS:
                    continue
                counts["junk_removed"].add(line)

                if script_ratio(line, lo, hi) < SCRIPT_RATIO:
                    continue
                counts["script_filtered"].add(line)

                digest = hashlib.md5(line.encode("utf-8")).digest()
                if digest in seen:
                    continue
                seen.add(digest)
                counts["deduplicated"].add(line)
                kept.append(line)

            if kept:
                fout.write("\n".join(kept) + "\n\n")
                articles_out += 1

    result = {s: counts[s].as_dict() for s in STEPS}
    result["articles"] = {"raw": articles_in, "clean": articles_out}
    return result


def print_summary(lang: str, res: dict):
    print(f"\n[{lang}]")
    for s in STEPS:
        r = res[s]
        print(f"  {s:<16} {r['lines']:>10,} lines {r['chars']:>13,} chars {r['bytes'] / 1e6:>8.1f} MB")
    print(f"  articles: {res['articles']['raw']:,} -> {res['articles']['clean']:,}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("langs", nargs="*", default=["te", "ne"])
    args = p.parse_args()

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8")) if STATS_PATH.exists() else {}
    for lang in args.langs:
        res = clean_language(lang)
        print_summary(lang, res)
        stats[lang] = {**stats.get(lang, {}), "clean": res}
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nstats -> {STATS_PATH}")
