"""Near-duplicate filter for templated text (e.g. thousands of village stubs).

Exact dedup (clean.py) cannot catch lines that share a template but differ in
names/numbers. Idea: build each line's "skeleton" = its words that are frequent
across the corpus (document frequency >= MIN_DF), ignoring numbers. Template words are frequent,
names are rare, so templated lines get the same skeleton. Keep at most
MAX_PER_SKELETON lines per skeleton.

Reads/overwrites data/clean/{lang}.txt (original saved as {lang}.exact.txt).
Run AFTER clean.py and BEFORE split.py. Adds "near_dedup" to data/stats.json.
"""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
STATS_PATH = ROOT / "data" / "stats.json"

MIN_DF = 1000           # a word is "template-like" if it appears in >= this many lines
MIN_SKELETON_LEN = 6    # only treat lines with long skeletons as templated
MAX_PER_SKELETON = 3


def read_articles(path):
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


def run(lang: str) -> dict:
    src = CLEAN_DIR / f"{lang}.txt"
    backup = CLEAN_DIR / f"{lang}.exact.txt"
    if not backup.exists():           # keep the exact-dedup output so reruns are idempotent
        shutil.copyfile(src, backup)
    src_path = backup

    df = Counter()
    for article in read_articles(src_path):
        for line in article:
            df.update(set(line.split()))
    common = {w for w, c in df.items() if c >= MIN_DF}

    seen = Counter()
    before = after = bytes_before = bytes_after = 0
    articles_before = articles_after = 0
    with open(src, "w", encoding="utf-8", newline="\n") as out:
        for article in read_articles(src_path):
            articles_before += 1
            kept = []
            for line in article:
                n = len(line.encode("utf-8"))
                before += 1
                bytes_before += n
                skel = [w for w in line.split() if w in common and not any(c.isdigit() for c in w)]
                if len(skel) >= MIN_SKELETON_LEN:
                    key = hashlib.md5(" ".join(skel).encode("utf-8")).digest()
                    seen[key] += 1
                    if seen[key] > MAX_PER_SKELETON:
                        continue
                kept.append(line)
                after += 1
                bytes_after += n
            if kept:
                out.write("\n".join(kept) + "\n\n")
                articles_after += 1
    res = {"lines_before": before, "lines_after": after,
           "bytes_before": bytes_before, "bytes_after": bytes_after,
           "articles_before": articles_before, "articles_after": articles_after,
           "params": {"min_df": MIN_DF, "min_skeleton_len": MIN_SKELETON_LEN,
                      "max_per_skeleton": MAX_PER_SKELETON}}
    print(f"{lang}: {before:,} -> {after:,} lines ({bytes_before / 1e6:.1f} -> {bytes_after / 1e6:.1f} MB), "
          f"articles {articles_before:,} -> {articles_after:,}")
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("langs", nargs="*", default=["te", "ne"])
    args = p.parse_args()
    stats = json.loads(STATS_PATH.read_text(encoding="utf-8")) if STATS_PATH.exists() else {}
    for lang in args.langs:
        stats.setdefault(lang, {})["near_dedup"] = run(lang)
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
