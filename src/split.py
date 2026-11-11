"""Split cleaned text into train/val/test (98/1/1) BY ARTICLE.

Splitting by article (not by line) keeps lines from one article out of both
train and test, so the test score is not inflated. A fixed seed makes the
split identical on every run.

Reads  data/clean/{lang}.txt
Writes data/clean/{lang}.{train,val,test}.txt  (articles still separated by a blank line)
Updates data/stats.json with a "split" section per language.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
STATS_PATH = ROOT / "data" / "stats.json"

SEED = 42
VAL_FRAC = 0.01
TEST_FRAC = 0.01


def read_articles(path: Path):
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


def split_language(lang: str) -> dict:
    articles = list(read_articles(CLEAN_DIR / f"{lang}.txt"))
    rng = random.Random(SEED)
    order = list(range(len(articles)))
    rng.shuffle(order)

    n = len(articles)
    n_val, n_test = round(n * VAL_FRAC), round(n * TEST_FRAC)
    parts = {
        "test": order[:n_test],
        "val": order[n_test:n_test + n_val],
        "train": order[n_test + n_val:],
    }

    result = {}
    for name, idxs in parts.items():
        lines = chars = nbytes = 0
        with open(CLEAN_DIR / f"{lang}.{name}.txt", "w", encoding="utf-8", newline="\n") as f:
            for i in idxs:
                for line in articles[i]:
                    lines += 1
                    chars += len(line)
                    nbytes += len(line.encode("utf-8"))
                f.write("\n".join(articles[i]) + "\n\n")
        result[name] = {"articles": len(idxs), "lines": lines, "chars": chars, "bytes": nbytes}
        print(f"  {lang}.{name:<5} {len(idxs):>7,} articles {lines:>9,} lines {nbytes / 1e6:>7.1f} MB")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("langs", nargs="*", default=["te", "ne"])
    args = p.parse_args()

    stats = json.loads(STATS_PATH.read_text(encoding="utf-8")) if STATS_PATH.exists() else {}
    for lang in args.langs:
        stats.setdefault(lang, {})["split"] = {"seed": SEED, **split_language(lang)}
    STATS_PATH.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"stats -> {STATS_PATH}")
