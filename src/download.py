"""Download Telugu and Nepali Wikipedia text to data/raw/{lang}.txt.

Articles are separated by a blank line so later steps can split by article.
Newlines inside an article are kept (one paragraph per line).
"""
import argparse
from pathlib import Path

from datasets import load_dataset

SNAPSHOT = "20231101"
MAX_BYTES = 200 * 1024 * 1024  # ~200 MB per language
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(lang: str, max_bytes: int = MAX_BYTES) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{lang}.txt"
    ds = load_dataset("wikimedia/wikipedia", f"{SNAPSHOT}.{lang}",
                      split="train", streaming=True)

    written, articles = 0, 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for row in ds:
            text = row["text"].strip()
            if not text:
                continue
            f.write(text + "\n\n")
            written += len(text.encode("utf-8")) + 2
            articles += 1
            if written >= max_bytes:
                break
    print(f"{lang}: {articles} articles, {written / 1e6:.1f} MB -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("langs", nargs="*", default=["te", "ne"])
    p.add_argument("--max-mb", type=int, default=200)
    a = p.parse_args()
    for lang in a.langs:
        download(lang, a.max_mb * 1024 * 1024)
