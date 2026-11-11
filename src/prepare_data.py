"""Tokenise train/val/test text into flat uint16 arrays: data/tokens/{lang}.{split}.bin

Each article is encoded and followed by an <eos> token, then all articles are
concatenated into one stream (training samples random windows from it).
"""
import argparse
from pathlib import Path

import numpy as np
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
OUT_DIR = ROOT / "data" / "tokens"


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


def prepare(lang: str, vocab: int):
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizers" / f"{lang}_{vocab}.model"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        chunks = []
        for article in read_articles(CLEAN_DIR / f"{lang}.{split}.txt"):
            for ids in sp.encode(article):
                chunks.append(np.asarray(ids, dtype=np.uint16))
            chunks.append(np.asarray([sp.eos_id()], dtype=np.uint16))
        arr = np.concatenate(chunks)
        arr.tofile(OUT_DIR / f"{lang}.{split}.bin")
        print(f"{lang}.{split}: {len(arr):,} tokens")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("langs", nargs="*", default=["te", "ne"])
    p.add_argument("--vocab", type=int, default=8000)
    a = p.parse_args()
    for lang in a.langs:
        prepare(lang, a.vocab)
