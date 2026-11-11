"""Train a SentencePiece tokenizer per language and compare vocab sizes.

For each language and vocab size, trains a Unigram model on the train split
and measures, on the held-out test split:
  - tokens per word  (fertility; lower = more efficient)
  - chars per token
  - unknown-token rate
Models go to tokenizers/{lang}_{vocab}.model, metrics to results/tokenizer_comparison.json.
"""
import argparse
import json
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
CLEAN_DIR = ROOT / "data" / "clean"
TOK_DIR = ROOT / "tokenizers"
RESULTS_PATH = ROOT / "results" / "tokenizer_comparison.json"

VOCAB_SIZES = [4000, 8000, 16000]
MODEL_TYPE = "unigram"
CHAR_COVERAGE = 0.9995  # Telugu/Devanagari have many rare conjunct forms


def lines_of(path: Path):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def train(lang: str, vocab: int) -> Path:
    TOK_DIR.mkdir(exist_ok=True)
    # SentencePiece wants a plain file of one sentence per line (no blank article separators)
    tmp = TOK_DIR / f"_{lang}_train.txt"
    tmp.write_text("\n".join(lines_of(CLEAN_DIR / f"{lang}.train.txt")), encoding="utf-8")
    prefix = TOK_DIR / f"{lang}_{vocab}"
    spm.SentencePieceTrainer.train(
        input=str(tmp), model_prefix=str(prefix), vocab_size=vocab,
        model_type=MODEL_TYPE, character_coverage=CHAR_COVERAGE,
        input_sentence_size=1_000_000, shuffle_input_sentence=True,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=[], num_threads=4,
    )
    tmp.unlink()
    return prefix.with_suffix(".model")


def evaluate(model_path: Path, test_lines) -> dict:
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    ids = sp.encode(test_lines)
    n_tokens = sum(len(x) for x in ids)
    n_words = sum(len(l.split()) for l in test_lines)
    n_chars = sum(len(l) for l in test_lines)
    n_unk = sum(i == sp.unk_id() for x in ids for i in x)
    return {
        "tokens": n_tokens, "words": n_words,
        "tokens_per_word": round(n_tokens / n_words, 3),
        "chars_per_token": round(n_chars / n_tokens, 3),
        "unk_rate": round(n_unk / n_tokens, 6),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("langs", nargs="*", default=["te", "ne"])
    args = p.parse_args()

    results = {}
    for lang in args.langs:
        test = lines_of(CLEAN_DIR / f"{lang}.test.txt")
        results[lang] = {}
        for v in VOCAB_SIZES:
            m = evaluate(train(lang, v), test)
            results[lang][str(v)] = m
            print(f"{lang} vocab={v:<6} tokens/word={m['tokens_per_word']:<6} "
                  f"chars/token={m['chars_per_token']:<6} unk={m['unk_rate']}")
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"-> {RESULTS_PATH}")
