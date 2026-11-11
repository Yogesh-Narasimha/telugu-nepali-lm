#!/bin/bash
# Rebuild everything after near-dedup: split -> tokenizers -> tokens -> train -> evaluate
set -e
PY="/c/Users/yoges/AppData/Local/Programs/Python/Python310/python.exe"
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
$PY src/split.py
$PY src/tokenizer.py
$PY src/prepare_data.py
for l in te ne; do $PY src/train.py $l --max-iters 5000 > results/train_$l.log 2>&1; done
$PY src/evaluate.py > results/evaluate.log 2>&1
