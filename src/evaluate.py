"""Evaluate trained models on the held-out TEST split.

For each language reports:
  - GPT test loss / perplexity (per token) and bits-per-character (BPC)
  - unigram and interpolated-bigram baselines on the same tokens
  - sample generations
  - loss curves (results/loss_curves.png)
BPC is the fair cross-language number: perplexity depends on each tokenizer,
bits per character does not.

  python src/evaluate.py            # both languages
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "data" / "tokens"
CLEAN_DIR = ROOT / "data" / "clean"
RESULTS_DIR = ROOT / "results"
VOCAB = 8000
LN2 = math.log(2)
device = "cuda" if torch.cuda.is_available() else "cpu"


def n_chars(lang):
    """Characters in the test text (non-empty lines, no separators) = what the tokens encode."""
    with open(CLEAN_DIR / f"{lang}.test.txt", encoding="utf-8") as f:
        return sum(len(l.rstrip("\n")) for l in f if l.strip())


@torch.no_grad()
def gpt_nll(model, data, stride=128):
    """Total NLL (nats) and token count using sliding windows, so each scored
    token sees at least block_size - stride tokens of context."""
    T = model.cfg.block_size
    data = np.asarray(data, dtype=np.int64)
    total, count, scored_upto = 0.0, 0, 0   # scored_upto: first target index not yet scored
    model.eval()
    N = len(data) - 1
    start = 0
    while scored_upto < N:
        end = min(start + T, N)
        x = torch.from_numpy(data[start:end]).unsqueeze(0).to(device)
        y = torch.from_numpy(data[start + 1:end + 1]).unsqueeze(0).to(device)
        logits, _ = model(x)
        lp = torch.log_softmax(logits.float(), dim=-1)
        nll = -lp.gather(-1, y.unsqueeze(-1)).squeeze(-1)[0]
        keep = nll[scored_upto - start:]
        total += keep.sum().item()
        count += keep.numel()
        scored_upto = end
        if end == N:
            break
        start += stride
    return total, count


def baselines(train, val, test):
    """Unigram and interpolated bigram (lambda tuned on val), add-one smoothed unigram."""
    uni = np.bincount(train, minlength=VOCAB).astype(np.float64) + 1
    p_uni = uni / uni.sum()
    big = np.zeros((VOCAB, VOCAB), dtype=np.int32)
    np.add.at(big, (train[:-1], train[1:]), 1)
    row = big.sum(1).astype(np.float64)

    def nll(data, lam):
        prev, nxt = data[:-1], data[1:]
        pb = np.where(row[prev] > 0, big[prev, nxt] / np.maximum(row[prev], 1), 0.0)
        p = lam * pb + (1 - lam) * p_uni[nxt]
        return -np.log(p).sum(), len(nxt)

    u = -np.log(p_uni[test[1:]]).sum(), len(test) - 1
    best_lam = min([0.1, 0.3, 0.5, 0.7, 0.8, 0.9], key=lambda l: nll(val, l)[0] / nll(val, l)[1])
    b = nll(test, best_lam)
    return {"unigram": u, "bigram": b}, best_lam


def generate_samples(model, sp, test, lang, n=3, prompt_len=12, new_tokens=60):
    out = []
    rng = np.random.default_rng(0)
    eos_pos = np.where(test == sp.eos_id())[0]
    torch.manual_seed(0)
    for k in range(n):
        s = int(rng.choice(eos_pos[:-1])) + 1          # start of a random article
        prompt = torch.tensor(test[s:s + prompt_len].astype(np.int64))[None].to(device)
        gen = model.generate(prompt.clone(), new_tokens, temperature=0.8, top_k=40)[0].tolist()
        gen = [t for t in gen if t != sp.eos_id()]
        out.append(f"[{k + 1}] PROMPT:    {sp.decode(gen[:prompt_len])}\n"
                   f"    CONTINUED: {sp.decode(gen[prompt_len:])}\n")
    return "\n".join(out)


def plot_curves(logs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, (lang, name) in zip(ax, [("te", "Telugu"), ("ne", "Nepali")]):
        lg = logs[lang]
        tr = lg["train"]
        k = 20  # smooth train loss
        sm = np.convolve([t["loss"] for t in tr], np.ones(k) / k, mode="valid")
        a.plot([t["iter"] for t in tr][k - 1:], sm, label="train (smoothed)")
        a.plot([v["iter"] for v in lg["val"]], [v["loss"] for v in lg["val"]], "o-", ms=3, label="validation")
        a.set_title(f"{name} GPT ({lg['params'] / 1e6:.1f}M params)")
        a.set_xlabel("iteration"); a.grid(alpha=0.3); a.set_ylim(top=8)
    ax[0].set_ylabel("cross-entropy loss (nats/token)"); ax[0].legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "loss_curves.png", dpi=150)


def main():
    results, logs = {}, {}
    for lang, name in [("te", "Telugu"), ("ne", "Nepali")]:
        sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizers" / f"{lang}_{VOCAB}.model"))
        train = np.fromfile(TOK_DIR / f"{lang}.train.bin", dtype=np.uint16).astype(np.int64)
        val = np.fromfile(TOK_DIR / f"{lang}.val.bin", dtype=np.uint16).astype(np.int64)
        test = np.fromfile(TOK_DIR / f"{lang}.test.bin", dtype=np.uint16).astype(np.int64)
        chars = n_chars(lang)

        ckpt = torch.load(ROOT / "checkpoints" / f"{lang}_best.pt", map_location=device)
        model = GPT(GPTConfig(**ckpt["config"])).to(device)
        model.load_state_dict(ckpt["model"])
        logs[lang] = json.loads((RESULTS_DIR / f"{lang}_log.json").read_text(encoding="utf-8"))

        base, lam = baselines(train, val, test)
        rows = {"gpt": gpt_nll(model, test), **base}
        res = {"best_iter": ckpt["iter"], "test_tokens": len(test), "test_chars": chars,
               "bigram_lambda": lam, "models": {}}
        for key, (nll, cnt) in rows.items():
            res["models"][key] = {
                "loss": round(nll / cnt, 4), "perplexity": round(math.exp(nll / cnt), 2),
                "bits_per_char": round(nll / LN2 / chars, 4)}
        results[lang] = res
        (RESULTS_DIR / f"samples_{lang}.txt").write_text(
            generate_samples(model, sp, test, lang), encoding="utf-8")
        print(name, json.dumps(res["models"]))

    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    md = ["| Language | Model | Test loss | Perplexity | Bits/char |", "|---|---|---|---|---|"]
    for lang, name in [("te", "Telugu"), ("ne", "Nepali")]:
        for key, label in [("unigram", "Unigram"), ("bigram", "Bigram"), ("gpt", "GPT (17.3M)")]:
            m = results[lang]["models"][key]
            md.append(f"| {name} | {label} | {m['loss']} | {m['perplexity']} | {m['bits_per_char']} |")
    (RESULTS_DIR / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    plot_curves(logs)
    print("\n".join(md))


if __name__ == "__main__":
    main()
