"""Upload each language's model to the HuggingFace Hub (needs `huggingface-cli login` first).

  python src/upload_hf.py            # uploads te and ne
"""
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
GITHUB = "https://github.com/Yogesh-Narasimha/telugu-nepali-lm"
NAMES = {"te": ("Telugu", "te"), "ne": ("Nepali", "ne")}

CARD = """---
language: {code}
license: mit
library_name: pytorch
tags:
  - language-model
  - gpt
  - from-scratch
  - {name_lower}
datasets:
  - wikimedia/wikipedia
---

# {name} GPT (17.3M parameters, trained from scratch)

A small decoder-only transformer language model for {name}, trained from scratch on
{name} Wikipedia (snapshot 20231101). Part of [telugu-nepali-lm]({github}).

## Results (held-out test articles)

| Model | Perplexity (per token) | Bits/char |
|---|---|---|
| Unigram baseline | {uni_ppl} | {uni_bpc} |
| Bigram baseline | {bi_ppl} | {bi_bpc} |
| **This GPT** | **{gpt_ppl}** | **{gpt_bpc}** |

Perplexity depends on the tokenizer, so it is not comparable across languages; use bits/char.
{extra}

## Model

8 layers, 8 heads, width 384, context 256, vocabulary 8,000 (SentencePiece Unigram),
pre-LayerNorm, learned positions, tied embeddings. Best validation checkpoint (iteration {best_iter}).

## Usage

```python
import torch, sentencepiece as spm
from model import GPT, GPTConfig   # model.py is included in this repo

ckpt = torch.load("model.pt", map_location="cpu")
model = GPT(GPTConfig(**ckpt["config"])).eval()
model.load_state_dict(ckpt["model"])
sp = spm.SentencePieceProcessor(model_file="tokenizer.model")

ids = torch.tensor([sp.encode("your prompt")])
out = model.generate(ids, max_new_tokens=60, temperature=0.8, top_k=40)
print(sp.decode(out[0].tolist()))
```

## Limitations

Trained on Wikipedia only (encyclopedic style) with about 8-9M tokens. It writes fluent text but invents
facts, and is not suitable for any real-world use. {limit}
"""

EXTRA = {
    "te": "\nTelugu Wikipedia is very regular (many templated articles), so this perplexity reflects an easy domain, not an unusually strong model.",
    "ne": "\nNepali was still improving when training stopped, so it is undertrained.",
}
LIMIT = {"te": "The low perplexity is partly due to very regular source text.",
         "ne": "It was trained for a fixed 5,000 steps and had not converged."}


def main():
    api = HfApi()
    user = api.whoami()["name"]
    res = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
    for lang, (name, code) in NAMES.items():
        repo = f"{user}/{name.lower()}-gpt-17m"
        m = res[lang]["models"]
        card = CARD.format(
            code=code, name=name, name_lower=name.lower(), github=GITHUB,
            uni_ppl=m["unigram"]["perplexity"], uni_bpc=m["unigram"]["bits_per_char"],
            bi_ppl=m["bigram"]["perplexity"], bi_bpc=m["bigram"]["bits_per_char"],
            gpt_ppl=m["gpt"]["perplexity"], gpt_bpc=m["gpt"]["bits_per_char"],
            best_iter=res[lang]["best_iter"], extra=EXTRA[lang], limit=LIMIT[lang])
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            shutil.copy(ROOT / "checkpoints" / f"{lang}_best.pt", tmp / "model.pt")
            shutil.copy(ROOT / "tokenizers" / f"{lang}_8000.model", tmp / "tokenizer.model")
            shutil.copy(ROOT / "tokenizers" / f"{lang}_8000.vocab", tmp / "tokenizer.vocab")
            shutil.copy(ROOT / "configs" / "gpt_17m.json", tmp / "config.json")
            shutil.copy(ROOT / "src" / "model.py", tmp / "model.py")
            (tmp / "README.md").write_text(card, encoding="utf-8")
            api.create_repo(repo, repo_type="model", exist_ok=True)
            api.upload_folder(folder_path=str(tmp), repo_id=repo, repo_type="model",
                              commit_message="Upload model, tokenizer and model card")
        print("uploaded", f"https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
