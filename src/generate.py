"""Generate text from a trained model.

  python src/generate.py te "తెలుగు భాష" --tokens 80
  python src/generate.py ne "नेपाल एक" --temperature 0.7
"""
import argparse
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import GPT, GPTConfig  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def load(lang, device):
    ckpt = torch.load(ROOT / "checkpoints" / f"{lang}_best.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "tokenizers" / f"{lang}_{ckpt['config']['vocab_size']}.model"))
    return model, sp


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("lang", choices=["te", "ne"])
    p.add_argument("prompt")
    p.add_argument("--tokens", type=int, default=80)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(a.seed)
    model, sp = load(a.lang, device)
    ids = torch.tensor([sp.encode(a.prompt)], device=device)
    out = model.generate(ids, a.tokens, a.temperature, a.top_k)[0].tolist()
    print(sp.decode(out))
