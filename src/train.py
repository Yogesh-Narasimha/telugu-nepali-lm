"""Train the GPT on one language.

  python src/train.py te            # full run
  python src/train.py te --max-iters 50 --tag smoke   # quick test

Reads  data/tokens/{lang}.{train,val}.bin
Writes checkpoints/{lang}{tag}_best.pt  (lowest validation loss)
       checkpoints/{lang}{tag}_last.pt
       results/{lang}{tag}_log.json
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "data" / "tokens"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at(it, a):
    if it < a.warmup:
        return a.lr * (it + 1) / a.warmup
    if it >= a.max_iters:
        return a.min_lr
    ratio = (it - a.warmup) / max(1, a.max_iters - a.warmup)
    return a.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (a.lr - a.min_lr)


@torch.no_grad()
def evaluate(model, data, a, device, ctx):
    model.eval()
    losses = []
    for _ in range(a.eval_batches):
        x, y = get_batch(data, a.batch_size, model.cfg.block_size, device)
        with ctx:
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("lang")
    p.add_argument("--config", default=str(ROOT / "configs" / "gpt_17m.json"))
    p.add_argument("--tag", default="")
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--eval-interval", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)

    cfg = GPTConfig.from_json(a.config)
    model = GPT(cfg).to(device)
    print(f"{a.lang}: {model.num_params() / 1e6:.2f}M params, device={device}, bf16={use_bf16}")

    train = np.memmap(TOK_DIR / f"{a.lang}.train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(TOK_DIR / f"{a.lang}.val.bin", dtype=np.uint16, mode="r")
    tok_per_iter = a.batch_size * a.grad_accum * cfg.block_size
    print(f"train tokens={len(train):,}  tokens/iter={tok_per_iter:,}  "
          f"epochs={a.max_iters * tok_per_iter / len(train):.1f}")

    decay = [p_ for n, p_ in model.named_parameters() if p_.dim() >= 2]
    no_decay = [p_ for n, p_ in model.named_parameters() if p_.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": a.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=a.lr, betas=(0.9, 0.95))

    CKPT_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    name = f"{a.lang}{a.tag}"
    log = {"args": vars(a), "config": cfg.to_dict(), "params": model.num_params(), "train": [], "val": []}
    best_val, t0 = float("inf"), time.time()

    for it in range(a.max_iters + 1):
        lr = lr_at(it, a)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % a.eval_interval == 0 or it == a.max_iters:
            vl = evaluate(model, val, a, device, ctx)
            log["val"].append({"iter": it, "loss": vl, "ppl": math.exp(vl)})
            print(f"iter {it:5d} | val loss {vl:.4f} ppl {math.exp(vl):8.2f} | {time.time() - t0:6.0f}s")
            state = {"model": model.state_dict(), "config": cfg.to_dict(), "iter": it, "val_loss": vl}
            torch.save(state, CKPT_DIR / f"{name}_last.pt")
            if vl < best_val:
                best_val = vl
                torch.save(state, CKPT_DIR / f"{name}_best.pt")
            (RESULTS_DIR / f"{name}_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        if it == a.max_iters:
            break

        loss_acc = 0.0
        for _ in range(a.grad_accum):
            x, y = get_batch(train, a.batch_size, cfg.block_size, device)
            with ctx:
                _, loss = model(x, y)
            (loss / a.grad_accum).backward()
            loss_acc += loss.item() / a.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        log["train"].append({"iter": it, "loss": loss_acc, "lr": lr})
        if it % 20 == 0:
            print(f"iter {it:5d} | train loss {loss_acc:.4f} | lr {lr:.2e}")

    print(f"done. best val loss {best_val:.4f} (ppl {math.exp(best_val):.2f})")


if __name__ == "__main__":
    main()
