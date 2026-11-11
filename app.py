"""Small Gradio demo: type a prompt, pick a language, get a continuation.

  python app.py

Uses local checkpoints (checkpoints/{lang}_best.pt) if present, otherwise
downloads the models from the Hugging Face Hub.
"""
import sys
from pathlib import Path

import gradio as gr
import sentencepiece as spm
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from model import GPT, GPTConfig  # noqa: E402

HUB = {"te": "YogeshNarasimha/telugu-gpt-17m", "ne": "YogeshNarasimha/nepali-gpt-17m"}
device = "cuda" if torch.cuda.is_available() else "cpu"
_cache = {}


def load(lang):
    if lang in _cache:
        return _cache[lang]
    local = ROOT / "checkpoints" / f"{lang}_best.pt"
    if local.exists():
        ckpt_path = local
        tok_path = ROOT / "tokenizers" / f"{lang}_8000.model"
    else:
        from huggingface_hub import hf_hub_download
        ckpt_path = hf_hub_download(HUB[lang], "model.pt")
        tok_path = hf_hub_download(HUB[lang], "tokenizer.model")
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    _cache[lang] = (model, spm.SentencePieceProcessor(model_file=str(tok_path)))
    return _cache[lang]


def generate(language, prompt, max_tokens, temperature, top_k, seed):
    lang = "te" if language == "Telugu" else "ne"
    if not prompt.strip():
        return "Please enter a prompt."
    model, sp = load(lang)
    torch.manual_seed(int(seed))
    ids = torch.tensor([sp.encode(prompt)], device=device)
    out = model.generate(ids, int(max_tokens), float(temperature), int(top_k))[0].tolist()
    return sp.decode([t for t in out if t != sp.eos_id()])


EXAMPLES = [
    ["Telugu", "భారతదేశం", 80, 0.8, 40, 0],
    ["Telugu", "హైదరాబాదు", 80, 0.8, 40, 0],
    ["Nepali", "नेपाल एक", 80, 0.8, 40, 0],
    ["Nepali", "काठमाडौं उपत्यका", 80, 0.8, 40, 0],
]

demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Radio(["Telugu", "Nepali"], value="Telugu", label="Language"),
        gr.Textbox(lines=3, label="Prompt"),
        gr.Slider(10, 200, value=80, step=10, label="New tokens"),
        gr.Slider(0.3, 1.5, value=0.8, step=0.05, label="Temperature"),
        gr.Slider(5, 200, value=40, step=5, label="Top-k"),
        gr.Number(value=0, label="Seed", precision=0),
    ],
    outputs=gr.Textbox(lines=8, label="Generated text"),
    title="Telugu and Nepali GPT (17M, trained from scratch)",
    description="Small language models trained on Wikipedia. Text is fluent but often factually wrong. "
                "Code: github.com/Yogesh-Narasimha/telugu-nepali-lm",
    examples=EXAMPLES,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
