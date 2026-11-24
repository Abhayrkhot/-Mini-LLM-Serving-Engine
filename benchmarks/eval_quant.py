"""What 4-bit weights cost in accuracy: perplexity, and agreement with the 16-bit model.

    python benchmarks/eval_quant.py --model Qwen/Qwen2.5-7B-Instruct --output eval_7b.json

Quantizes every decoder projection of the Hugging Face model as
``--quantization w4a16`` does (asymmetric round-to-nearest, groups of 128, scales
rounded to bf16 as the kernel stores them) and runs the dequantized weights in
bf16. So it measures what the 4-bit codes lose; tests/test_gpu.py checks that the
engine's kernel reproduces this dequantized model. The text is the WikiText-2
test split in non-overlapping windows; "top-1 agreement" is how often both models
predict the same next token.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from minillm.quant import dequantize, quantize_rtn

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def wikitext_ids(tokenizer) -> torch.Tensor:  # type: ignore[no-untyped-def]
    """The WikiText-2 test split, joined as ``datasets``-based evaluations join it."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"
    )
    text = "\n\n".join(pq.read_table(path).column("text").to_pylist())
    return torch.tensor(tokenizer(text).input_ids)


@torch.no_grad()
def quantize_in_place(model: torch.nn.Module, group_size: int) -> int:
    n = 0
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.rsplit(".", 1)[-1] in PROJECTIONS:
            codes, scale, zero = quantize_rtn(mod.weight.data, group_size)
            scale = scale.to(torch.bfloat16).float()
            mod.weight.data = dequantize(codes, scale, zero).to(mod.weight.dtype)
            n += 1
    return n


@torch.no_grad()
def score(model: torch.nn.Module, ids: torch.Tensor, window: int) -> tuple[float, torch.Tensor]:
    """Perplexity over ``ids`` and the predicted next token at every scored position."""
    nll, count, top1 = 0.0, 0, []
    for start in range(0, len(ids) - 1, window):
        chunk = ids[start : start + window].to(model.device)
        if len(chunk) < 2:
            break
        logits = model(chunk[None]).logits[0, :-1].float()
        nll += torch.nn.functional.cross_entropy(logits, chunk[1:], reduction="sum").item()
        count += len(chunk) - 1
        top1.append(logits.argmax(-1).cpu())
    return math.exp(nll / count), torch.cat(top1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--output")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = wikitext_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()

    t0 = time.perf_counter()
    ppl16, top16 = score(model, ids, args.window)
    swapped = quantize_in_place(model, args.group_size)
    ppl4, top4 = score(model, ids, args.window)
    result = {
        "model": args.model,
        "text": "wikitext-2-raw-v1 test",
        "tokens": len(ids),
        "window": args.window,
        "group_size": args.group_size,
        "projections_quantized": swapped,
        "bf16_perplexity": ppl16,
        "w4a16_perplexity": ppl4,
        "perplexity_increase": ppl4 / ppl16 - 1,
        "top1_agreement": (top16 == top4).float().mean().item(),
        "seconds": time.perf_counter() - t0,
    }
    print(json.dumps(result))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
