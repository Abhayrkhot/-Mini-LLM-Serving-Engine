from __future__ import annotations

import pytest
import torch

from minillm.layers.sampler import sample
from minillm.sampling import SamplingParams


def run(logits, temperature, top_p=1.0, top_k=-1, seed=0):
    n = logits.shape[0]
    g = torch.Generator().manual_seed(seed)
    return sample(
        logits,
        torch.full((n,), float(temperature)),
        torch.full((n,), float(top_p)),
        torch.full((n,), top_k, dtype=torch.long),
        g,
    )


def test_greedy_is_argmax():
    logits = torch.randn(16, 100)
    assert torch.equal(run(logits, 0.0), logits.argmax(-1))


def test_top_k_1_and_tiny_top_p_are_greedy():
    logits = torch.randn(16, 100)
    assert torch.equal(run(logits, 1.0, top_k=1), logits.argmax(-1))
    assert torch.equal(run(logits, 1.0, top_p=1e-6), logits.argmax(-1))


def test_samples_follow_the_softmax_distribution():
    base = torch.tensor([2.0, 1.0, 0.0, -1.0])
    logits = base.repeat(40_000, 1)
    tokens = run(logits, 1.0)
    freq = torch.bincount(tokens, minlength=4).float() / tokens.numel()
    torch.testing.assert_close(freq, base.softmax(-1), atol=0.01, rtol=0)


def test_temperature_sharpens_and_flattens():
    logits = torch.tensor([1.0, 0.0]).repeat(20_000, 1)
    p_cold = (run(logits, 0.25) == 0).float().mean()
    p_hot = (run(logits, 4.0) == 0).float().mean()
    assert p_cold > 0.95 and 0.5 < p_hot < 0.6


def test_top_k_and_top_p_exclude_the_tail():
    logits = torch.tensor([3.0, 2.9, 0.0, -1.0, -2.0]).repeat(5_000, 1)
    assert set(run(logits, 1.0, top_k=2).tolist()) <= {0, 1}
    # Tokens 0 and 1 hold ~0.92 of the mass, so top_p=0.9 keeps exactly those two.
    assert set(run(logits, 1.0, top_p=0.9).tolist()) == {0, 1}


def test_per_row_parameters_in_one_batch():
    logits = torch.randn(3, 50)
    g = torch.Generator().manual_seed(0)
    out = sample(
        logits,
        torch.tensor([0.0, 1.0, 1.0]),
        torch.tensor([1.0, 1.0, 1e-6]),
        torch.tensor([-1, 1, -1]),
        g,
    )
    assert torch.equal(out, logits.argmax(-1))  # greedy, top_k=1, tiny top_p


def test_sampling_params_validation():
    assert SamplingParams(temperature=0).greedy
    assert SamplingParams(top_k=0).top_k == -1
    for bad in (dict(max_tokens=0), dict(temperature=-1), dict(top_p=0), dict(top_k=-5)):
        with pytest.raises(ValueError):
            SamplingParams(**bad)
