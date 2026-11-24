"""4-bit weights: round-to-nearest quantization at load time, run by a Triton W4A16 kernel.

The weights of every decoder projection (q/k/v, o, gate/up, down) are quantized
to 4-bit codes with one scale and zero point per 128 input columns of each
output row (asymmetric round-to-nearest, as GPTQ checkpoints store them). The
embedding, final norm and LM head stay 16-bit. Activations stay 16-bit:
``y = x @ dequant(W)^T`` happens inside the kernel, so a decode step reads a
quarter of the weight bytes.
"""

from __future__ import annotations

import torch
from torch import nn


def quantize_rtn(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, ...]:
    """Asymmetric 4-bit round-to-nearest, per output row and group of input columns.

    Returns uint8 codes ``[out, in]`` and float32 scale and zero ``[out, groups]``;
    ``weight ~= (codes - zero) * scale``, with 0.0 exactly representable.
    """
    out_features, in_features = weight.shape
    if in_features % group_size:
        raise ValueError(f"in_features ({in_features}) is not a multiple of {group_size}")
    w = weight.float().reshape(out_features, in_features // group_size, group_size)
    wmin = w.amin(dim=-1).clamp(max=0)
    wmax = w.amax(dim=-1).clamp(min=0)
    scale = (wmax - wmin) / 15
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    zero = torch.round(-wmin / scale).clamp(0, 15)
    codes = (torch.round(w / scale.unsqueeze(-1)) + zero.unsqueeze(-1)).clamp(0, 15)
    return codes.to(torch.uint8).reshape(weight.shape), scale, zero


def dequantize(codes: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
    out_features, in_features = codes.shape
    groups = scale.shape[1]
    q = codes.float().reshape(out_features, groups, in_features // groups)
    return ((q - zero.unsqueeze(-1)) * scale.unsqueeze(-1)).reshape(codes.shape)


class QuantLinear(nn.Module):
    """``nn.Linear`` with 4-bit group-wise weights in the GPTQ layout; one kernel launch."""

    qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: torch.Tensor

    def __init__(self, linear: nn.Linear, group_size: int = 128) -> None:
        super().__init__()
        from minillm.quant.w4a16 import pack_for_kernel

        codes, scale, zero = quantize_rtn(linear.weight.data, group_size)
        qweight, scales, qzeros = pack_for_kernel(codes, scale, zero, linear.weight.dtype)
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.group_size = group_size
        self.register_buffer("qweight", qweight)
        self.register_buffer("scales", scales)
        self.register_buffer("qzeros", qzeros)
        self.bias = linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minillm.quant.w4a16 import w4a16_matmul

        y = w4a16_matmul(x, self.qweight, self.scales, self.qzeros, self.group_size)
        return y + self.bias if self.bias is not None else y

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, group={self.group_size}"


def quantize_model(model: nn.Module, group_size: int = 128) -> int:
    """Swap every decoder projection for a QuantLinear; returns how many were swapped."""
    from minillm.quant.w4a16 import reserve_workspace

    names = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
    swapped, widest, device = 0, 0, None
    for module in list(model.modules()):
        for name in names:
            linear = getattr(module, name, None)
            if isinstance(linear, nn.Linear):
                q = QuantLinear(linear, group_size)
                setattr(module, name, q)
                widest = max(widest, q.out_features)
                device = q.qweight.device
                swapped += 1
    if swapped and device is not None and device.type == "cuda":
        # The kernel's split-K workspace, allocated before any CUDA graph capture.
        reserve_workspace(device, widest)
    return swapped
