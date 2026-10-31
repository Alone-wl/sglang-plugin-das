"""DCU W8A8 FP8 quantization adaptations."""

from __future__ import annotations


def register_hooks() -> None:
    from . import fp8_linear, fp8_quant, fp8_block, fp8_mxscale
    fp8_quant.register_hooks()
    fp8_linear.register_hooks()
    fp8_block.register_hooks()
    fp8_mxscale.register_hooks()
