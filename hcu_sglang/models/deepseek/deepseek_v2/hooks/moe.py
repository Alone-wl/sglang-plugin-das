"""DeepseekV2MoE method hooks."""

from __future__ import annotations

from hcu_sglang.models.deepseek.hook_utils import (
    register_method_replaces_from_path,
)

_BASE = "sglang.srt.models.deepseek_v2"
_SRC = "hcu_sglang.models.deepseek.deepseek_v2.methods.moe"

_MOE_METHODS = (
    "__init__",
    "forward",
    "forward_normal",
    "forward_deepep",
    "forward_normal_dual_stream",
    "_forward_shared_experts",
)


def register_hooks() -> int:
    from hcu_sglang.utils.detection import is_hcu

    if is_hcu():
        from sglang.srt.models.deepseek_v2 import DeepseekV2MoE
        from hcu_sglang.models.deepseek.deepseek_v2.methods import moe

        DeepseekV2MoE.set_shared_expert_input_quant_args = moe.set_shared_expert_input_quant_args
        DeepseekV2MoE.clear_shared_expert_input_quant_args = moe.clear_shared_expert_input_quant_args
    return register_method_replaces_from_path(
        _BASE,
        "DeepseekV2MoE",
        _MOE_METHODS,
        _SRC,
    )
