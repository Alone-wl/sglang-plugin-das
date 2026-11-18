from __future__ import annotations

from hcu_sglang.utils.detection import is_hcu


def requested():
    if not is_hcu():
        return False
    from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend

    return (
        get_moe_runner_backend().is_deep_gemm()
        and get_moe_a2a_backend().is_deepep()
    )
