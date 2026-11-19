from __future__ import annotations

import os

from hcu_sglang.utils.detection import is_hcu


def requested():
    if "SGLANG_USE_DEEPGEMM_MOE" in os.environ:
        from sglang.srt.utils import get_bool_env_var

        if not get_bool_env_var("SGLANG_USE_DEEPGEMM_MOE"):
            return False
    if not is_hcu():
        return False
    from sglang.srt.layers.moe import get_moe_a2a_backend, get_moe_runner_backend

    return (
        get_moe_runner_backend().is_deep_gemm()
        and get_moe_a2a_backend().is_deepep()
    )
