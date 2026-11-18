# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import time

from hcu_sglang.glm5_next_policy import is_glm5_next_hcu
from hcu_sglang.utils.detection import is_hcu


_original_dsa_backend = None
logger = logging.getLogger(__name__)


def around_create_dsa_backend(original, runner):
    if is_glm5_next_hcu(runner.model_config.hf_config):
        from hcu_sglang.layers.attention.glm5_next.dsa_backend import (
            NativeSparseAttnBackend,
        )

        return NativeSparseAttnBackend(runner)
    return original(runner)


def create_hcu_glm5_backend(runner):
    return around_create_dsa_backend(_original_dsa_backend, runner)


def around_create_dsa_decode_backend(original, factory):
    runner = factory.draft_model_runner
    if not is_glm5_next_hcu(runner.model_config.hf_config):
        return original(factory)

    from hcu_sglang.layers.attention.glm5_next.dsa_backend import (
        NativeSparseAttnMultiStepBackend,
    )

    return "dsa", NativeSparseAttnMultiStepBackend(
        runner,
        factory.topk,
        factory.speculative_num_steps,
        seed_dsa_topk_from_draft_extend=factory.seed_dsa_topk_from_draft_extend,
    )


def around_create_dsa_prefill_backend(original, factory):
    runner = factory.draft_model_runner
    if not is_glm5_next_hcu(runner.model_config.hf_config):
        return original(factory)

    from hcu_sglang.layers.attention.glm5_next.dsa_backend import NativeSparseAttnBackend

    return "dsa", NativeSparseAttnBackend(
        runner,
        skip_prefill=False,
        seed_dsa_topk_from_draft_extend=factory.seed_dsa_topk_from_draft_extend,
    )


def after_capture_draft_graphs(_result, worker):
    if (
        worker.cuda_graph_runner_for_draft_extend is not None
        or not is_glm5_next_hcu(worker.draft_runner.model_config.hf_config)
    ):
        return

    from hcu_sglang.layers.attention.glm5_next.dsa_backend import NativeSparseAttnBackend
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )
    from sglang.srt.runtime_context import get_model, get_server_args
    from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
        EAGLEDraftExtendCudaGraphRunner,
    )
    from sglang.srt.utils import get_available_gpu_memory, log_info_on_rank0

    if (
        not isinstance(worker.draft_extend_attn_backend, NativeSparseAttnBackend)
        or get_server_args().disaggregation_mode != "decode"
        or worker.device != "cuda"
        or check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED)
        or get_model().model_impl == "mindspore"
        or envs.SGLANG_DISABLE_DRAFT_EXTEND_CUDA_GRAPH.get()
    ):
        return

    # The upstream backend allowlist does not include plugin DSA backends.
    begin = time.perf_counter()
    before = get_available_gpu_memory(worker.device, worker.gpu_id)
    log_info_on_rank0(logger, "Capture GLM5 Draft Extend CUDA graph begin")
    worker.cuda_graph_runner_for_draft_extend = EAGLEDraftExtendCudaGraphRunner(worker)
    elapsed = time.perf_counter() - begin
    after = get_available_gpu_memory(worker.device, worker.gpu_id)
    worker._specialized_graph_memory_usage["draft_extend"] = before - after
    worker._specialized_graph_time_usage["draft_extend"] = elapsed
    log_info_on_rank0(
        logger, f"Capture GLM5 Draft Extend CUDA graph end. elapsed={elapsed:.2f} s"
    )


def register() -> None:
    if not is_hcu():
        return
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    global _original_dsa_backend
    if ATTENTION_BACKENDS["dsa"] is not create_hcu_glm5_backend:
        _original_dsa_backend = ATTENTION_BACKENDS["dsa"]
        ATTENTION_BACKENDS["dsa"] = create_hcu_glm5_backend

    HookRegistry.register(
        "sglang.srt.layers.attention.attention_registry.create_dsa_backend",
        around_create_dsa_backend,
        HookType.AROUND,
    )
    for method, hook in (
        ("_create_dsa_decode_backend", around_create_dsa_decode_backend),
        ("_create_dsa_prefill_backend", around_create_dsa_prefill_backend),
    ):
        HookRegistry.register(
            f"sglang.srt.speculative.draft_utils.DraftBackendFactory.{method}",
            hook,
            HookType.AROUND,
        )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker._capture_cuda_graphs",
        after_capture_draft_graphs,
        HookType.AFTER,
    )
