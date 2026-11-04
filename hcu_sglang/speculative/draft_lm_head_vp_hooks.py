from contextlib import nullcontext
from contextvars import ContextVar

import torch

from hcu_sglang.config import draft_lm_head_vp_size
from hcu_sglang.utils.detection import is_hcu

_ACTIVE_VP = ContextVar("hcu_draft_lm_head_vp", default=None)


def validate_config(view):
    size = draft_lm_head_vp_size()
    if size == 1:
        return
    if not is_hcu():
        raise ValueError("Draft LM-head VP requires the HCU plugin platform")
    checks = {
        "EAGLE topk1": view.speculative_algorithm == "EAGLE"
        and view.speculative_eagle_topk == 1,
        "DP Attention and DP LM Head": view.enable_dp_attention
        and view.enable_dp_lm_head,
        "attention TP1, CP1, PP1": view.tp_size
        == getattr(view, "attn_dp_size", view.dp_size)
        and view.attn_cp_size == 1
        and view.pp_size == 1,
        "node-local VP groups": view.nnodes > 0
        and view.tp_size % (view.nnodes * size) == 0,
        "no token map": view.speculative_token_map is None,
        "no FP32 LM head": not view.enable_fp32_lm_head,
        "no rejection sampling": not view.speculative_use_rejection_sampling,
        "fixed single-layer speculation": not view.speculative_adaptive
        and not view.enable_multi_layer_eagle,
        "no LoRA": not view.enable_lora,
        "decode worker": view.disaggregation_mode == "decode",
        "no draft TP override": getattr(view, "speculative_draft_model_tp_size", None)
        in (None, view.tp_size),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError("Draft LM-head VP requires " + ", ".join(failed))


def around_init_lm_head(original, worker):
    result = original(worker)
    size = draft_lm_head_vp_size()
    if size == 1:
        return result
    from sglang.srt.runtime_context import get_parallel, get_spec

    from hcu_sglang.speculative.draft_lm_head_vp import DraftLMHeadVocabParallelTop1

    parallel = get_parallel()
    spec = get_spec()
    if (
        not is_hcu()
        or torch.distributed.get_world_size() != parallel.tp_size
        or parallel.attn_tp_size != 1
        or parallel.attn_cp_size != 1
        or worker.topk != 1
        or spec.speculative_use_rejection_sampling
        or worker.hot_token_id is not None
    ):
        raise ValueError("Unsupported runtime topology or sampling mode for draft VP")
    model = worker.draft_runner.model
    target_config = worker.target_worker.model_runner.model_config.hf_config
    if "Glm5NextForConditionalGeneration" not in target_config.architectures:
        raise ValueError("HCU draft VP currently supports GLM5.3 Flash only")
    if not hasattr(model, "logits_processor") or not hasattr(model, "lm_head"):
        raise ValueError("Draft VP requires draft logits_processor and lm_head")
    processor = model.logits_processor
    if processor.use_fp32_lm_head or processor.return_full_logits:
        raise ValueError("Draft VP cannot replace an FP32 or full-logits LM head")
    weight = model.lm_head.weight
    if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("Draft VP requires an unquantized floating-point LM head")
    helper = DraftLMHeadVocabParallelTop1(
        full_weight=weight,
        vocab_size=worker.draft_runner.model_config.vocab_size,
        vp_size=size,
        max_rows_per_rank=worker.draft_runner.req_to_token_pool.size,
        local_world_size=parallel.tp_size // parallel.nnodes,
    )
    worker._hcu_draft_lm_head_vp = helper
    processor._hcu_draft_lm_head_vp = helper
    return result


def around_logits_forward(
    original, processor, input_ids, hidden_states, lm_head, logits_metadata,
    aux_hidden_states=None, hidden_states_before_norm=None,
):
    helper = getattr(processor, "_hcu_draft_lm_head_vp", None)
    if helper is None or _ACTIVE_VP.get() is not helper:
        return original(
            processor, input_ids, hidden_states, lm_head, logits_metadata,
            aux_hidden_states, hidden_states_before_norm,
        )
    from sglang.srt.layers import logits_processor as N

    if isinstance(logits_metadata, N.ForwardBatch):
        logits_metadata = N.LogitsMetadata.from_forward_batch(logits_metadata)
    if N._autotune_run_lm_head is False:
        raise RuntimeError("Draft VP does not support skipping LM-head collectives during autotune")
    if not logits_metadata.forward_mode.is_decode_or_idle():
        raise RuntimeError("Draft VP was entered outside decode/idle")
    hidden_states, hidden_states_before_norm = N.layernorm_sp.maybe_exit_gather(
        hidden_states=hidden_states,
        hidden_states_before_norm=hidden_states_before_norm,
        input_ids=input_ids,
        forward_mode=logits_metadata.forward_mode,
    )
    pruned, before_norm, aux_pruned, sample_indices, _, _ = processor._get_pruned_states(
        hidden_states, hidden_states_before_norm, aux_hidden_states, logits_metadata,
    )
    stored = processor._get_hidden_states_to_store(
        hidden_states, hidden_states_before_norm, aux_hidden_states,
        pruned, before_norm, aux_pruned, sample_indices, logits_metadata,
    )
    scores, token_ids = helper.project_top1(
        pruned, lm_head.weight,
        logit_scale=processor.logit_scale,
        final_logit_softcapping=processor.final_logit_softcapping,
    )
    output = N.LogitsProcessorOutput(
        next_token_logits=None, hidden_states=stored,
        mm_input_embeds=logits_metadata.mm_input_embeds,
    )
    output._hcu_draft_top1_ids = token_ids[:, None]
    output._hcu_draft_top1_scores = scores
    return output


def around_draft_forward(original, worker, forward_batch):
    helper = getattr(worker, "_hcu_draft_lm_head_vp", None)
    if helper is None:
        return original(worker, forward_batch)
    token = _ACTIVE_VP.set(helper)
    try:
        if forward_batch.forward_mode.is_idle():
            return worker._draft_forward_idle(forward_batch, forward_batch.spec_info)
        return draft_forward_top1(worker, forward_batch)
    finally:
        _ACTIVE_VP.reset(token)


def draft_forward_top1(worker, forward_batch):
    from sglang.srt.speculative import eagle_worker_v2 as N

    spec_info = forward_batch.spec_info
    topk_index = spec_info.topk_index
    hidden_states = spec_info.hidden_states
    rows = topk_index.shape[0]
    if topk_index.shape != (rows, 1) or rows > worker._topk1_parents_prealloc.shape[0]:
        raise RuntimeError("Draft VP requires topk1 rows within the allocated chain capacity")
    cache_loc = N.per_step_draft_out_cache_loc(
        forward_batch.out_cache_loc, forward_batch.batch_size, 1,
        worker.speculative_num_steps,
    )
    tokens = torch.empty(
        (rows, worker.speculative_num_steps), dtype=topk_index.dtype,
        device=topk_index.device,
    )
    tokens[:, :1].copy_(topk_index)
    with N.IndexTopKShareState.mtp_iteration(
        forward_batch, enabled=worker.index_share_for_mtp_iteration,
        keep_carry_seed=worker.seed_dsa_topk_from_draft_extend,
    ):
        for step in range(worker.speculative_num_steps - 1):
            forward_batch.input_ids = topk_index.flatten()
            forward_batch.out_cache_loc = cache_loc[step]
            spec_info.hidden_states = hidden_states
            canary = worker.draft_runner.canary_manager
            canary_context = (
                canary.with_active_single_forward_manager(step)
                if canary is not None else nullcontext()
            )
            with N.forward_context(N.ForwardContext(
                attn_backend=worker.draft_attn_backend.attn_backends[step],
            )), canary_context:
                output = worker.draft_runner.forward(forward_batch).logits_output
            ids = getattr(output, "_hcu_draft_top1_ids", None)
            if ids is None or ids.shape[0] < rows:
                raise RuntimeError("Draft VP top1 output is missing or shorter than active rows")
            if output.hidden_states is None or output.hidden_states.shape[0] < rows:
                raise RuntimeError("Draft VP hidden states are shorter than active rows")
            topk_index = ids[:rows]
            hidden_states = output.hidden_states[:rows]
            N.maybe_detect_nan(output._hcu_draft_top1_scores, "draft VP top1")
            N.maybe_detect_inf(output._hcu_draft_top1_scores, "draft VP top1")
            N.maybe_detect_oob(
                topk_index, 0, worker.draft_runner.model_config.vocab_size,
                "draft VP token id",
            )
            tokens[:, step + 1 : step + 2].copy_(topk_index)
            forward_batch.positions.add_(1)
            if worker.draft_runner.model_config.model_is_mrope:
                forward_batch.mrope_positions.add_(1)
    return (
        worker._topk1_parents_prealloc[:rows],
        worker._topk1_score_indices_prealloc[:rows],
        tokens, None,
    )


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for target, hook in (
        ("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.init_lm_head", around_init_lm_head),
        ("sglang.srt.speculative.eagle_worker_v2.EagleDraftWorker.draft_forward", around_draft_forward),
        ("sglang.srt.layers.logits_processor.LogitsProcessor.forward", around_logits_forward),
    ):
        HookRegistry.register(target, hook, HookType.AROUND)
