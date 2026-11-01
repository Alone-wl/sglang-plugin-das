# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, List
from sglang.srt.managers.scheduler_components import dp_attn as N
from sglang.srt.managers.schedule_batch import ScheduleBatch as _NativeScheduleBatch
from hcu_sglang.cp_token_counts import folded_replay_rows
from hcu_sglang.utils.detection import is_hcu

_NATIVE_INFO = [N.MLPSyncBatchInfo]


@dataclass
class ScheduleBatch(_NativeScheduleBatch):
    global_cp_num_tokens: Optional[List[int]] = None


@dataclass
class MLPSyncBatchInfo(_NATIVE_INFO[0]):
    cp_num_tokens: int = 0
    global_cp_num_tokens: Optional[List[int]] = None

    def _get_local_tensor(self, device, dtype=N.torch.int64) -> N.torch.Tensor:
        return N.torch.tensor(
            [
                self.num_tokens,
                self.num_tokens_for_logprob,
                int(self.can_run_decode_cuda_graph),
                int(self.is_extend_in_batch),
                int(self.local_can_run_tbo),
                self.local_forward_mode,
                int(self.can_run_prefill_cuda_graph),
                self.prefill_cuda_graph_max_prefix_len,
                int(self.can_run_draft_cuda_graph),
                int(self.is_health_check),
                self.attention_pairs,
                self.cp_num_tokens,
            ],
            device=device,
            dtype=dtype,
        )

    def _get_fallback_tensor(self, device, dtype=N.torch.int64) -> N.torch.Tensor:
        return N.torch.tensor(
            [
                0,  # num_tokens
                0,  # num_tokens_for_logprob
                1,  # can_run_decode_cuda_graph
                0,  # is_extend_in_batch
                1,  # local_can_run_tbo
                N.ForwardMode.IDLE.value,  # local_forward_mode
                0,  # can_run_prefill_cuda_graph
                0,  # prefill_cuda_graph_max_prefix_len
                1,  # can_run_draft_cuda_graph
                0,  # is_health_check
                0,  # attention_pairs
                0,  # cp_num_tokens
            ],
            device=device,
            dtype=dtype,
        )

    def finalize_local(self):
        """Populate gather-derived metadata from the sole attention-DP rank."""
        self.tp0_info_cpu = self._get_local_tensor(device="cpu").view(1, -1)
        self.global_num_tokens = [self.num_tokens]
        self.global_num_tokens_for_logprob = [self.num_tokens_for_logprob]
        self.global_attention_pairs = [self.attention_pairs]
        self.global_cp_num_tokens = [self.cp_num_tokens]
        self.any_health_check = self.is_health_check
        if N._ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                self.tp0_info_cpu[:, 5].tolist()
            )

    def all_gather(
        self,
        device,
        group: N.torch.distributed.ProcessGroup,
        use_all_reduce: bool = False,
    ):
        sync_start = N.time.perf_counter()
        local_info_tensor = self._get_local_tensor(device=device)
        fallback_tensor = self._get_fallback_tensor(device=device)
        info_width = local_info_tensor.numel()
        # Inactive max_world_size slots must decode as IDLE. repeat() (not
        # expand().contiguous()) so the buffer never aliases fallback_tensor:
        # at world size 1 the expanded view is already contiguous, contiguous()
        # is a no-op, and the masked fallback writes below would then read and
        # write the same storage.
        global_info_tensor = fallback_tensor.repeat(
            self.num_dp_ranks, self.tp_size * self.cp_size, 1
        )

        if use_all_reduce:
            # Admission can expose different WORLD sizes; use fixed global slots.
            global_info_tensor.zero_()
            flat_info = global_info_tensor.view(-1, info_width)
            rank = N.torch.distributed.get_rank(group)
            if 0 <= rank < flat_info.shape[0]:
                flat_info[rank] = local_info_tensor
            N.torch.distributed.all_reduce(
                global_info_tensor,
                op=N.torch.distributed.ReduceOp.SUM,
                group=group,
            )
            missing = flat_info.abs().sum(dim=1) == 0
            flat_info[missing] = fallback_tensor
        else:
            exchange_sync_rows(
                global_info_tensor.flatten(),
                local_info_tensor,
                group=group,
            )

        tp_info = global_info_tensor.view(
            self.num_dp_ranks * self.tp_size * self.cp_size, info_width
        )
        num_ranks_in_tp_info = tp_info.shape[0]
        if device == "cpu":
            tp_active_ranks = N.get_parallel().tp_group.active_ranks_cpu
        else:
            tp_active_ranks = N.get_parallel().tp_group.active_ranks
        if tp_active_ranks.shape[0] < num_ranks_in_tp_info:
            tp_active_ranks = N.torch.ones(
                num_ranks_in_tp_info,
                dtype=tp_active_ranks.dtype,
                device=tp_active_ranks.device,
            )
        tp_info[tp_active_ranks[:num_ranks_in_tp_info] == 0] = fallback_tensor

        # One D2H for every field: each `.item()` / `.tolist()` on a device
        # tensor is its own stream sync. Copy the whole tensor, not the
        # `[:, 0, :]` slice -- that slice is non-contiguous once
        # attn_tp * attn_cp > 1, adding a gather kernel inside the wait.
        tp0_info_cpu = global_info_tensor.cpu()[:, 0, :]
        self.tp0_info_cpu = tp0_info_cpu
        self.global_num_tokens = tp0_info_cpu[:, 0].tolist()
        self.global_num_tokens_for_logprob = tp0_info_cpu[:, 1].tolist()
        self.global_cp_num_tokens = tp0_info_cpu[:, 11].tolist()
        if self.cp_size > 1:
            from hcu_sglang.cp_token_counts import normalize_dp_cp_token_counts

            self.global_cp_num_tokens = normalize_dp_cp_token_counts(
                self.global_num_tokens, self.global_cp_num_tokens
            )
        self.can_run_decode_cuda_graph = bool(tp0_info_cpu[:, 2].min())
        self.is_extend_in_batch = bool(tp0_info_cpu[:, 3].max())
        self.can_run_prefill_cuda_graph = bool(tp0_info_cpu[:, 6].min())
        self.prefill_cuda_graph_max_prefix_len = int(tp0_info_cpu[:, 7].max())
        self.can_run_draft_cuda_graph = bool(tp0_info_cpu[:, 8].min())
        self.any_health_check = bool(tp0_info_cpu[:, 9].max())
        self.global_attention_pairs = tp0_info_cpu[:, 10].tolist()
        self.sync_wait_seconds = N.time.perf_counter() - sync_start
        if N._ENABLE_METRICS_DP_ATTENTION:
            self.dp_cooperation_info = DPCooperationInfo.create(
                tp0_info_cpu[:, 5].tolist()
            )


def exchange_sync_rows(
    global_flat: N.torch.Tensor,
    local_row: N.torch.Tensor,
    group: N.torch.distributed.ProcessGroup,
) -> None:
    """Fill ``global_flat`` with every rank's ``local_row`` in group-rank order,
    like ``all_gather_single``. On gloo the all-gather is a ring of world_size - 1
    serial hops; gather + broadcast is one hop plus a log-depth tree. At two
    ranks the ring is the single hop, so it stays."""
    world_size = N.torch.distributed.get_world_size(group)
    if local_row.device.type != "cpu" or world_size <= 2:
        N.all_gather_single(global_flat, local_row, group=group)
        return
    root = N.torch.distributed.get_global_rank(group, 0)
    rows = global_flat.view(world_size, local_row.numel())
    if N.torch.distributed.get_rank(group) == 0:
        N.torch.distributed.gather(
            local_row, gather_list=list(rows.unbind(0)), dst=root, group=group
        )
    else:
        N.torch.distributed.gather(local_row, dst=root, group=group)
    N.torch.distributed.broadcast(global_flat, src=root, group=group)


def _resolve_elastic_world_num_dp_ranks(
    num_dp_ranks: int,
    *,
    group: N.torch.distributed.ProcessGroup,
    local_num_tokens: int,
    local_forward_mode: int,
) -> int:
    if not N.world_dp_gather_enabled():
        return num_dp_ranks

    from sglang.srt.elastic_ep.elastic_ep import ElasticEPStateManager

    live_num_dp_ranks = N.dp_gather_width()
    effective_ep_size = ElasticEPStateManager.get_effective_ep_size()
    # Query live membership because elastic joins can expand WORLD.
    from hcu_sglang.cp_token_counts import attn_replica_size

    replica_size = attn_replica_size()
    world_size = N.torch.distributed.get_world_size(group)

    if live_num_dp_ranks * replica_size != effective_ep_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync num_dp_ranks is out of sync: "
            f"rank={N.torch.distributed.get_rank(group)} "
            f"live_num_dp_ranks={live_num_dp_ranks} "
            f"effective_ep_size={effective_ep_size} "
            f"attn_replica_size={replica_size} "
            f"world_size={world_size} configured_num_dp_ranks={num_dp_ranks} "
            f"local_num_tokens={local_num_tokens} "
            f"local_forward_mode={local_forward_mode}"
        )
    if effective_ep_size > world_size:
        raise RuntimeError(
            "[Elastic EP] WORLD MLP sync EP size exceeds WORLD size: "
            f"rank={N.torch.distributed.get_rank(group)} "
            f"live_num_dp_ranks={live_num_dp_ranks} world_size={world_size} "
            f"effective_ep_size={effective_ep_size} "
            f"attn_replica_size={replica_size}"
        )

    return live_num_dp_ranks


def _update_gather_batch(
    batch: ScheduleBatch,
    mlp_sync_info: MLPSyncBatchInfo,
    require_mlp_tp_gather: bool,
    draft_require_mlp_tp_gather: Optional[bool] = None,
    skip_global_metadata=False,
):
    if not N.require_mlp_tp_gather:
        batch.global_num_tokens = [mlp_sync_info.num_tokens]
        batch.global_num_tokens_for_logprob = [mlp_sync_info.num_tokens_for_logprob]
        batch.global_cp_num_tokens = [mlp_sync_info.cp_num_tokens]
    else:
        batch.global_num_tokens = mlp_sync_info.global_num_tokens
        batch.global_num_tokens_for_logprob = (
            mlp_sync_info.global_num_tokens_for_logprob
        )
        batch.global_cp_num_tokens = mlp_sync_info.global_cp_num_tokens
    if N.get_parallel().enable_cp_tp_group_sharing:
        batch.global_cp_num_tokens = None

    # Reuse the same all-gather result for a draft model whose A2A backend
    # requires a different local/full token-count representation.
    if draft_require_mlp_tp_gather is not None:
        if draft_require_mlp_tp_gather:
            batch.draft_global_num_tokens = mlp_sync_info.global_num_tokens
            batch.draft_global_num_tokens_for_logprob = (
                mlp_sync_info.global_num_tokens_for_logprob
            )
        else:
            batch.draft_global_num_tokens = [mlp_sync_info.num_tokens]
            batch.draft_global_num_tokens_for_logprob = [
                mlp_sync_info.num_tokens_for_logprob
            ]
    if N.envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get():
        # Fresh counts have not yet been adjusted by the coordination plan.
        batch.dp_spec_prefill_coordination_applied = False
        info = mlp_sync_info.tp0_info_cpu
        batch.dp_spec_prefill_coordination_metadata = (
            mlp_sync_info.global_num_tokens,
            mlp_sync_info.global_num_tokens_for_logprob,
            info[:, 3],
        )
    if not skip_global_metadata:
        batch.is_extend_in_batch = mlp_sync_info.is_extend_in_batch
        batch.tbo_split_seq_index = mlp_sync_info.tbo_split_seq_index
        batch.global_forward_mode = mlp_sync_info.global_forward_mode

    # Check forward mode for cuda graph
    batch.can_run_decode_cuda_graph = mlp_sync_info.can_run_decode_cuda_graph
    batch.can_run_dp_draft_cuda_graph = mlp_sync_info.can_run_draft_cuda_graph
    batch.can_run_dp_prefill_cuda_graph = mlp_sync_info.can_run_prefill_cuda_graph
    batch.dp_prefill_cuda_graph_max_prefix_len = (
        mlp_sync_info.prefill_cuda_graph_max_prefix_len
    )


def prepare_mlp_sync_batch_raw(
    local_batch: ScheduleBatch,
    model_runner: ModelRunner,
    get_idle_batch: Callable[[], ScheduleBatch],
    disable_cuda_graph: bool,
    require_mlp_tp_gather: bool,
    disable_overlap_schedule: bool,
    offload_tags: set[str],
    draft_require_mlp_tp_gather: Optional[bool] = None,
    dwdp: bool = False,
    sync_wait_carry: Optional[DPSyncWaitCarry] = None,
):
    parallel = N.get_parallel()
    num_dp_ranks = parallel.num_dp_ranks
    attn_tp_size = parallel.attn_tp_size
    tp_group = parallel.tp_group
    # Check if other DP workers have running batches
    if (
        local_batch is None
        or local_batch.forward_mode.is_prebuilt()
        or local_batch.forward_mode.is_idle()
    ):
        num_tokens = 0
        num_tokens_for_logprob = 0
    elif local_batch.forward_mode.is_decode():
        num_tokens = local_batch.batch_size()
        num_tokens_for_logprob = num_tokens
    else:
        num_tokens = local_batch.extend_num_tokens
        if (
            local_batch.encoder_swa_reset is not None
            and getattr(model_runner.attn_backend, "folds_encoder_swa_replay", False)
        ):
            num_tokens += folded_replay_rows(local_batch)
        num_tokens_for_logprob = sum(
            # We should have at least 1 token for sample in every case.
            max(extend_len - logprob_start_len, 1)
            for logprob_start_len, extend_len in zip(
                local_batch.extend_logprob_start_lens,
                local_batch.extend_lens,
            )
        )
        assert (
            local_batch.return_logprob
            or num_tokens_for_logprob == local_batch.batch_size()
        )

    cp_num_tokens = 0
    if local_batch is not None and local_batch.forward_mode.is_extend():
        from hcu_sglang.cp_token_counts import get_cp_v2_physical_token_count

        cp_num_tokens = get_cp_v2_physical_token_count(
            num_tokens=num_tokens,
            extend_seq_lens=local_batch.extend_lens,
            cp_size=N.get_parallel().attn_cp_size,
        )

    # With a single DP replica and no residual attention-TP/MLP-TP gather,
    # every CP rank receives the same work through the CP control broadcast.
    # The scheduler metadata all-gather is therefore redundant and can hang
    # on HCU CPU process groups during the idle loop.  Reuse the established
    # skip path, which records each rank's local token count as required by CP.
    skip_all_gather = N.envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get() or (
        N.get_parallel().num_dp_ranks == 1
        and attn_tp_size == 1
        and not N.require_mlp_tp_gather
    )
    can_run_decode_cuda_graph = N._local_decode_cuda_graph_vote(
        local_batch=local_batch, disable_cuda_graph=disable_cuda_graph
    )
    can_run_draft_cuda_graph = N._spec_input_cuda_graph_compatible(local_batch)
    breakable_prefill = N.check_cuda_graph_backend(N.Phase.PREFILL, N.Backend.BREAKABLE)
    full_prefill = N.check_cuda_graph_backend(N.Phase.PREFILL, N.Backend.FULL)
    coordinated_prefill = breakable_prefill or full_prefill
    prefill_graph_runner = (
        model_runner.prefill_cuda_graph_runner if coordinated_prefill else None
    )
    prefill_cuda_graph_max_prefix_len = (
        max(local_batch.prefix_lens, default=0)
        if full_prefill
        and local_batch is not None
        and local_batch.forward_mode in (N.ForwardMode.EXTEND, N.ForwardMode.MIXED)
        else 0
    )
    can_run_prefill_cuda_graph = N._local_prefill_cuda_graph_vote(
        local_batch=local_batch,
        num_tokens=num_tokens,
        prefill_graph_runner=prefill_graph_runner,
        coordinated_prefill=coordinated_prefill,
        breakable_prefill=breakable_prefill,
        spec_algorithm=model_runner.spec_algorithm,
        model_config=model_runner.model_config,
    )

    is_extend_in_batch = local_batch.forward_mode.is_extend() if local_batch else False
    is_health_check = bool(
        local_batch is not None
        and local_batch.reqs
        and all(N.is_health_check_generate_req(req) for req in local_batch.reqs)
    )
    if local_batch is not None:
        local_batch.is_extend_in_batch = is_extend_in_batch

    tbo_preparer = N.TboDPAttentionPreparer()
    use_world_group = N.world_dp_gather_enabled()
    if use_world_group:
        world = parallel.world_group
        group = N.torch.distributed.group.WORLD
        device = world.device
    elif len(offload_tags) == 0 and (
        disable_overlap_schedule
        or N.envs.SGLANG_NCCL_ALL_GATHER_IN_OVERLAP_SCHEDULER_SYNC_BATCH.get()
    ):
        group = tp_group.device_group
        device = tp_group.device
    else:
        group = tp_group.cpu_group
        device = "cpu"

    local_can_run_tbo, local_forward_mode = tbo_preparer.prepare_all_gather(local_batch)
    if use_world_group:
        num_dp_ranks = _resolve_elastic_world_num_dp_ranks(
            num_dp_ranks,
            group=group,
            local_num_tokens=num_tokens,
            local_forward_mode=local_forward_mode,
        )
    skip_all_gather = N.should_skip_scheduler_all_gather(num_dp_ranks)

    mlp_sync_info = MLPSyncBatchInfo(
        num_dp_ranks=num_dp_ranks,
        tp_size=attn_tp_size,
        cp_size=parallel.attn_cp_size,
        num_tokens=num_tokens,
        num_tokens_for_logprob=num_tokens_for_logprob,
        cp_num_tokens=cp_num_tokens,
        can_run_decode_cuda_graph=can_run_decode_cuda_graph,
        can_run_draft_cuda_graph=can_run_draft_cuda_graph,
        can_run_prefill_cuda_graph=can_run_prefill_cuda_graph,
        is_extend_in_batch=is_extend_in_batch,
        local_can_run_tbo=local_can_run_tbo,
        local_forward_mode=local_forward_mode,
        prefill_cuda_graph_max_prefix_len=prefill_cuda_graph_max_prefix_len,
        is_health_check=is_health_check,
        # Gathered unconditionally like the other columns, so a per-process
        # metrics flag cannot make ranks disagree on what the tensor holds.
        attention_pairs=N._local_attention_pairs(local_batch),
    )

    if num_dp_ranks == 1:
        mlp_sync_info.finalize_local()
    elif not skip_all_gather:
        mlp_sync_info.all_gather(
            device=device,
            group=group,
            use_all_reduce=use_world_group,
        )

    metadata_ready = mlp_sync_info.tp0_info_cpu is not None
    if metadata_ready:
        mlp_sync_info.tbo_split_seq_index, mlp_sync_info.global_forward_mode = (
            tbo_preparer.compute_output(
                mlp_sync_info.tp0_info_cpu[:, 4:6],
            )
        )
    # Decide whether to emit idle batch
    if skip_all_gather:
        # Skip idle batch when attn-dp=1 (and always under DWDP: ranks run independently)
        need_idle_batch = not dwdp and num_dp_ranks > 1
    else:
        need_idle_batch = max(mlp_sync_info.global_num_tokens) > 0

    batch_to_gather = local_batch
    if need_idle_batch:
        if local_batch is None:
            batch_to_gather = local_batch = get_idle_batch()
        elif local_batch.forward_mode.is_prebuilt():
            # NOTE: for prebuilt batch, we add an inner idle batch to run MLP sync
            batch_to_gather = local_batch.inner_idle_batch = get_idle_batch()

    if batch_to_gather is not None:
        _update_gather_batch(
            batch_to_gather,
            mlp_sync_info,
            N.require_mlp_tp_gather,
            draft_require_mlp_tp_gather,
            skip_global_metadata=not metadata_ready,
        )

    # Set on `local_batch`, not `batch_to_gather`: for PREBUILT batches the
    # scheduler's `last_batch` is the prebuilt batch, not its inner idle batch.
    if local_batch is not None and metadata_ready:
        local_batch.recv_skipper_forward_mode = (
            N.SchedulerRecvSkipper.derive_forward_mode(
                mlp_sync_info.tp0_info_cpu[:, 5].tolist()
            )
        )

    if N._ENABLE_METRICS_DP_ATTENTION and local_batch is not None:
        local_batch.dp_cooperation_info = mlp_sync_info.dp_cooperation_info
        local_batch.dp_balance_stats = None
    if N._ENABLE_METRICS_DP_ATTENTION and num_dp_ranks > 1 and not skip_all_gather:
        sync_wait_seconds = mlp_sync_info.sync_wait_seconds
        if sync_wait_carry is not None:
            sync_wait_seconds += sync_wait_carry.seconds
            sync_wait_carry.seconds = 0.0
        # Health-check probes run on an otherwise idle engine; recording them
        # would show maximal imbalance where there is no load.
        if need_idle_batch and not mlp_sync_info.any_health_check:
            local_batch.dp_balance_stats = N.DPBalanceStats.create(
                local_tokens=num_tokens,
                global_num_tokens=mlp_sync_info.global_num_tokens,
                local_attention_pairs=mlp_sync_info.attention_pairs,
                global_attention_pairs=mlp_sync_info.global_attention_pairs,
                sync_wait_seconds=sync_wait_seconds,
            )
        elif local_batch is None and sync_wait_carry is not None:
            # No batch yet: this iteration may gather again, or end in on_idle.
            sync_wait_carry.seconds = sync_wait_seconds

    return local_batch


def around__resolve_elastic_world_num_dp_ranks(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return _resolve_elastic_world_num_dp_ranks(*args, **kwargs)


def around__update_gather_batch(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return _update_gather_batch(*args, **kwargs)


def around_prepare_mlp_sync_batch_raw(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return prepare_mlp_sync_batch_raw(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn._resolve_elastic_world_num_dp_ranks", around__resolve_elastic_world_num_dp_ranks, HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn._update_gather_batch", around__update_gather_batch, HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn.prepare_mlp_sync_batch_raw", around_prepare_mlp_sync_batch_raw, HookType.AROUND)
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn.MLPSyncBatchInfo", MLPSyncBatchInfo, HookType.REPLACE)
    HookRegistry.register("sglang.srt.managers.schedule_batch.ScheduleBatch", ScheduleBatch, HookType.REPLACE)
