from functools import cache
from pathlib import Path

import torch


@cache
def _module():
    from torch.utils.cpp_extension import load_inline

    path = Path(__file__).parents[1] / "jit_overlay/csrc/memory/hcu_allocation.cu"
    return load_inline(
        name="hcu_plugin_allocation",
        cpp_sources="",
        cuda_sources=path.read_text(),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-DUSE_ROCM=1"],
        verbose=False,
    )


@cache
def _extend_kernel():
    try:
        from lightop import op
    except ModuleNotFoundError as error:
        if error.name != "lightop":
            raise
        return _module().hcu_alloc_extend_kernel
    kernel = getattr(op, "hcu_alloc_extend_kernel", None)
    return kernel if kernel is not None else _module().hcu_alloc_extend_kernel


def prepare():
    _module()
    _extend_kernel()


def _check_tensors(*specs):
    device = specs[0][0].device
    for tensor, dtype in specs:
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError("HCU allocation tensors must share a GPU device")
        if tensor.dtype != dtype or not tensor.is_contiguous():
            raise ValueError(f"HCU allocation requires contiguous {dtype} tensors")


def hcu_alloc_extend_kernel(
    pre_lens_ptr, seq_lens_ptr, last_loc_ptr, free_page_ptr, out_indices, bs, page_size
):
    _check_tensors(*((t, torch.int64) for t in (
        pre_lens_ptr, seq_lens_ptr, last_loc_ptr, free_page_ptr, out_indices
    )))
    if bs == 0:
        return
    with torch.cuda.device(out_indices.device):
        _extend_kernel()(
            pre_lens_ptr, seq_lens_ptr, last_loc_ptr, free_page_ptr,
            out_indices, bs, page_size,
        )


def hcu_alloc_decode_kernel(
    seq_lens_ptr, last_loc_ptr, free_page_ptr, out_indices, bs, page_size
):
    _check_tensors(
        (seq_lens_ptr, torch.int64), (last_loc_ptr, torch.int32),
        (free_page_ptr, torch.int64), (out_indices, torch.int64),
    )
    if bs == 0:
        return
    with torch.cuda.device(out_indices.device):
        _module().hcu_alloc_decode_kernel(
            seq_lens_ptr, last_loc_ptr, free_page_ptr, out_indices, bs, page_size
        )


def hcu_assign_req_to_token_pool(
    req_pool_indices, req_to_token, allocate_lens, new_allocate_lens,
    out_cache_loc, shape, bs,
):
    _check_tensors(
        (req_pool_indices, torch.int64), (req_to_token, torch.int32),
        (allocate_lens, torch.int32), (new_allocate_lens, torch.int32),
        (out_cache_loc, torch.int64),
    )
    if bs == 0:
        return
    with torch.cuda.device(req_to_token.device):
        _module().hcu_assign_req_to_token_pool(
            req_pool_indices, req_to_token, allocate_lens, new_allocate_lens,
            out_cache_loc, shape, bs,
        )


def hcu_get_last_loc(req_to_token, req_pool_indices, prefix_lens):
    _check_tensors(
        (req_to_token, torch.int32), (req_pool_indices, torch.int64),
        (prefix_lens, torch.int32),
    )
    if prefix_lens.numel() == 0:
        return torch.empty_like(req_pool_indices)
    with torch.cuda.device(req_to_token.device):
        return _module().hcu_get_last_loc(req_to_token, req_pool_indices, prefix_lens)
