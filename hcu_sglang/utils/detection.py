"""HCU hardware detection (plugin-internal)."""

from __future__ import annotations

import logging
from functools import lru_cache

import torch


logger = logging.getLogger(__name__)

_SUPPORTED_ARCHS = ("gfx936", "gfx938", "gfx928")
_FP8_ARCHS = ("gfx938",)


def _detection_arch(*, require_arch=False):
    if not getattr(torch.cuda, "is_initialized", lambda: True)():
        try:
            from hcu_adapter.hcu_env import arch_name
        except ModuleNotFoundError as exc:
            if exc.name not in {"hcu_adapter", "hcu_adapter.hcu_env"}:
                raise
        else:
            return arch_name()
    properties = torch.cuda.get_device_properties(0)
    return properties.gcnArchName if require_arch else getattr(properties, "gcnArchName", "")


@lru_cache(maxsize=1)
def is_hcu() -> bool:
    if getattr(torch.version, "hip", None) is None:
        return False
    try:
        gcn_arch = _detection_arch()
        return any(gfx in gcn_arch for gfx in _SUPPORTED_ARCHS)
    except Exception as e:
        logger.warning("HCU detection failed: %s", e)
        return False


@lru_cache(maxsize=1)
def is_hcu_native_fp8_supported() -> bool:
    if not is_hcu() or not torch.version.hip:
        return False
    gcn_arch = _detection_arch(require_arch=True)
    return any(gfx in gcn_arch for gfx in _FP8_ARCHS)


def get_gcn_arch(device_id: int = 0) -> str:
    try:
        props = torch.cuda.get_device_properties(device_id)
        return getattr(props, "gcnArchName", "")
    except Exception:
        return ""
