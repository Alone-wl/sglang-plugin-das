"""Plugin-owned process options, inherited by spawned SGLang workers."""
import os

MINIMAX_OPT_ENV = "HCU_PLUGIN_MINIMAX_OPT"


def minimax_opt_enabled():
    raw = os.environ.get(MINIMAX_OPT_ENV, "0").lower()
    if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{MINIMAX_OPT_ENV}: invalid boolean {raw!r}")
    return raw in {"1", "true", "yes", "on"}


def configure_minimax_opt(value):
    if value is not None:
        os.environ[MINIMAX_OPT_ENV] = "1" if value else "0"


CP_CACHE_LAYER_SPLIT_ENV = "HCU_PLUGIN_CP_CACHE_LAYER_SPLIT"


def cp_cache_layer_split_enabled():
    """Candidate CP selection, inherited by workers; full service remains WIP."""
    raw = os.environ.get(CP_CACHE_LAYER_SPLIT_ENV, "0").lower()
    if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{CP_CACHE_LAYER_SPLIT_ENV}: invalid boolean {raw!r}")
    return raw in {"1", "true", "yes", "on"}


def configure_cp_cache_layer_split(value):
    if value is not None:
        os.environ[CP_CACHE_LAYER_SPLIT_ENV] = "1" if value else "0"


CUSTOM_ALL_REDUCE_BACKEND_ENV = "HCU_PLUGIN_CUSTOM_ALL_REDUCE_BACKEND"
CUSTOM_ALL_REDUCE_BACKENDS = ("auto", "native", "aiter", "off")


def custom_all_reduce_backend():
    value = os.environ.get(CUSTOM_ALL_REDUCE_BACKEND_ENV, "auto").lower()
    if value not in CUSTOM_ALL_REDUCE_BACKENDS:
        raise ValueError(f"{CUSTOM_ALL_REDUCE_BACKEND_ENV}: invalid backend {value!r}")
    return value


def configure_custom_all_reduce_backend(value):
    if value is not None:
        if value not in CUSTOM_ALL_REDUCE_BACKENDS:
            raise ValueError(f"Invalid custom all-reduce backend: {value!r}")
        os.environ[CUSTOM_ALL_REDUCE_BACKEND_ENV] = value


def resolved_custom_all_reduce_backend(disabled=False):
    from hcu_sglang.utils.detection import is_dcu
    from sglang.srt.utils import get_bool_env_var
    backend = custom_all_reduce_backend()
    if disabled:
        return "off"
    if backend == "auto" and is_dcu() and get_bool_env_var("SGLANG_USE_AITER_AR", default="false"):
        return "aiter"
    return backend
