from __future__ import annotations

import torch

from sglang.srt.layers.quantization.base_config import QuantizationConfig


class SlimQuantW4A8Int8Config(QuantizationConfig):
    """Config class for W8A8 Int8 Quantization.

    - Weight: static, per-channel, symmetric
    - Activation: dynamic, per-token, symmetric
    """

    def __init__(self, config=None):
        super().__init__()
        config = {} if config is None else dict(config)
        if config.get("weight_bits", 4) != 4:
            raise ValueError("SlimQuant W4A8 requires weight_bits=4")
        if config.get("group_size", -1) != -1 or config.get("sym", True) is not True:
            raise ValueError("SlimQuant W4A8 requires symmetric per-channel weights")
        self.quant_description = config
        self.ignore = list(config.get("ignore") or [])
        self.packed_modules_mapping = dict(config.get("packed_modules_mapping") or {})
        self.moe_int8_layers = set(config.get("moe_int8_layers") or [])
        self.moe_int8_prefixes = [
            f"model.layers.{index}.mlp.experts" for index in sorted(self.moe_int8_layers)
        ]
        self.w4_dir = config.get("w4_dir", False)
        if not isinstance(self.w4_dir, bool):
            raise ValueError("SlimQuant w4_dir must be a boolean")

    @classmethod
    def get_supported_act_dtypes(cls):
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls):
        return 75

    @classmethod
    def get_name(cls):
        return "slimquant_w4a8"

    @classmethod
    def get_config_filenames(cls):
        return []

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def get_scaled_act_names(self):
        return []

    def apply_weight_name_mapper(self, hf_to_sglang_mapper):
        self.ignore = list(dict.fromkeys(hf_to_sglang_mapper.apply_list(self.ignore)))
        self.moe_int8_prefixes = hf_to_sglang_mapper.apply_list(self.moe_int8_prefixes)

    def get_quant_method(self, layer, prefix):
        from sglang.srt.layers.linear import LinearBase
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.quantization.compressed_tensors.utils import (
            should_ignore_layer,
        )
        from sglang.srt.layers.quantization.unquant import (
            UnquantizedFusedMoEMethod,
            UnquantizedLinearMethod,
        )

        if not isinstance(layer, (LinearBase, FusedMoE)):
            return None
        if should_ignore_layer(prefix, self.ignore, self.packed_modules_mapping):
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod()
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            from hcu_sglang.layers.quantization.slimquant_w4a8 import (
                SlimQuantW4A8Int8LinearMethod,
            )

            return SlimQuantW4A8Int8LinearMethod(self)
        if (
            prefix in self.moe_int8_prefixes
            or getattr(layer, "layer_id", None) in self.moe_int8_layers
        ):
            from sglang.srt.layers.quantization.w8a8_int8 import W8A8Int8MoEMethod

            return W8A8Int8MoEMethod(self)
        return self.get_int4_moe_method()

    def get_int4_moe_method(self):
        from hcu_sglang.layers.quantization.slimquant_w4a8 import (
            SlimQuantW4A8Int8MoEMethod,
        )

        return SlimQuantW4A8Int8MoEMethod(self)

    def convert_weights(self, weights):
        if not self.w4_dir:
            yield from weights
            return
        for name, tensor in weights:
            if name.endswith(".qweight"):
                if tensor.dtype != torch.uint8 or tensor.ndim != 2:
                    raise ValueError(f"Expected uint8 [N, K/2] SlimQuant qweight: {name}")
                packed = ((tensor & 15) << 4) | (tensor >> 4)
                yield name.removesuffix(".qweight") + ".weight", packed.view(torch.int8)
            elif name.endswith(".scales"):
                if (
                    tensor.ndim != 2
                    or tensor.shape[1] != 1
                    or not tensor.is_floating_point()
                ):
                    raise ValueError(f"Expected floating [N, 1] SlimQuant scales: {name}")
                yield name.removesuffix(".scales") + ".weight_scale", tensor.float() / 16
            else:
                yield name, tensor
