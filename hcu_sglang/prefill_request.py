# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations
from sglang.srt.managers import schedule_batch as N
from sglang.srt.mem_cache.storage_prefetch import StagedPrefetchPlan as _LegacyStagedPrefetchPlan
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from hcu_sglang.utils.detection import is_dcu


_LEGACY_PLAN = [_LegacyStagedPrefetchPlan]


class Req(N.Req):
    @property
    def auxiliary_swa_host_hit_length(self) -> int:
        plan = self.staged_prefetch_plan
        if isinstance(plan, _LEGACY_PLAN[0]):
            return 0
        return (
            plan.window_tokens(ComponentType.AUXILIARY_SWA) if plan is not None else 0
        )

    def needs_host_load_back(self) -> bool:
        """Whether any cache layer has a host hit that needs L2 H2D load_back."""
        return (
            self.host_hit_length > 0
            or self.swa_host_hit_length > 0
            or self.mamba_host_hit_length > 0
            or self.auxiliary_swa_host_hit_length > 0
        )


def register_hooks():
    if not is_dcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.schedule_batch.Req", Req, HookType.REPLACE)
