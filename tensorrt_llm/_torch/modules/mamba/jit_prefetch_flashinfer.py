# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""FlashInfer JIT modules of the Mamba2 decode, built ahead.

A ``Mamba2Mixer`` whose shape FlashInfer supports (``_use_flashinfer``) runs
a single-token generation step's state update through FlashInfer's
``selective_state_update``. FlashInfer builds that op's JIT module, one per
argument signature, at its first call, in warmup's first generation forward.

``flashinfer_modules`` names the module of that call, which follows from the
layer alone, for ``jit_prefetch_engine.start_model_builds`` to build in a CPU-only
helper while the weights load. The name is the module's JitSpec name, which
spells out every argument of its generator, so the helper rebuilds the spec
from it (``jit_prefetch_flashinfer_helper``).
"""

from __future__ import annotations

from typing import List, Optional

import torch

# The prefix of FlashInfer's selective_state_update JitSpec names.
SSU_PREFIX = "selective_state_update_"


def _sm_suffix(major: int) -> str:
    """The JitSpec name suffix of the module FlashInfer builds for a device of
    compute capability ``major`` (``flashinfer.mamba.selective_state_update``
    picks the generator by it)."""
    return "_sm100" if major >= 10 else "_sm90" if major >= 9 else ""


def decode_module_name(mixer, sm_major: int) -> Optional[str]:
    """The JitSpec name of the module ``mixer``'s single-token decode step
    builds: its ``selective_state_update`` call passes the SSM cache in the
    cache dtype, ``x`` in the activation dtype, ``dt`` converted to float32,
    the float32 ``A`` expanded at load, the int32 state indices of
    ``Mamba2Metadata``, and a Philox seed when the step rounds stochastically.
    None if the cache dtype is not a concrete dtype yet."""
    from flashinfer.jit.mamba.selective_state_update import get_selective_state_update_uri

    state_dtype = mixer._mamba_ssm_cache_dtype
    input_dtype = mixer.norm.weight.dtype
    if not isinstance(state_dtype, torch.dtype) or input_dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        return None
    philox_rounds = mixer._philox_rounds if mixer._stochastic_rounding_for_flashinfer else 0
    uri = get_selective_state_update_uri(
        state_dtype,
        input_dtype,
        torch.float32,
        torch.float32,
        torch.int32,
        None,
        mixer.head_dim,
        mixer.d_state,
        1,
        torch.int32,
        torch.int64,
        philox_rounds,
    )
    return uri + _sm_suffix(sm_major)


def flashinfer_modules(model) -> List[str]:
    """The FlashInfer ``selective_state_update`` modules ``model``'s Mamba2
    layers build in their first single-token decode step. None for a model
    with speculative decoding: its generation steps verify drafts, through
    another module or kernel chosen at run time."""
    from .mamba2_mixer import Mamba2Mixer

    model_config = getattr(model, "model_config", None)
    if getattr(model_config, "spec_config", None) is not None:
        return []
    mixers = [m for m in model.modules() if isinstance(m, Mamba2Mixer) and m._use_flashinfer]
    if not mixers:
        return []
    major = torch.cuda.get_device_capability()[0]
    names = (decode_module_name(m, major) for m in mixers)
    return list(dict.fromkeys(n for n in names if n))
