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
"""The sampler's finish-reasons kernel is planned before warmup samples.

``FinishReasonsProvider`` plans ``fused_write_finish_reasons`` from the
sampler's buffers alone; the sampler's real launches, at any request count,
must be among the planned variants.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch import jit_prefetch as jp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _keys(calls):
    from triton import knobs
    from triton.runtime.driver import driver
    from triton.runtime.jit import compute_cache_key

    keys = set()
    for call in calls:
        variants, _ = jp._expand(call)
        jit_fn, _, _ = jp._unwrap_jit(call.fn)
        for kw in variants:
            kw = dict(kw)
            kw["debug"] = kw.get("debug", jit_fn.debug) or knobs.runtime.debug
            kw["instrumentation_mode"] = knobs.compilation.instrumentation_mode
            _, kkc, _, _, binder = jit_fn.device_caches[driver.active.get_current_device()]
            _, spec, opts = binder(*call.args, **kw)
            keys.add((jit_fn.fn.__name__, compute_cache_key(kkc, spec, opts)))
    return keys


@pytest.mark.parametrize("max_tokens", [1, 4])
def test_finish_reasons_provider_covers_the_sampler_kernel(max_tokens):
    from tensorrt_llm._torch.pyexecutor.sampler.finish_reasons import FinishReasonsHandler
    from tensorrt_llm._torch.pyexecutor.sampler.jit_prefetch import FinishReasonsProvider
    from tensorrt_llm._torch.pyexecutor.sampler.ops.triton import fused_write_finish_reasons
    from tensorrt_llm.bindings.executor import FinishReason

    slots = 8
    handler = FinishReasonsHandler(
        max_num_sequences=slots, max_beam_width=1, max_tokens=max_tokens, max_seq_len=64
    )
    # The sampler's new_tokens buffer: one scratch slot row past the real ones.
    new_tokens = torch.zeros(max_tokens, slots + 1, 1, dtype=torch.int, device="cuda")
    sampler = SimpleNamespace(
        _finish_reasons_handler=handler, store=SimpleNamespace(new_tokens=new_tokens)
    )
    provider = FinishReasonsProvider(sampler)
    assert provider
    (batch,) = list(provider.early_batches())
    planned = _keys(provider(batch))
    store = handler.store
    for n in (1, 3, slots):
        for seq_len_dtype in (torch.int32, torch.int64):
            with jp.shadow_launches() as rec:
                fused_write_finish_reasons(
                    finish_reasons=store.finish_reasons_cuda,
                    new_tokens=new_tokens,
                    seq_slots=torch.arange(n, dtype=torch.int64, device="cuda"),
                    seq_lens=torch.full((n,), 5, dtype=seq_len_dtype, device="cuda"),
                    max_lengths=store.max_lengths_cuda,
                    end_ids=store.end_ids_cuda,
                    max_tokens=max_tokens,
                    beam_width=1,
                    not_finished_value=FinishReason.NOT_FINISHED.value,
                    length_value=FinishReason.LENGTH.value,
                    end_id_value=FinishReason.END_ID.value,
                )
            assert _keys(rec) <= planned, (n, seq_len_dtype)
    assert not FinishReasonsProvider(SimpleNamespace())
