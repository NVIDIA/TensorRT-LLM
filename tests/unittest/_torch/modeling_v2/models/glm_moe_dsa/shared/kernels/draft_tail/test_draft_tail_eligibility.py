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
"""draft_tail dispatch: native fallback is chosen before any device work.

Uses the real ``LMHead`` and ``DeepseekV3MTPHead`` classes with stand-in
worker / metadata objects; tp_size 1, so no exchange workspace is needed.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

VOCAB = 4096


def _objects(**over):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3MTPHead
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.modules.linear import TensorParallelMode
    from tensorrt_llm.mapping import Mapping

    mapping = over.pop("mapping", Mapping(world_size=1, rank=0, tp_size=1))
    head = LMHead(
        VOCAB,
        6144,
        dtype=over.pop("head_dtype", torch.bfloat16),
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
    )
    head.weight.data = (torch.randn(head.weight.shape, device="cuda") * 0.02).to(head.weight.dtype)
    shared_head = over.pop("shared_head", object.__new__(DeepseekV3MTPHead))
    worker = SimpleNamespace(
        is_mtp_eagle=over.pop("is_mtp_eagle", True),
        guided_decoder=over.pop("guided_decoder", None),
        _d2t=over.pop("d2t", None),
        mapping=mapping,
    )
    spec = SimpleNamespace(
        wants_advanced_draft_sampling=over.pop("advanced", False),
        draft_vocab_size=0,
        vocab_size=VOCAB,
        num_seq_slots=over.pop("num_seq_slots", 8),
        max_num_requests=over.pop("max_num_requests", 8),
        is_cuda_graph=over.pop("is_cuda_graph", False),
    )
    draft = SimpleNamespace(mtp_layers=[SimpleNamespace(shared_head=shared_head)], lm_head=head)
    assert not over, over
    return worker, draft, spec


def _select(worker, draft, spec):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    return dispatch.get_impl("draft_tail", "select_draft_tail")(worker, draft, spec)


def test_eligible_path_matches_native_argmax():
    worker, draft, spec = _objects()
    tail = _select(worker, draft, spec)
    assert tail is not None
    gen = torch.Generator(device="cuda").manual_seed(0)
    for rows in range(1, 9):
        h = torch.randn(rows, 6144, generator=gen, device="cuda").to(torch.bfloat16)
        tokens = tail(h)
        ref = torch.argmax(F.linear(h, draft.lm_head.weight), dim=-1).to(torch.int32)
        assert tokens is not None and tokens.dtype == torch.int32
        torch.testing.assert_close(tokens, ref, rtol=0, atol=0)


def test_per_step_fallback_only_for_rank_shared_properties():
    """A step leaves draft_tail only for properties every TP rank shares: the row
    count (same requests on every rank), dtype and hidden width (same
    model)."""
    worker, draft, spec = _objects()
    tail = _select(worker, draft, spec)
    h = torch.randn(9, 6144, device="cuda").to(torch.bfloat16)
    assert tail(h) is None  # more than 8 rows
    assert tail(h[:0]) is None  # no rows
    assert tail(h[:2].half()) is None  # dtype
    assert tail(h[:2, :4096].contiguous()) is None  # hidden size
    assert tail(h[0]) is None  # not a row block


def _misaligned_layouts(rows, gen):
    """Row blocks whose layout the kernels cannot read directly: non-unit
    last stride, 8-byte-aligned base, row stride not a multiple of 8."""
    src = torch.randn(rows, 6144, generator=gen, device="cuda").to(torch.bfloat16)
    strided = src.t().contiguous().t()
    off = torch.empty(rows * 6144 + 4, dtype=torch.bfloat16, device="cuda")
    shifted = off[4:].view(rows, 6144)
    shifted.copy_(src)
    wide = torch.empty(rows, 6148, dtype=torch.bfloat16, device="cuda")
    pitched = wide[:, :6144]
    pitched.copy_(src)
    layouts = [shifted, pitched]
    if rows > 1:  # one row has no meaningful last-dim transpose
        layouts.append(strided)
    return src, layouts


def test_per_step_layout_is_staged_never_native():
    """Strides and alignment are rank-local (allocator placement): they must
    not move one rank to the native collective while its peers run draft_tail's
    exchange. Such rows are copied into draft_tail's aligned staging rows."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_tail.draft_tail import (  # noqa: E501
        _kernel_layout_ok,
    )

    worker, draft, spec = _objects()
    tail = _select(worker, draft, spec)
    runner = dispatch.get_impl("draft_tail", "get_runner")(worker.mapping)
    gen = torch.Generator(device="cuda").manual_seed(3)
    for rows in (1, 2, 5, 8):
        src, layouts = _misaligned_layouts(rows, gen)
        ref = torch.argmax(F.linear(src, draft.lm_head.weight), dim=-1).to(torch.int32)
        for x in layouts:
            assert not _kernel_layout_ok(x)
            before = runner.staged_steps
            tokens = tail(x)
            assert tokens is not None
            assert runner.staged_steps == before + 1
            torch.testing.assert_close(tokens, ref, rtol=0, atol=0)
        before = runner.staged_steps
        torch.testing.assert_close(tail(src), ref, rtol=0, atol=0)
        assert runner.staged_steps == before  # aligned rows: no copy


def test_per_step_staging_under_cuda_graph():
    """The staging copy is captured with the step and replays with new
    contents of the same (misaligned) source buffer."""
    worker, draft, spec = _objects()
    tail = _select(worker, draft, spec)
    gen = torch.Generator(device="cuda").manual_seed(4)
    off = torch.zeros(3 * 6144 + 4, dtype=torch.bfloat16, device="cuda")
    x = off[4:].view(3, 6144)
    x.copy_(torch.randn(3, 6144, generator=gen, device="cuda"))
    tail(x)  # warm-up outside capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = tail(x)
    for _ in range(5):
        x.copy_(torch.randn(3, 6144, generator=gen, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        ref = torch.argmax(F.linear(x.contiguous(), draft.lm_head.weight), dim=-1).to(torch.int32)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_per_step_device_mismatch_raises():
    worker, draft, spec = _objects()
    tail = _select(worker, draft, spec)
    with pytest.raises(RuntimeError):
        tail(torch.zeros(1, 6144, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "change",
    [
        dict(is_mtp_eagle=False),
        dict(guided_decoder=object()),
        dict(d2t=torch.zeros(4, dtype=torch.int64)),
        dict(advanced=True),
        dict(shared_head=object()),
        dict(head_dtype=torch.float16),
    ],
)
def test_configuration_fallbacks(change):
    worker, draft, spec = _objects(**change)
    assert _select(worker, draft, spec) is None


@pytest.mark.parametrize(
    "change,eligible",
    [
        (dict(num_seq_slots=9), False),
        (dict(num_seq_slots=16, max_num_requests=4, is_cuda_graph=True), False),
        (dict(num_seq_slots=0, max_num_requests=16), False),
        (dict(num_seq_slots=0, max_num_requests=4, is_cuda_graph=True), False),
        (dict(num_seq_slots=0, max_num_requests=4), True),
        (dict(num_seq_slots=2, max_num_requests=1, is_cuda_graph=True), True),
    ],
)
def test_row_bound_decides_per_configuration(change, eligible):
    """draft_tail engages only if no draft step of the configuration can exceed 8
    rows, judged from the engine's sequence-slot pool, which CUDA-graph
    metadata keeps: a step can then never switch between draft_tail and native
    rounding because of its (padded) row count."""
    worker, draft, spec = _objects(**change)
    assert (_select(worker, draft, spec) is not None) == eligible


def test_one_kernel_for_every_row_count():
    """The shipped options use one kernel shape for every row count, in the
    native accumulation order (native draft tokens, near-ties included)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_tail.draft_tail import (  # noqa: E501
        FIRST_MMA_VARIANT,
        NATIVE_ORDER_VARIANTS,
        DraftTailOptions,
    )

    opts = DraftTailOptions()
    assert opts.mode == "fused" and opts.variant >= FIRST_MMA_VARIANT
    assert opts.variant in NATIVE_ORDER_VARIANTS
    assert opts.shape(152) == (opts.variant, 0)


def test_attention_dp_falls_back():
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(world_size=2, rank=0, tp_size=2, enable_attention_dp=True)
    worker, draft, spec = _objects(mapping=mapping)
    assert _select(worker, draft, spec) is None


def test_feature_off_by_default(monkeypatch):
    """Without an installed plan (a process that does not build the target) draft_tail is off."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    monkeypatch.setattr(dispatch, "_INSTALLED", [None])
    assert not dispatch.is_enabled("draft_tail")
