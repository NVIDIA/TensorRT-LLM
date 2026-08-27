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
"""How much KV a request must have reserved before its FIRST verify step.

The context phase reserves ``prompt + num_extra_kv_tokens``. From the second
step on, the scheduler grows a generation request by ``1 + draft`` per step and
the margin is comfortable, so this is the one place the reservation has to be
right.

The generic one-engine reserve is ``max_draft_len - 1``
(``get_num_extra_kv_tokens``), which is two short of the ``1 + max_draft_len``
positions an Inkling verify step writes. Being short is invisible unless the
last drafted position lands on a page boundary -- which is why the end-to-end
runs passed for weeks and one 5-shot GSM8K prompt did not (job 6026096:
prompt 669, capacity 671, writing 669..672, page_size 32).
"""

import pytest

from tensorrt_llm._torch.speculative.utils import get_num_extra_kv_tokens


def _pages(last_pos: int, page_size: int) -> int:
    """Pages needed to hold positions 0..last_pos, as write_kv_cache_hnd indexes."""
    return last_pos // page_size + 1


def _reserved_pages(prompt: int, extra: int, page_size: int) -> int:
    """Pages the context phase's reservation of ``prompt + extra`` materialises.

    Blocks follow capacity in KVCacheManagerV2 (``div_up(capacity, page)``), so
    the page count is derivable from the token count alone.
    """
    return -(-(prompt + extra) // page_size)


@pytest.mark.parametrize("max_draft_len", [1, 2, 3, 4, 7])
def test_a_verify_step_writes_one_more_than_the_generic_reserve(max_draft_len):
    """The gap this fix closes, stated as the arithmetic that produced it.

    A verify step presents ``1 + max_draft_len`` tokens and writes every one of
    them, so the first one needs that many positions past the prompt. The
    generic reserve stops ``2`` short of it.
    """

    class _SpecConfig:
        pass

    spec_config = _SpecConfig()
    spec_config.max_draft_len = max_draft_len
    spec_config.spec_dec_mode = type("_Mode", (), {"use_one_engine": staticmethod(lambda: True)})()

    generic = get_num_extra_kv_tokens(spec_config)
    needed = 1 + max_draft_len
    assert generic == max_draft_len - 1
    assert needed - generic == 2


@pytest.mark.parametrize("page_size", [16, 32, 128])
def test_the_generic_reserve_is_short_exactly_at_a_page_boundary(page_size):
    """Why this was not caught earlier: it needs the prompt to line up.

    Over a sweep of prompt lengths the generic reserve is enough for most and
    short for those where the last drafted position opens a new page. The fixed
    reserve is enough for every one of them -- that difference, not the average,
    is the regression.
    """
    max_draft_len = 3
    generic = max_draft_len - 1
    fixed = 1 + max_draft_len
    short_for = []
    for prompt in range(page_size * 4, page_size * 8):
        last_pos = prompt + max_draft_len  # positions prompt .. prompt + draft
        need = _pages(last_pos, page_size)
        if need > _reserved_pages(prompt, generic, page_size):
            # Never more than one page short: the reservation misses by two
            # tokens, so it can only ever miss the page those two open.
            assert need == _reserved_pages(prompt, generic, page_size) + 1
            short_for.append(prompt)
        assert need <= _reserved_pages(prompt, fixed, page_size)
    # It is an alignment, not a size: the same two residues in every page cycle
    # (job 6026096's prompt of 669 is 669 % 32 == 29, the first of them).
    assert short_for, "the sweep must contain the failing alignment"
    assert {p % page_size for p in short_for} == {page_size - 3, page_size - 2}


def test_the_manager_raises_the_reservation_over_the_generic_one(monkeypatch):
    """``InklingHybridCacheManager`` takes the larger of the two.

    Asserted through the constructor rather than by re-deriving the number,
    because the bug was that the generic value reached the context phase
    unchanged.

    ``num_extra_kv_tokens`` is set inside the base ``__init__`` from the spec
    config, so it can only be raised afterwards -- and ``max_blocks_per_seq``
    was already derived from the old value. Re-deriving it is asserted here
    too: without that, the per-sequence block bound is short by exactly the
    tokens just reserved, and only at page alignments (the padding to a
    multiple of four blocks hides it the rest of the time), which is the same
    intermittent shape as the bug this reservation exists to fix.
    """
    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as cm

    captured = {}

    def _fake_super_init(self, *args, **kwargs):
        # The real base sets these from the constructor args; the current
        # manager reads them (self.max_draft_len for the reservation, self.mapping
        # / self.max_batch_size for the conv-pool geometry properties), so the
        # mock has to provide them too.
        _spec = kwargs.get("spec_config")
        self.max_draft_len = _spec.max_draft_len if _spec is not None else 0
        self.mapping = kwargs.get("mapping")
        self.max_batch_size = kwargs.get("max_batch_size")
        self.num_extra_kv_tokens = get_num_extra_kv_tokens(kwargs.get("spec_config"))
        captured["generic"] = self.num_extra_kv_tokens
        # What the real base derives from it, and what the override re-derives.
        self.max_seq_len = 1024
        self.tokens_per_block = 32
        self._kv_reserve_draft_tokens = 3
        self.max_blocks_per_seq = 0  # deliberately wrong; the override must fix it
        self.num_local_layers = 66

    class _FakeConvCache:
        num_slots = 5

        # _InklingConvGeometry asks the pool class for its reserved-row count
        # rather than re-deriving it, so a stand-in has to answer too.
        @staticmethod
        def reserved_slot_count(*, reserve_attention_dp_slot):
            return 1 + int(reserve_attention_dp_slot)

        def __init__(self, *args, **kwargs):
            captured["max_draft_len"] = kwargs.get("max_draft_len")
            captured["num_layers"] = kwargs.get("num_layers")
            captured["layer_offset"] = kwargs.get("layer_offset")

        def conv_state_bytes(self):
            return 0

    monkeypatch.setattr(cm.KVCacheManagerV2, "__init__", _fake_super_init)
    monkeypatch.setattr(cm, "InklingConvStateCache", _FakeConvCache)

    class _Mapping:
        enable_attention_dp = False
        tp_size = 1
        pp_size = 1

    class _Text:
        num_hidden_layers = 66
        torch_dtype = "bfloat16"
        sconv_kernel_size = 4
        hidden_size = 4096

        @staticmethod
        def layer_num_kv_heads(_i):
            return 8

        @staticmethod
        def layer_head_dim(_i):
            return 128

    class _Pretrained:
        text_config = _Text()

    class _SpecConfig:
        max_draft_len = 3
        spec_dec_mode = type("_Mode", (), {"use_one_engine": staticmethod(lambda: True)})()

    mgr = cm.InklingHybridCacheManager(
        pretrained_config=_Pretrained(),
        mapping=_Mapping(),
        max_batch_size=4,
        spec_config=_SpecConfig(),
    )

    assert captured["generic"] == 2  # max_draft_len - 1, the generic reserve
    assert mgr.num_extra_kv_tokens == 4  # 1 + max_draft_len, what a step writes
    # The pool derives its capture depth from the same number rather than being
    # told separately, so max_draft_len is what has to reach it.
    assert captured["max_draft_len"] == 3
    # Re-derived with the base's own formula: ceil(1024 + 4 + 3 + 1 / 32) = 33,
    # rounded up to a multiple of four blocks.
    assert mgr.max_blocks_per_seq == 36


def test_a_target_manager_sizes_its_conv_pool_from_the_whole_trunk(monkeypatch):
    """Only a DRAFT manager narrows the pool to the chain's layers.

    The draft manager covers the chain alone and addresses it by global layer
    index; the target's covers the trunk and starts at zero. Passing the draft
    narrowing on the target would allocate a pool of the wrong depth and index
    past it -- so the two are pinned apart here.
    """
    from tensorrt_llm._torch.attention_backend.sparse.inkling import cache_manager as cm

    captured = {}

    def _fake_super_init(self, *args, **kwargs):
        _spec = kwargs.get("spec_config")
        self.max_draft_len = _spec.max_draft_len if _spec is not None else 0
        self.mapping = kwargs.get("mapping")
        self.max_batch_size = kwargs.get("max_batch_size")
        self.num_extra_kv_tokens = 0
        self.max_seq_len = 1024
        self.tokens_per_block = 32
        self._kv_reserve_draft_tokens = 0
        self.max_blocks_per_seq = 0
        self.num_local_layers = 66

    class _FakeConvCache:
        num_slots = 5

        # _InklingConvGeometry asks the pool class for its reserved-row count
        # rather than re-deriving it, so a stand-in has to answer too.
        @staticmethod
        def reserved_slot_count(*, reserve_attention_dp_slot):
            return 1 + int(reserve_attention_dp_slot)

        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def conv_state_bytes(self):
            return 0

    monkeypatch.setattr(cm.KVCacheManagerV2, "__init__", _fake_super_init)
    monkeypatch.setattr(cm, "InklingConvStateCache", _FakeConvCache)

    class _Mapping:
        enable_attention_dp = False
        tp_size = 1
        pp_size = 1

    class _Text:
        num_hidden_layers = 66
        torch_dtype = "bfloat16"
        sconv_kernel_size = 4
        hidden_size = 4096

        @staticmethod
        def layer_num_kv_heads(_i):
            return 8

        @staticmethod
        def layer_head_dim(_i):
            return 128

    class _Pretrained:
        text_config = _Text()

    cm.InklingHybridCacheManager(
        pretrained_config=_Pretrained(), mapping=_Mapping(), max_batch_size=4
    )
    assert captured["num_layers"] is None  # the whole trunk
    assert captured["layer_offset"] == 0  # addressed from zero

    captured.clear()
    cm.InklingHybridCacheManager(
        pretrained_config=_Pretrained(),
        mapping=_Mapping(),
        max_batch_size=4,
        is_draft=True,
        num_layers=3,
    )
    assert captured["num_layers"] == 3  # the chain's depths only
    assert captured["layer_offset"] == 66  # addressed past the trunk


def test_the_media_towers_do_not_abort_meta_init(monkeypatch):
    """Meta init has to survive the vision and audio towers.

    ``Module(...).to(bfloat16)`` lowers to ``aten._to_copy``, which a meta
    tensor refuses; TRT-LLM catches that, gives up on meta init for the WHOLE
    model and materialises every parameter instead. On the full BF16 checkpoint
    that is ~950B parameters built the slow way before a weight is read -- 30
    minutes at 100% CPU with idle GPUs, twice mistaken for a hang.

    The mechanism that avoids it is deferral: under meta init the tower is not
    built at all, and its config is handed back for ``load_weights`` to rebuild
    from. So the assertable property is that a tower which cannot be converted
    yields a deferral rather than propagating the exception.
    """
    from tensorrt_llm._torch.models import modeling_inkling as mi

    class _Cfg:
        decoder_dmodel = 64

    class _RaisesOnConstruct:
        def __init__(self, _config):
            raise mi.MetaInitException("meta init refuses _to_copy")

    cfg = _Cfg()
    tower, deferred = mi._build_replicated_bf16_tower(_RaisesOnConstruct, cfg)
    assert tower is None, "a tower that cannot be built must not be returned"
    assert deferred is cfg, "the config must come back so load_weights can rebuild"

    # A tower left holding meta parameters is deferred too: constructing can
    # succeed and only the conversion afterwards would raise.
    import torch

    class _LeavesMetaParams:
        def __init__(self, _config):
            with torch.device("meta"):
                self._p = torch.nn.Linear(4, 4)

        def parameters(self):
            return self._p.parameters()

    tower, deferred = mi._build_replicated_bf16_tower(_LeavesMetaParams, cfg)
    assert tower is None
    assert deferred is cfg

    # No tower configured at all is not a deferral -- there is nothing to build.
    assert mi._build_replicated_bf16_tower(_RaisesOnConstruct, None) == (None, None)
