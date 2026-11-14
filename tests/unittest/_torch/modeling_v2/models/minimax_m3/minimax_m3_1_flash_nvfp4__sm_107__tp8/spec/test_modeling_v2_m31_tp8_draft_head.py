# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP8_DRAFT_HEAD_MXFP8 on the sm_107 TP8 target: the drafter head's Markov-chain shard GEMM on this target's MXFP8
copy (dspark_draft_head.DraftHeadMx4), taken by the chain's draft-logits processor ahead of FOLD_F5B, written into the
padded buffer with its -inf tail kept."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
    dspark_markov_worker as s4,
)

SHARD, HIDDEN, K = 25008, 6144, 7


def test_default_on_with_kill_switch() -> None:
    src = inspect.getsource(s4)
    assert 'DHEAD8 = FOLD_F5B and not m31_disabled("TP8_DRAFT_HEAD_MXFP8")' in src
    assert s4.DHEAD8_BATCHES == (1,)


def test_set_up_after_prob1_before_state_published() -> None:
    src = inspect.getsource(s4)
    i = src.index("self._prebuild_prob1(")
    j = src.index("self._prebuild_dhead8(lm_head, state, block)")
    k = src.index("self._s4 = state")
    assert i < j < k


def _sm107() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)


cuda = pytest.mark.skipif(not _sm107(), reason="the sm_107 TP8 target runs on SM107 only")


def _lm_head(seed=11):
    from tensorrt_llm._torch.modules.linear import UnquantizedLinearMethod

    g = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(SHARD, HIDDEN, device="cuda", generator=g) * 0.02).bfloat16()
    return SimpleNamespace(
        weight=weight,
        bias=None,
        quant_method=UnquantizedLinearMethod(),
        _locality_domain_weight_shards=None,
        use_cute_dsl_bf16_gemm=False,
        use_custom_cublas_mm=False,
    )


def _state(rows, head4):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
        dspark_markov_chain as chain,
    )

    buf = chain.pad_logits_buffer(1, K, rows, torch.device("cuda"))
    st = SimpleNamespace(block=K, shard=SHARD, rows=rows, head4=head4, bufs={(1, K): buf})
    st.buffer = lambda batch: st.bufs[(batch, K)]
    return st


@cuda
def test_processor_takes_mxfp8_copy_first() -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8 import (  # noqa: E501
        dspark_draft_head as dh,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
        dspark_markov_chain as chain,
    )

    rows = chain.padded_rows(SHARD, s4.MAX_FAST_BATCH, torch.device("cuda"))
    if rows == 0:
        pytest.skip("the chain's kernel does not split this shard on this device")
    head = _lm_head()
    head4 = dh.DraftHeadMx4(head, (K,), rows, rank=0)
    st = _state(rows, head4)
    proc = s4._ShardedDraftLogits(lambda *a, **k: pytest.fail("inner called"), st, 1)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = (torch.randn(K, HIDDEN, device="cuda", generator=g)).bfloat16()
    got = proc(x, head, None)
    buf = st.buffer(1)
    assert got.data_ptr() == buf.data_ptr() and tuple(got.shape) == (K, SHARD)
    ref = torch.empty(K, rows, dtype=torch.bfloat16, device="cuda").fill_(float("-inf"))
    head4.run_into(x, ref)
    assert torch.equal(
        buf.view(K, rows)[:, :SHARD].view(torch.int16), ref[:, :SHARD].view(torch.int16)
    )
    assert torch.isneginf(buf.view(K, rows)[:, SHARD:].float()).all()
    # Against the BF16 head the MXFP8 quantization error counts too: the oracle's bound, not the self-check's
    # (which compares with the dequantized operands).
    bf16 = x.float() @ head.weight.float().t()
    rel = float((got.float() - bf16).abs().max() / bf16.abs().max())
    assert rel < 2.0**-4, rel
    # A foreign head is declined: FOLD_F5B / the BF16 head serves the call.
    other = _lm_head(seed=12)
    got2 = proc(x, other, None)
    want = torch.nn.functional.linear(x, other.weight)
    assert torch.equal(got2.view(torch.int16), want.view(torch.int16))
