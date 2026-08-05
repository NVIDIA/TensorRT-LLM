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

"""The nanojet integration must refuse what it cannot serve, and say so early.

The backend keeps no KV cache: every answer comes from the Q/K/V of the current call. Any
configuration that needs history is therefore out of reach, and the failure mode if it were
allowed through is the dangerous one — attention over just the tokens present looks perfectly
plausible and is wrong. These tests pin the refusals.
"""

import pytest
import torch

MODEL = "/tmp/nanojet_quant_modelopt/Qwen3-Embedding-0.6B"


def _batch_info(
    num_prefill: int, num_prefill_tokens: int, num_decode: int = 0, num_extend: int = 0
):
    """The 14-slot host tensor the attention ops read their batch composition from."""
    info = torch.zeros(14, dtype=torch.int32)
    info[0] = num_prefill
    info[1] = num_prefill_tokens
    info[2] = num_extend
    info[3] = num_extend
    info[4] = num_decode
    info[5] = num_decode
    return info


def _call_attention(**overrides):
    """Drive the op with a minimal single-sequence prefill, overriding one thing at a time."""
    from tensorrt_llm._torch.auto_deploy.custom_ops.attention import nanojet_attention

    assert nanojet_attention.register(), "nanojet must be installed for this test"

    tokens, heads, kv_heads, head_dim = 8, 4, 2, 128
    kwargs = dict(
        q=torch.randn(1, tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16),
        k=torch.randn(1, tokens, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16),
        v=torch.randn(1, tokens, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16),
        batch_info_host=_batch_info(1, tokens),
        seq_len=torch.tensor([tokens], dtype=torch.int32, device="cuda"),
        input_pos=torch.zeros(1, dtype=torch.int32, device="cuda"),
        slot_idx=torch.zeros(1, dtype=torch.int32, device="cuda"),
        cu_seqlen=torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        scale=None,
    )
    kwargs.update(overrides)
    return torch.ops.auto_deploy.nanojet_attention(**kwargs)


# --------------------------------------------------------------------------------------
# Config time: rejected before weights are loaded or a graph is built.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "unsupported, needle",
    [
        ({"enable_chunked_prefill": True}, "chunked prefill"),
    ],
)
def test_config_rejects_unsupported_nanojet_attention(unsupported, needle):
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    with pytest.raises(ValueError, match=needle):
        LlmArgs(model=MODEL, attn_backend="nanojet", **unsupported)


def test_config_accepts_plain_prefill():
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    args = LlmArgs(model=MODEL, attn_backend="nanojet")
    assert args.attn_backend == "nanojet"


def test_other_backends_keep_chunked_prefill():
    """The refusal is scoped to this backend and must not restrict the shipped ones."""
    from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs

    assert LlmArgs(model=MODEL, attn_backend="trtllm", enable_chunked_prefill=True)


# --------------------------------------------------------------------------------------
# Import hygiene: nothing about nanojet is touched unless it was asked for.
# --------------------------------------------------------------------------------------


def test_nanojet_not_imported_unless_requested():
    import subprocess
    import sys

    probe = (
        "import sys;"
        "import tensorrt_llm._torch.auto_deploy.custom_ops;"
        "import tensorrt_llm._torch.auto_deploy.transform.library;"
        "from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs;"
        f"LlmArgs(model='{MODEL}');"
        "print([m for m in sys.modules if m.startswith('nanojet_kernels')])"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.stdout.strip().endswith("[]"), f"nanojet was imported: {out.stdout}"


# --------------------------------------------------------------------------------------
# Runtime: data-dependent cases the config cannot see.
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runtime_rejects_decode():
    with pytest.raises(RuntimeError, match="prefill-only"):
        _call_attention(batch_info_host=_batch_info(0, 0, num_decode=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runtime_rejects_continuation_chunk():
    """An extend request carries cached context this backend does not keep."""
    with pytest.raises(RuntimeError, match="continuation chunk"):
        _call_attention(batch_info_host=_batch_info(0, 0, num_extend=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runtime_rejects_shared_kv():
    with pytest.raises(RuntimeError, match="KV cache"):
        _call_attention(read_cache_only=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runtime_rejects_custom_attention_mask():
    mask = torch.ones(1, 1, 8, 8, dtype=torch.bool, device="cuda")
    with pytest.raises(RuntimeError, match="custom_attn_mask"):
        _call_attention(custom_attn_mask=mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_runtime_accepts_plain_prefill():
    """The supported case must still go through, or the guards above prove nothing."""
    out = _call_attention()
    assert out.shape == (1, 8, 4, 128)
    assert not out.isnan().any()
