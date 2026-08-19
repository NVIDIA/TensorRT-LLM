"""Unit tests for the packed joint-QKV projection op and its eligibility
census (``tensorrt_llm/_torch/visual_gen/utils.py``).

The op is the concat-elimination fast lane used by Qwen-Image joint
attention: two ``addmm`` calls writing both per-stream merged-QKV
projections straight into row slices of one packed buffer, replacing the
per-stream projections + seq-dim ``torch.cat``.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear

# Importing utils registers the trtllm_vgoa::packed_qkv_proj custom op.
from tensorrt_llm._torch.visual_gen.utils import linear_supports_packed_addmm

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# Small stand-ins for the production shape (txt [1,1024,3072] +
# img [1,6889,3072] -> [1,7913,9216]); the packed dim stays a multiple of
# 128 like the real q_dim + 2*kv_dim.
HIDDEN = 128
PACKED = 3 * HIDDEN
S_TXT = 17
S_IMG = 64


def _make_inputs(device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    txt = torch.randn(1, S_TXT, HIDDEN, device=device, dtype=dtype)
    img = torch.randn(1, S_IMG, HIDDEN, device=device, dtype=dtype)
    w_txt = torch.randn(PACKED, HIDDEN, device=device, dtype=dtype)
    b_txt = torch.randn(PACKED, device=device, dtype=dtype)
    w_img = torch.randn(PACKED, HIDDEN, device=device, dtype=dtype)
    b_img = torch.randn(PACKED, device=device, dtype=dtype)
    return txt, img, w_txt, b_txt, w_img, b_img


@requires_cuda
def test_packed_qkv_proj_matches_merged_linear_plus_cat():
    """The op output must be bit-identical to the fallback path it replaces
    (per-stream merged projection + one seq-dim cat): same GEMM problems on
    the same inputs, only the output placement differs."""
    txt, img, w_txt, b_txt, w_img, b_img = _make_inputs()
    out = torch.ops.trtllm_vgoa.packed_qkv_proj(txt, img, w_txt, b_txt, w_img, b_img)
    ref = torch.cat([F.linear(txt, w_txt, b_txt), F.linear(img, w_img, b_img)], dim=1)
    assert out.shape == (1, S_TXT + S_IMG, PACKED)
    assert out.dtype == txt.dtype
    assert torch.equal(out, ref)


@requires_cuda
def test_packed_qkv_proj_opcheck():
    """torch.library.opcheck validates the schema, fake impl, and
    functional (alias-free) contract the compiled path relies on."""
    args = _make_inputs()
    torch.library.opcheck(torch.ops.trtllm_vgoa.packed_qkv_proj, args)


@requires_cuda
def test_linear_supports_packed_addmm_census():
    eligible = Linear(HIDDEN, PACKED, bias=True, dtype=torch.bfloat16).cuda()
    assert linear_supports_packed_addmm(eligible, out_features=PACKED, in_features=HIDDEN)
    # Shape mismatch must be rejected.
    assert not linear_supports_packed_addmm(
        eligible, out_features=PACKED + HIDDEN, in_features=HIDDEN
    )
    # The op signature requires a bias tensor.
    no_bias = Linear(HIDDEN, PACKED, bias=False, dtype=torch.bfloat16).cuda()
    assert not linear_supports_packed_addmm(no_bias, out_features=PACKED, in_features=HIDDEN)
    # Only bf16 weights take the raw-addmm lane.
    fp16 = Linear(HIDDEN, PACKED, bias=True, dtype=torch.float16).cuda()
    assert not linear_supports_packed_addmm(fp16, out_features=PACKED, in_features=HIDDEN)


def test_linear_supports_packed_addmm_rejects_cpu_weights():
    cpu_linear = Linear(HIDDEN, PACKED, bias=True, dtype=torch.bfloat16)
    assert not linear_supports_packed_addmm(cpu_linear, out_features=PACKED, in_features=HIDDEN)
