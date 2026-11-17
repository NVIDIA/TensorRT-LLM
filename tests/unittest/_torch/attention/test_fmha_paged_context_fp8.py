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
"""trtllm::fmha_paged_context_fp8: causal FP8 paged-KV context attention through the trtllm-gen runner."""

import math

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm  # noqa: F401  (loads the trtllm ops)

pytestmark = skip_pre_blackwell

_TOKENS_PER_PAGE = 64
_HEAD_DIM = 256
# The V descale (scale_out / bmm2_scale) of the calls, not 1 so that its application is checked.
_SCALE_OUT = 0.75
# The FP8 kernel's error against the FP32 reference relative to the output norm (P is rounded to E4M3 for P @ V).
_MAX_REL_ERR = 6e-2


def _make_case(q_lens, kv_lens, num_heads_q, num_heads_kv, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    batch = len(q_lens)
    max_pages = max(math.ceil(n / _TOKENS_PER_PAGE) for n in kv_lens)
    # Separate K and V pages per sequence, shuffled through the pool.
    num_pages = 2 * batch * max_pages
    pool = (torch.randn(num_pages, num_heads_kv, _TOKENS_PER_PAGE, _HEAD_DIM, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    perm = torch.randperm(num_pages, device=dev, dtype=torch.int32)
    block_tables = perm.view(batch, 2, max_pages).contiguous()
    q = (torch.randn(sum(q_lens), num_heads_q, _HEAD_DIM, device=dev) * 0.5).to(torch.float8_e4m3fn)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    cu_q = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    cu_q[1:] = torch.tensor(q_lens, device=dev).cumsum(0)
    cu_kv = torch.zeros(batch + 1, dtype=torch.int32, device=dev)
    cu_kv[1:] = seq_lens.cumsum(0)
    softmax_scale = 1.0 / math.sqrt(_HEAD_DIM)
    scale_bmm1 = torch.tensor(
        [softmax_scale, softmax_scale * math.log2(math.e)], dtype=torch.float32, device=dev
    )
    return q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, softmax_scale


def _reference(q, pool, block_tables, q_lens, kv_lens, softmax_scale):
    num_heads_q, num_heads_kv = q.shape[1], pool.shape[1]
    out = torch.empty(q.shape[0], num_heads_q, _HEAD_DIM, dtype=torch.float32, device=q.device)
    q_start = 0
    for b, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens)):
        pages = math.ceil(kv_len / _TOKENS_PER_PAGE)

        def gather(kind):
            idx = block_tables[b, kind, :pages].long()
            x = pool[idx].float()  # [pages, Hkv, P, D]
            return x.permute(1, 0, 2, 3).reshape(num_heads_kv, -1, _HEAD_DIM)[:, :kv_len]

        k = gather(0).repeat_interleave(num_heads_q // num_heads_kv, 0)
        v = gather(1).repeat_interleave(num_heads_q // num_heads_kv, 0)
        qb = q[q_start : q_start + q_len].float().transpose(0, 1)
        scores = qb @ k.transpose(1, 2) * softmax_scale
        pos_q = torch.arange(kv_len - q_len, kv_len, device=q.device)[:, None]
        pos_k = torch.arange(kv_len, device=q.device)[None, :]
        scores = scores.masked_fill(pos_k > pos_q, float("-inf"))
        out[q_start : q_start + q_len] = (scores.softmax(-1) @ v).transpose(0, 1)
        q_start += q_len
    return out


# The K / V stage is always fixed: SM100 / SM103 run the precompiled stage-256 kernels of cubin/kernelMetaInfoStageKv.h,
# every other stage (and arch) is generated with NVRTC (allow_nvrtc). The V descale (scale_out) is not 1.
@pytest.mark.parametrize("head_dim_per_stage_kv", [128, 256])
@pytest.mark.parametrize(
    "q_lens,kv_lens", [([300, 77], [428, 77]), ([1], [130]), ([1024, 64, 513], [1024, 4160, 600])]
)
def test_fmha_paged_context_fp8(head_dim_per_stage_kv, q_lens, kv_lens):
    num_heads_q, num_heads_kv = 16, 1
    q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, softmax_scale = _make_case(
        q_lens, kv_lens, num_heads_q, num_heads_kv
    )
    out = torch.empty(q.shape[0], num_heads_q * _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    assert torch.ops.trtllm.fmha_paged_context_fp8(
        q,
        pool,
        block_tables,
        seq_lens,
        cu_q,
        cu_kv,
        out,
        scale_bmm1,
        torch.full((1,), _SCALE_OUT, dtype=torch.float32, device="cuda"),
        max(q_lens),
        max(kv_lens),
        head_dim_per_stage_kv,
        True,
    )
    ref = _reference(q, pool, block_tables, q_lens, kv_lens, softmax_scale) * _SCALE_OUT
    got = out.view(-1, num_heads_q, _HEAD_DIM).float()
    torch.testing.assert_close(got, ref, atol=3e-2, rtol=5e-2)
    # Relative to the output's norm, which a zero output does not meet at long KV (where atol alone would pass it).
    rel = ((got - ref).norm() / ref.norm()).item()
    assert rel < _MAX_REL_ERR, rel


# Every shape of the FP32-reference test, and one with a 5000-token KV.
@pytest.mark.parametrize(
    "q_lens,kv_lens",
    [
        ([700, 129, 64], [2048, 129, 5000]),
        ([300, 77], [428, 77]),
        ([1], [130]),
        ([1024, 64, 513], [1024, 4160, 600]),
    ],
)
def test_fmha_paged_context_fp8_matches_flashinfer(q_lens, kv_lens):
    """The K / V stage size does not change the output bits, and the output matches FlashInfer's prebuilt
    trtllm-gen context kernel up to one BF16 ulp in a few elements (the JIT and prebuilt kernels differ there)."""
    flashinfer = pytest.importorskip("flashinfer")
    num_heads_q, num_heads_kv = 16, 1
    q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, softmax_scale = _make_case(
        q_lens, kv_lens, num_heads_q, num_heads_kv, seed=1
    )

    def run_op(head_dim_per_stage_kv):
        out = torch.empty(q.shape[0], num_heads_q * _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        assert torch.ops.trtllm.fmha_paged_context_fp8(
            q,
            pool,
            block_tables,
            seq_lens,
            cu_q,
            cu_kv,
            out,
            scale_bmm1,
            torch.full((1,), _SCALE_OUT, dtype=torch.float32, device="cuda"),
            max(q_lens),
            max(kv_lens),
            head_dim_per_stage_kv,
            True,
        )
        return out

    # On SM100 / SM103, stage 256 is the precompiled E4 kernel and stage 128 the NVRTC one.
    out = run_op(_HEAD_DIM)
    assert torch.equal(out.view(torch.int16), run_op(_HEAD_DIM // 2).view(torch.int16))
    expected = torch.empty(q.shape[0], num_heads_q, _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    workspace = torch.zeros(128 << 20, dtype=torch.uint8, device="cuda")
    flashinfer.prefill.trtllm_batch_context_with_kv_cache(
        query=q,
        kv_cache=(pool, pool),
        workspace_buffer=workspace,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_q_len=max(q_lens),
        max_kv_len=max(kv_lens),
        bmm1_scale=scale_bmm1[:1],
        bmm2_scale=_SCALE_OUT,
        batch_size=len(q_lens),
        cum_seq_lens_q=cu_q,
        cum_seq_lens_kv=cu_kv,
        window_left=-1,
        out=expected,
        kv_layout="HND",
        uses_shared_paged_kv_idx=False,
        causal=True,
    )
    expected = expected.view(-1, num_heads_q * _HEAD_DIM)
    torch.testing.assert_close(out, expected, atol=0.0, rtol=2.0**-7)
    assert (out != expected).sum().item() <= out.numel() * 1e-5


@pytest.mark.parametrize("head_dim_per_stage_kv", [128, 256])
def test_fmha_paged_context_fp8_last_rows_match_full_call(head_dim_per_stage_kv):
    """The op's last-row call (one query row per request against its full KV) gives each request's last output row
    bit for bit as the full context call does."""
    q_lens, kv_lens = [700, 1, 129, 2048], [5000, 130, 129, 2048]
    num_heads_q, num_heads_kv = 16, 1
    q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, _ = _make_case(
        q_lens, kv_lens, num_heads_q, num_heads_kv, seed=2
    )

    def run(query, cu, max_q):
        out = torch.empty(
            query.shape[0], num_heads_q * _HEAD_DIM, dtype=torch.bfloat16, device="cuda"
        )
        assert torch.ops.trtllm.fmha_paged_context_fp8(
            query,
            pool,
            block_tables,
            seq_lens,
            cu,
            cu_kv,
            out,
            scale_bmm1,
            torch.ones(1, dtype=torch.float32, device="cuda"),
            max_q,
            max(kv_lens),
            head_dim_per_stage_kv,
            True,
        )
        return out

    last = (cu_q[1:] - 1).long()
    batch = len(q_lens)
    full = run(q, cu_q, max(q_lens))
    last_rows = run(
        q.index_select(0, last), torch.arange(batch + 1, dtype=torch.int32, device="cuda"), 1
    )
    assert torch.equal(last_rows.view(torch.int16), full[last].view(torch.int16))


def _native_context_forward(q_lens, last_rows_only, head_dim_per_stage_kv):
    """One context-only forward of a TRTLLM attention layer (FP8 KV cache, paged context FMHA, BF16 output) over
    fresh requests of ``q_lens`` tokens: (output [tokens, heads * head_dim], the layer's K / V cache)."""
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
    from tensorrt_llm._torch.attention.backends.utils import create_attention
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    num_heads, num_seqs = 16, len(q_lens)
    attn = create_attention(
        "TRTLLM",
        layer_idx=0,
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        num_kv_heads=1,
        quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8),
        context_last_rows_only=last_rows_only,
        context_fmha_head_dim_per_stage_kv=head_dim_per_stage_kv,
    )
    request_ids = list(range(num_seqs))
    pages = (max(q_lens) + _TOKENS_PER_PAGE - 1) // _TOKENS_PER_PAGE
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    manager = KVCacheManagerV2(
        KvCacheConfig(max_tokens=num_seqs * pages * _TOKENS_PER_PAGE),
        CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_PAGE,
        max_seq_len=pages * _TOKENS_PER_PAGE,
        max_batch_size=num_seqs,
        mapping=mapping,
        dtype=DataType.FP8,
    )
    try:
        manager.add_dummy_requests(request_ids, list(q_lens))
        metadata = TrtllmAttentionMetadata(
            num_contexts=num_seqs,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0] * num_seqs),
            seq_lens=torch.tensor(q_lens, dtype=torch.int),
            max_num_requests=num_seqs,
            max_num_tokens=sum(q_lens),
            kv_cache_manager=manager,
            request_ids=request_ids,
            prompt_lens=list(q_lens),
            mapping=mapping,
            runtime_features=AttentionRuntimeFeatures(chunked_prefill=True),
        )
        metadata.prepare()
        assert metadata.use_paged_context_fmha
        g = torch.Generator(device="cuda").manual_seed(5)
        qkv = torch.randn(sum(q_lens), (num_heads + 2) * _HEAD_DIM, generator=g, device="cuda").to(
            torch.bfloat16
        )
        with torch.no_grad():
            out, _ = attn.forward(qkv, None, None, metadata)
        torch.cuda.synchronize()
        cache = manager.get_buffers(0, kv_layout="HND").clone()
    finally:
        manager.shutdown()
    return out.view(sum(q_lens), -1), cache


@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="stage-256 cubins exist for SM100 / SM103",
)
@pytest.mark.parametrize("head_dim_per_stage_kv", [0, _HEAD_DIM], ids=["unstaged", "stage256"])
def test_native_context_last_rows_only(head_dim_per_stage_kv):
    """The native context attention with context_last_rows_only: in a context-only step each request's last output
    row matches the full call's, the other rows are zero, and every row's K / V is still written."""
    q_lens = [700, 1, 129, 64]
    full, full_cache = _native_context_forward(q_lens, False, head_dim_per_stage_kv)
    last_only, last_cache = _native_context_forward(q_lens, True, head_dim_per_stage_kv)
    last = torch.tensor(q_lens, device="cuda").cumsum(0) - 1
    others = torch.ones(sum(q_lens), dtype=torch.bool, device="cuda")
    others[last] = False
    assert torch.equal(last_cache.view(torch.uint8), full_cache.view(torch.uint8))
    assert bool((last_only[others] == 0).all())
    torch.testing.assert_close(last_only[last], full[last], atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="stage-256 cubins exist for SM100 / SM103",
)
def test_fmha_paged_context_fp8_stage256_runs_precompiled_kernel():
    """A stage-256 call runs the precompiled E4 kernel instead of generating one."""
    q_lens, kv_lens = [300, 77], [428, 77]
    num_heads_q, num_heads_kv = 16, 1
    q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, _ = _make_case(
        q_lens, kv_lens, num_heads_q, num_heads_kv
    )
    out = torch.empty(q.shape[0], num_heads_q * _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        assert torch.ops.trtllm.fmha_paged_context_fp8(
            q,
            pool,
            block_tables,
            seq_lens,
            cu_q,
            cu_kv,
            out,
            scale_bmm1,
            torch.ones(1, dtype=torch.float32, device="cuda"),
            max(q_lens),
            max(kv_lens),
            _HEAD_DIM,
        )
        torch.cuda.synchronize()
    names = {e.key for e in prof.key_averages()}
    assert any(n.endswith("ContextStageKv256E4") for n in names), names


@pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="stage-256 cubins exist for SM100 / SM103",
)
@pytest.mark.parametrize(
    "tokens_per_page,head_dim_per_stage_kv,ran",
    [(_TOKENS_PER_PAGE, _HEAD_DIM, True), (32, _HEAD_DIM, False), (_TOKENS_PER_PAGE, 128, False)],
)
def test_fmha_paged_context_fp8_declines_without_a_precompiled_kernel(
    tokens_per_page, head_dim_per_stage_kv, ran
):
    """Without allow_nvrtc the op runs only a precompiled kernel: for a call whose selected kernel has none (another
    page size or stage) it returns False and writes nothing, and the caller runs another kernel."""
    q_lens, kv_lens = [300, 77], [428, 77]
    num_heads_q, num_heads_kv = 16, 1
    q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, _ = _make_case(
        q_lens, kv_lens, num_heads_q, num_heads_kv
    )
    # The same KV bytes in pages of tokens_per_page tokens.
    pages_per_page = _TOKENS_PER_PAGE // tokens_per_page
    pool = pool.view(pool.shape[0], num_heads_kv, pages_per_page, tokens_per_page, _HEAD_DIM)
    pool = pool.permute(0, 2, 1, 3, 4).reshape(-1, num_heads_kv, tokens_per_page, _HEAD_DIM)
    block_tables = (
        block_tables[..., None] * pages_per_page
        + torch.arange(pages_per_page, dtype=torch.int32, device="cuda")
    ).flatten(2)
    out = torch.full(
        (q.shape[0], num_heads_q * _HEAD_DIM), 7.0, dtype=torch.bfloat16, device="cuda"
    )
    assert (
        torch.ops.trtllm.fmha_paged_context_fp8(
            q,
            pool.contiguous(),
            block_tables.contiguous(),
            seq_lens,
            cu_q,
            cu_kv,
            out,
            scale_bmm1,
            torch.ones(1, dtype=torch.float32, device="cuda"),
            max(q_lens),
            max(kv_lens),
            head_dim_per_stage_kv,
        )
        is ran
    )
    torch.cuda.synchronize()
    assert bool((out == 7.0).all()) is not ran


def test_fmha_paged_context_fp8_needs_a_fixed_stage():
    q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, _ = _make_case([1], [130], 16, 1)
    out = torch.empty(1, 16 * _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="head_dim_per_stage_kv must be > 0"):
        torch.ops.trtllm.fmha_paged_context_fp8(
            q,
            pool,
            block_tables,
            seq_lens,
            cu_q,
            cu_kv,
            out,
            scale_bmm1,
            torch.ones(1, dtype=torch.float32, device="cuda"),
            1,
            130,
            0,
        )


def _stage_kv_generator():
    """scripts/trtllm_gen_fmha_stage_kv/gen_stage_kv_cubins.py as a module (stdlib imports only at module level)."""
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[4]
        / "scripts/trtllm_gen_fmha_stage_kv/gen_stage_kv_cubins.py"
    )
    if not path.is_file():
        pytest.skip("needs the repository sources")
    spec = importlib.util.spec_from_file_location("gen_stage_kv_cubins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_kv_cubins_match_the_kernel_inputs_they_were_built_against():
    """The stage-KV cubins are compiled against the trtllm-gen kernel headers and launched with parameters the
    runner sets up from KernelParams / KernelTraits / FmhaOptions: a change to any of them needs a regeneration
    (scripts/trtllm_gen_fmha_stage_kv/gen_stage_kv_cubins.py), which records their hash in every header block."""
    import re

    gen = _stage_kv_generator()
    names = {p.relative_to(gen.EXPORT_DIR).as_posix() for p in gen.kernel_input_files()}
    assert {"KernelParams.h", "KernelParamsDecl.h", "KernelTraits.h", "FmhaOptions.h"} <= names
    assert any(name.startswith("trtllm/dev/") for name in names)
    recorded = re.findall(r"// Kernel inputs sha256: ([0-9a-f]{64})", gen.META_HEADER.read_text())
    assert recorded and set(recorded) == {gen.kernel_inputs_sha256()}


@pytest.mark.parametrize("touched", ["KernelTraits.h", "FmhaOptions.h", "trtllm/dev/Utils.h"])
def test_stage_kv_kernel_inputs_hash_covers_the_parameter_setup(tmp_path, touched):
    """Editing KernelTraits.h, FmhaOptions.h or a device header changes the recorded hash (not only KernelParams)."""
    import shutil

    gen = _stage_kv_generator()
    export = tmp_path / "export"
    for path in gen.kernel_input_files():
        dst = export / path.relative_to(gen.EXPORT_DIR)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dst)
    before = gen.kernel_inputs_sha256(export)
    assert before == gen.kernel_inputs_sha256()
    with open(export / touched, "a") as f:
        f.write("\n")
    assert gen.kernel_inputs_sha256(export) != before


def test_stage_kv_regeneration_keeps_other_page_sizes_and_drops_stale_archives(
    tmp_path, monkeypatch
):
    """A regeneration for one page size replaces only that configuration's header blocks of its SM, and deletes
    the SM's archives no entry references any more (CMake links every archive in cubin/)."""
    gen = _stage_kv_generator()
    cubin = tmp_path / "cubin"
    cubin.mkdir()
    monkeypatch.setattr(gen, "CUBIN_DIR", cubin)
    monkeypatch.setattr(gen, "META_HEADER", cubin / "kernelMetaInfoStageKv.h")
    monkeypatch.setattr(gen, "kernel_inputs_sha256", lambda: "0" * 64)

    def kernel(sm, page, sched):
        stem = (
            f"FmhaSm{sm}aKernel_QkvE4m3OBfloat16H256PagedKvCausalP{page}{sched}ContextStageKv256E4"
        )
        (cubin / f"{stem}.cubin.tar.zst").write_bytes(b"x")
        decl = f"extern unsigned char const {stem}_cubin[];\nextern unsigned int const {stem}_cubin_len;\n"
        return stem, decl, f"{{ 256, {{ /* {stem} */ }} }},\n"

    p64 = gen.config_key(256, 64, 256)
    p32 = gen.config_key(256, 32, 256)
    persistent64, d1, e1 = kernel(103, 64, "Persistent")
    static64, d2, e2 = kernel(103, 64, "Static")
    gen.write_header(103, p64, d1 + d2, e1 + e2)
    other_sm, d3, e3 = kernel(100, 64, "Static")
    gen.write_header(100, p64, d3, e3)
    static32, d4, e4 = kernel(103, 32, "Static")
    keep = gen.write_header(103, p32, d4, e4)
    assert keep == {persistent64, static64, static32}
    assert gen.remove_stale_archives(103, keep) == []
    text = gen.META_HEADER.read_text()
    for stem in (persistent64, static64, static32, other_sm):
        assert f"{stem}_cubin[]" in text and f"/* {stem} */" in text

    # Regenerate page size 64 on SM103: only the persistent kernel is selected now.
    keep = gen.write_header(103, p64, d1, e1)
    assert keep == {persistent64, static32}
    assert gen.remove_stale_archives(103, keep) == [cubin / f"{static64}.cubin.tar.zst"]
    text = gen.META_HEADER.read_text()
    assert static64 not in text and static32 in text and other_sm in text
    assert sorted(p.name for p in cubin.glob("*.tar.zst")) == sorted(
        f"{s}.cubin.tar.zst" for s in (persistent64, static32, other_sm)
    )


def test_fmha_paged_context_fp8_rejects_a_pool_without_kv_heads():
    """The Q / KV head ratio check runs after the KV-head count check (no integer division by zero)."""
    q, pool, block_tables, seq_lens, cu_q, cu_kv, scale_bmm1, _ = _make_case([1], [130], 16, 1)
    empty_pool = torch.empty(
        (pool.shape[0], 0, pool.shape[2], pool.shape[3]), dtype=pool.dtype, device="cuda"
    )
    out = torch.empty(1, 16 * _HEAD_DIM, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="kv_pool must have KV heads"):
        torch.ops.trtllm.fmha_paged_context_fp8(
            q,
            empty_pool,
            block_tables,
            seq_lens,
            cu_q,
            cu_kv,
            out,
            scale_bmm1,
            torch.ones(1, dtype=torch.float32, device="cuda"),
            1,
            130,
            _HEAD_DIM,
        )


def test_fmha_paged_context_fp8_fake_declines():
    """Traced, the op answers False (whether a cubin exists is known only at run time): a trace keeps the FlashInfer
    kernel the caller runs on False, which is always valid."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty((8, 4, 256), dtype=torch.float8_e4m3fn, device="cuda")
        kv_pool = torch.empty((4, 2, 1, 32, 256), dtype=torch.float8_e4m3fn, device="cuda")
        block_tables = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
        seq_lens = torch.full((1,), 8, dtype=torch.int32, device="cuda")
        cu = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
        out = torch.empty((8, 4, 256), dtype=torch.bfloat16, device="cuda")
        scale = torch.ones(1, dtype=torch.float32, device="cuda")
        took = torch.ops.trtllm.fmha_paged_context_fp8(
            q, kv_pool, block_tables, seq_lens, cu, cu, out, scale, scale, 8, 8, 256
        )
    assert took is False
