# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram short-convolution unit tests for Qwen4-Exp models.

The batched production module
:class:`Qwen4ExpPLE` is validated on CUDA against an **independent** per-sequence
implementation of the same math, for a
prefill step **and** two subsequent decode steps that reuse the carried conv
state + n-gram-context history.

``_RefPLE`` independently re-derives the prime head-vocab sizes with
its own trial-division ``nextprime``, hashes each token's n-gram window with
plain Python integer arithmetic (not batched tensor ops), and streams the short
conv + n-gram history per sequence with explicit slicing — structurally distinct
from the module's batched context grids / ``unfold`` / scatter / batched
``conv1d``. Agreement is therefore a real cross-check of the module's batching,
indexing, and prefill->decode state carry-over, not a tautology. The reference
reads the module's *weights* so both compute the identical function.

Reduced scale: only ``ngram_vocab_size_base`` is shrunk (2003 vs 20_000_000) so
the n-gram embedding table fits in unit-test memory; every other field
(``hidden_size=2560``, ``hc_count=4``, ``ngram_size=3``, ``heads_per_ngram=8``,
``ple_conv_kernel_size=4``, real ``vocab_size``/``eos_token_id``/``seed``) is the
checkpoint value, so the hashing multipliers, grouped-norm gate, and dilated
causal short-convolution math are exercised exactly as in production.
"""

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.qwen4_exp.ple import (
    PLEMetadata,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLE,
    _splitmix64,
    _uses_scaled_fp8_ngram_table,
)

# Parity is measured against an independent per-token reference whose GEMMs /
# conv have a different reduction shape than the batched module. TF32 (default-on
# for fp32 matmul/conv on Ampere+/Blackwell) rounds those two shapes differently,
# injecting ~1e-3 noise that is invisible in bf16 (below its ulp) but swamps the
# tight fp32 tolerance. Force true IEEE fp32 so the fp32 check is a real
# high-precision cross-check; the module itself is dtype-agnostic to this flag.


@pytest.fixture(scope="module", autouse=True)
def _disable_tf32_for_reference_parity():
    """Keep the independent FP32 reference precise without leaking global state."""
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
    torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
    torch.set_float32_matmul_precision(old_matmul_precision)


# Checkpoint-scale dimensions except for the reduced n-gram table base.
HIDDEN = 2560
HC_COUNT = 4
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
PLE_EMBED_DIM = 2560
PLE_CONV_KERNEL = 4
VOCAB_SIZE = 248320
EOS_TOKEN_ID = 248044
SEED = 1234
NGRAM_VOCAB_BASE = 2003  # reduced (real: 20_000_000) so the table is tiny
MAKE_DIVISIBLE_BY = 128
RMS_EPS = 1e-6

NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM  # 16
HEAD_DIM_PER_NGRAM = PLE_EMBED_DIM // NGRAM_HEADS  # 160
CONV_CHANNELS = HIDDEN * HC_COUNT  # 10240
SHORT_CONV_STATE_LEN = (PLE_CONV_KERNEL - 1) * NGRAM_SIZE  # 9
NGRAM_CONTEXT_LEN = NGRAM_SIZE - 1  # 2


def _make_config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        hc_count=HC_COUNT,
        ngram_size=NGRAM_SIZE,
        heads_per_ngram=HEADS_PER_NGRAM,
        ple_embed_dim=PLE_EMBED_DIM,
        ple_conv_kernel_size=PLE_CONV_KERNEL,
        vocab_size=VOCAB_SIZE,
        eos_token_id=EOS_TOKEN_ID,
        seed=SEED,
        ngram_vocab_size_base=NGRAM_VOCAB_BASE,
        make_ngram_vocab_size_divisible_by=MAKE_DIVISIBLE_BY,
        rms_norm_eps=RMS_EPS,
    )


def test_empty_ple_metadata_has_no_request_mapping() -> None:
    metadata = PLEMetadata.build(
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        torch.empty(0, dtype=torch.long),
        is_decode=False,
        eos_token_id=EOS_TOKEN_ID,
    )

    assert metadata.processed_tokens == 0
    assert metadata.req_indices.numel() == 0
    assert metadata.padded_tokens.shape == (0, 0)


def test_scaled_fp8_ngram_storage_preserves_scale_and_output_dtype() -> None:
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size=32,
        eos_token_id=2,
        seed=1234,
        ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=8,
    )
    embedding = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=2,
        dtype=torch.bfloat16,
    )
    scale = torch.tensor(0.5, dtype=torch.float32)

    embedding.configure_fp8_weight_storage(scale, torch.float8_e4m3fn)
    stored = torch.tensor([[1.0, -2.0]], dtype=torch.float8_e4m3fn)
    actual = embedding._dequantize_embeddings(stored)

    assert embedding.ngram_embedding.weight.dtype == torch.float8_e4m3fn
    assert embedding.ngram_embedding_weight_scale.item() == 0.5
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16))


def test_scaled_fp8_ngram_config_accepts_a_single_exclusion_string() -> None:
    excluded = SimpleNamespace(
        quantization_config={
            "quant_method": "fp8",
            "modules_to_not_convert": "model.ple.ple_embedding.ngram_embedding",
        }
    )
    included = SimpleNamespace(
        quantization_config={
            "quant_method": "fp8",
            "modules_to_not_convert": "model.layers.0.self_attn",
        }
    )

    assert not _uses_scaled_fp8_ngram_table(excluded)
    assert _uses_scaled_fp8_ngram_table(included)


def test_scaled_fp8_ngram_lookup_requires_checkpoint_scale() -> None:
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size=32,
        eos_token_id=2,
        seed=1234,
        ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=8,
        quantization_config={"quant_method": "fp8"},
    )
    embedding = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=2,
        dtype=torch.bfloat16,
    )

    with pytest.raises(RuntimeError, match="missing its weight scale"):
        embedding.embed(torch.zeros((1, 2), dtype=torch.long))


def test_ngram_embedding_accepts_hf_eos_token_list() -> None:
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        vocab_size=32,
        eos_token_id=[2, 3],
        seed=1234,
        ngram_vocab_size_base=11,
        make_ngram_vocab_size_divisible_by=8,
    )

    embedding = Qwen4ExpNGramEmbedding(config, embedding_dim=2, dtype=torch.float32)

    assert embedding.eos_token_id == 2


# ---------------------------------------------------------------------------
# Independent reference (per-sequence, plain-Python hashing, streamed state).
# ---------------------------------------------------------------------------
def _ref_is_prime(n: int) -> bool:
    if n < 2:
        return False
    d = 2
    while d * d <= n:
        if n % d == 0:
            return False
        d += 1
    return True


def _ref_nextprime(p: int) -> int:
    c = p + 1
    while not _ref_is_prime(c):
        c += 1
    return c


def _ref_head_primes(base: int, ngram_heads: int, ple_layer_index: int):
    primes, offsets, total = [], [], 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        p = base - 1
        for _ in range(global_head_idx + 1):
            p = _ref_nextprime(p)
        primes.append(p)
        offsets.append(total)
        total += p
    return primes, offsets


def _ref_multipliers(seed: int, ngram_size: int, vocab_size: int, ple_layer_index: int):
    max_long = (1 << 63) - 1
    m_max = max_long // max(vocab_size, 1)
    half_bound = max(1, m_max // 2)
    base_seed = (seed + 10007 * ple_layer_index) & ((1 << 64) - 1)
    gamma = 0x9E3779B97F4A7C15
    mults = []
    for idx in range(ngram_size):
        x0 = (base_seed + gamma * (idx + 1)) & ((1 << 64) - 1)
        mults.append(2 * (_splitmix64(x0) % half_bound) + 1)
    return mults


class _RefPLE:
    """Per-sequence transcription of the sglang PLE math, sharing module weights."""

    def __init__(self, module: Qwen4ExpPLE, dtype: torch.dtype, device: torch.device):
        self.m = module
        self.dtype = dtype
        self.device = device
        self.hidden = HIDDEN
        self.hc = HC_COUNT
        self.ngram_size = NGRAM_SIZE
        self.heads_per_ngram = HEADS_PER_NGRAM
        self.ngram_heads = NGRAM_HEADS
        self.head_dim = HEAD_DIM_PER_NGRAM
        self.conv_channels = CONV_CHANNELS
        self.state_len = SHORT_CONV_STATE_LEN
        self.dilation = NGRAM_SIZE
        self.kernel = PLE_CONV_KERNEL
        self.eos = EOS_TOKEN_ID
        # Independently derived hashing constants.
        self.primes, self.offsets = _ref_head_primes(NGRAM_VOCAB_BASE, self.ngram_heads, 0)
        self.mults = _ref_multipliers(SEED, self.ngram_size, VOCAB_SIZE, 0)
        # Weights read from the module (both compute the same function).
        emb = module.ple_embedding.ngram_embedding
        self.emb_w = emb.weight.detach()
        self.key_w = module.key_proj.weight.detach()
        self.value_w = module.value_proj.weight.detach()
        self.conv_w = module.conv1d.weight.detach()  # [C, 1, kernel]
        self.norm_key_w = module.norm_key.weight.detach().float()
        self.norm_query_w = module.norm_query.weight.detach().float()
        self.norm_conv_w = module.norm_conv.weight.detach().float()
        # Per-sequence recurrent state, seeded fresh.
        self.hist = None  # list[list[int]] of length ngram_size-1 per seq
        self.conv_state = None  # list[Tensor[C, state_len]]

    def reset(self, num_seq: int):
        self.hist = [[self.eos] * (self.ngram_size - 1) for _ in range(num_seq)]
        # Conv state round-trips through the module dtype pool (bf16 in prod),
        # so keep the reference state in dtype to match the carried rounding.
        self.conv_state = [
            torch.zeros(self.conv_channels, self.state_len, device=self.device, dtype=self.dtype)
            for _ in range(num_seq)
        ]

    def _shifted_last(self, window, k: int) -> int:
        """shifted_right_ignore_eos(window, k) evaluated at the last position."""
        L = len(window)
        src = L - 1 - k
        if src < 0:
            return self.eos
        # last eos strictly before the final position -> segment boundary
        last_eos = -1
        for j in range(L - 1):
            if window[j] == self.eos:
                last_eos = j
        pos_in_segment = (L - 1) - (last_eos + 1)
        if pos_in_segment < k:
            return self.eos
        return window[src]

    def _hash_window(self, window):
        ids = []
        for n in range(2, self.ngram_size + 1):
            mix = 0
            for k in range(n):
                tok = self._shifted_last(window, k)
                mix ^= tok * self.mults[k]
            for h in range(self.heads_per_ngram):
                head = (n - 2) * self.heads_per_ngram + h
                ids.append(mix % self.primes[head] + self.offsets[head])
        return ids  # length ngram_heads

    def _grouped_norm(self, x_flat: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x_flat: [hc*hidden]; per-stream RMSNorm (weight + 1) in fp32, then cast
        # back to the module dtype (matches GroupedRMSNorm exactly).
        xf = x_flat.float().reshape(self.hc, self.hidden)
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        xn = (xf * torch.rsqrt(var + RMS_EPS)).reshape(-1)
        return (xn * (weight + 1.0)).to(self.dtype)

    def forward(self, hidden_states: torch.Tensor, input_ids_per_seq, seq_offsets):
        """hidden_states packed [T, hc*hidden]; input_ids_per_seq: list[list[int]]."""
        outputs = []
        for s, toks in enumerate(input_ids_per_seq):
            L = len(toks)
            if L == 0:
                continue
            full = self.hist[s] + list(toks)
            normed_cols = []  # gated_value_normed per token [C] (dtype)
            gated_flat_list = []  # gated_value (unnormed) per token [C] (dtype)
            for o in range(L):
                window = full[o : o + self.ngram_size]
                ids = self._hash_window(window)
                emb = torch.cat([self.emb_w[i] for i in ids]).to(self.dtype)  # [ple_embed_dim]
                key = F.linear(emb, self.key_w)  # [C]
                value = F.linear(emb, self.value_w)  # [hidden]
                tok_global = seq_offsets[s] + o
                query = hidden_states[tok_global]  # [C]
                key_n = self._grouped_norm(key, self.norm_key_w)  # dtype
                query_n = self._grouped_norm(query, self.norm_query_w)  # dtype
                kn = key_n.reshape(self.hc, self.hidden)
                qn = query_n.reshape(self.hc, self.hidden)
                gate = (kn * qn).sum(dim=-1) / math.sqrt(self.hidden)  # dtype[hc]
                gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
                gate = torch.sigmoid(gate)  # dtype [hc]
                val = value.reshape(1, self.hidden)  # dtype
                gated = (gate.reshape(self.hc, 1) * val).reshape(-1)  # dtype [C]
                gv_norm = self._grouped_norm(gated, self.norm_conv_w)  # dtype
                normed_cols.append(gv_norm)
                gated_flat_list.append(gated)
            # Streamed dilated causal short conv in module dtype. Per-seq F.conv1d
            # over [state | this-chunk]; the chunk is NOT padded to a batch
            # row_width, which is exactly equivalent for the causal reach + the
            # ``lengths``-based state window (independent of the batched module).
            chunk = torch.stack(normed_cols, dim=1)  # [C, L] dtype
            conv_input = torch.cat([self.conv_state[s], chunk], dim=1)  # [C,S+L]
            conv_out = F.conv1d(
                conv_input.unsqueeze(0),
                self.conv_w.to(self.dtype),
                bias=None,
                dilation=self.dilation,
                groups=self.conv_channels,
            ).squeeze(0)  # [C, L]
            # advance conv state = conv_input[:, L : L+state_len]
            self.conv_state[s] = conv_input[:, L : L + self.state_len].clone()
            # advance ngram history = last (ngram_size-1) tokens of full
            self.hist[s] = full[L : L + (self.ngram_size - 1)]
            for o in range(L):
                out = gated_flat_list[o] + F.silu(conv_out[:, o])
                outputs.append(out)
        return torch.stack(outputs, dim=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _init_module_weights(module: Qwen4ExpPLE, gen: torch.Generator, device, dtype):
    """Give the module non-degenerate weights so gate + conv actually matter."""
    with torch.no_grad():
        emb = module.ple_embedding.ngram_embedding
        emb.weight.copy_(
            torch.randn(emb.weight.shape, generator=gen, device=device).to(emb.weight.dtype) * 0.05
        )
        module.key_proj.weight.copy_(
            torch.randn(module.key_proj.weight.shape, generator=gen, device=device).to(
                module.key_proj.weight.dtype
            )
            * 0.03
        )
        module.value_proj.weight.copy_(
            torch.randn(module.value_proj.weight.shape, generator=gen, device=device).to(
                module.value_proj.weight.dtype
            )
            * 0.03
        )
        module.conv1d.weight.copy_(
            torch.randn(module.conv1d.weight.shape, generator=gen, device=device).to(
                module.conv1d.weight.dtype
            )
            * 0.5
        )
        # Gemma norms center on 0 (effective weight ~ 1); add mild variation.
        for norm in (module.norm_key, module.norm_query, module.norm_conv):
            norm.weight.copy_(
                torch.randn(norm.weight.shape, generator=gen, device=device).to(norm.weight.dtype)
                * 0.1
            )


def _packed_ids(seq_token_lists):
    flat = [t for seq in seq_token_lists for t in seq]
    offsets, cur = [], 0
    for seq in seq_token_lists:
        offsets.append(cur)
        cur += len(seq)
    return flat, offsets


def _metrics(a: torch.Tensor, b: torch.Tensor):
    a = a.float()
    b = b.float()
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    cos = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    return max_abs, mean_abs, cos


def _run_parity(dtype: torch.dtype, tol_max: float):
    device = torch.device("cuda")
    torch.manual_seed(0)
    gen = torch.Generator(device=device).manual_seed(1234)
    cfg = _make_config()
    module = Qwen4ExpPLE(cfg, dtype=dtype, ple_layer_index=0, layer_id=1).to(device)
    module.eval()
    _init_module_weights(module, gen, device, dtype)

    # Sanity: the module's derived cache-pool contract matches config_utils.
    assert module.conv_state_shape == (CONV_CHANNELS, SHORT_CONV_STATE_LEN)
    assert module.ngram_context_len == NGRAM_CONTEXT_LEN

    num_slots = 4
    state_idx = torch.tensor([0, 1], device=device, dtype=torch.long)

    # Two caller-owned recurrent-state pools (mamba-style, updated in place).
    conv_state = torch.zeros(
        num_slots, CONV_CHANNELS, SHORT_CONV_STATE_LEN, device=device, dtype=dtype
    )
    ngram_context = torch.full(
        (num_slots, NGRAM_CONTEXT_LEN), EOS_TOKEN_ID, device=device, dtype=torch.long
    )

    ref = _RefPLE(module, dtype, device)
    ref.reset(num_seq=2)

    # ----- Prefill: 2 sequences, lengths [5, 3] -----
    seqs = [[11, 42, 7, 900, 5], [312, 8, 64]]
    flat, offsets = _packed_ids(seqs)
    input_ids = torch.tensor(flat, device=device, dtype=torch.long)
    T = len(flat)
    hs = torch.randn(T, CONV_CHANNELS, generator=gen, device=device).to(dtype)
    seq_lens = torch.tensor([len(s) for s in seqs], device=device, dtype=torch.long)
    meta = PLEMetadata.build(
        input_ids, seq_lens, state_idx, is_decode=False, eos_token_id=EOS_TOKEN_ID
    )
    with torch.no_grad():
        out = module.forward(hs, meta, conv_state, ngram_context)
    ref_out = ref.forward(hs, seqs, offsets)
    ma, _, _ = _metrics(out, ref_out)
    assert ma <= tol_max, f"prefill max_abs {ma} > {tol_max}"

    # ----- Decode step 1: one new token per sequence -----
    dec1 = [[99], [1234]]
    flat1, off1 = _packed_ids(dec1)
    ids1 = torch.tensor(flat1, device=device, dtype=torch.long)
    hs1 = torch.randn(len(flat1), CONV_CHANNELS, generator=gen, device=device).to(dtype)
    meta1 = PLEMetadata.build(
        ids1,
        torch.ones(2, device=device, dtype=torch.long),
        state_idx,
        is_decode=True,
        eos_token_id=EOS_TOKEN_ID,
    )
    with torch.no_grad():
        out1 = module.forward(hs1, meta1, conv_state, ngram_context)
    ref_out1 = ref.forward(hs1, dec1, off1)
    ma1, _, _ = _metrics(out1, ref_out1)
    assert ma1 <= tol_max, f"decode max_abs {ma1} > {tol_max}"

    # ----- Decode step 2: reuse carried state again -----
    dec2 = [[7], [88]]
    flat2, off2 = _packed_ids(dec2)
    ids2 = torch.tensor(flat2, device=device, dtype=torch.long)
    hs2 = torch.randn(len(flat2), CONV_CHANNELS, generator=gen, device=device).to(dtype)
    meta2 = PLEMetadata.build(
        ids2,
        torch.ones(2, device=device, dtype=torch.long),
        state_idx,
        is_decode=True,
        eos_token_id=EOS_TOKEN_ID,
    )
    with torch.no_grad():
        out2 = module.forward(hs2, meta2, conv_state, ngram_context)
    ref_out2 = ref.forward(hs2, dec2, off2)
    ma2, _, _ = _metrics(out2, ref_out2)
    assert ma2 <= tol_max, f"decode2 max_abs {ma2} > {tol_max}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_metadata_accepts_existing_host_lengths() -> None:
    device = torch.device("cuda")
    metadata = PLEMetadata.build(
        torch.arange(5, device=device),
        torch.tensor([2, 3], dtype=torch.int32, device=device),
        torch.arange(2, device=device),
        is_decode=False,
        eos_token_id=EOS_TOKEN_ID,
        num_contexts=1,
        host_seq_lens=[2, 3],
    )

    assert metadata.row_width == 3
    assert metadata.context_tokens == 2
    torch.testing.assert_close(metadata.req_indices, torch.tensor([0, 0, 1, 1, 1], device=device))


def test_ple_metadata_rejects_inconsistent_packed_layout() -> None:
    with pytest.raises(ValueError, match="do not match the packed token count"):
        PLEMetadata.build(
            torch.arange(5),
            torch.tensor([2, 2]),
            torch.arange(2),
            is_decode=False,
            eos_token_id=EOS_TOKEN_ID,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_parity_fp32():
    _run_parity(torch.float32, tol_max=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_parity_bf16():
    _run_parity(torch.bfloat16, tol_max=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_ple_state_carryover_matters():
    """A decode from primed recurrent state must differ from fresh state.

    Runs the same decode token through the module twice — once continuing the
    carried conv state + n-gram history from a prefill, once from zeroed/eos
    state. If either recurrent pool were silently ignored, the two outputs would
    coincide; a large delta proves the state is actually consumed.
    """
    device = torch.device("cuda")
    dtype = torch.float32
    gen = torch.Generator(device=device).manual_seed(7)
    cfg = _make_config()
    module = Qwen4ExpPLE(cfg, dtype=dtype, ple_layer_index=0, layer_id=1).to(device)
    module.eval()
    _init_module_weights(module, gen, device, dtype)

    state_idx = torch.tensor([0], device=device, dtype=torch.long)
    conv_state = torch.zeros(1, CONV_CHANNELS, SHORT_CONV_STATE_LEN, device=device, dtype=dtype)
    ngram_context = torch.full(
        (1, NGRAM_CONTEXT_LEN), EOS_TOKEN_ID, device=device, dtype=torch.long
    )

    # Prime the state with a prefill.
    seqs = [[11, 42, 7, 900, 5]]
    flat, _ = _packed_ids(seqs)
    ids = torch.tensor(flat, device=device, dtype=torch.long)
    hs = torch.randn(len(flat), CONV_CHANNELS, generator=gen, device=device).to(dtype)
    meta = PLEMetadata.build(
        ids,
        torch.tensor([len(flat)], device=device),
        state_idx,
        is_decode=False,
        eos_token_id=EOS_TOKEN_ID,
    )
    with torch.no_grad():
        module.forward(hs, meta, conv_state, ngram_context)

    # Same decode token, primed vs fresh state.
    dtok = torch.tensor([99], device=device, dtype=torch.long)
    dhs = torch.randn(1, CONV_CHANNELS, generator=gen, device=device).to(dtype)
    dmeta = PLEMetadata.build(
        dtok,
        torch.ones(1, device=device, dtype=torch.long),
        state_idx,
        is_decode=True,
        eos_token_id=EOS_TOKEN_ID,
    )
    primed_conv = conv_state.clone()
    primed_ctx = ngram_context.clone()
    fresh_conv = torch.zeros_like(conv_state)
    fresh_ctx = torch.full_like(ngram_context, EOS_TOKEN_ID)
    with torch.no_grad():
        out_primed = module.forward(dhs, dmeta, primed_conv, primed_ctx)
        out_fresh = module.forward(dhs, dmeta, fresh_conv, fresh_ctx)
    diff = (out_primed - out_fresh).abs().max().item()
    assert diff > 1e-2, (
        "carried conv/ngram state had no effect — prefill->decode carry-over is broken"
    )


def test_ple_speculative_commit_selects_accepted_prefix_state() -> None:
    module = object.__new__(Qwen4ExpPLE)
    torch.nn.Module.__init__(module)
    conv_pool = torch.zeros(3, 1)
    context_pool = torch.zeros(3, 1, dtype=torch.long)
    slots = torch.tensor([1, 2])
    conv_candidates = torch.tensor([[[10.0], [11.0], [12.0]], [[20.0], [21.0], [22.0]]])
    context_candidates = torch.tensor([[[100], [101], [102]], [[200], [201], [202]]])
    module._pending_conv_states = (conv_pool, slots, conv_pool[slots].clone(), conv_candidates)
    module._pending_ngram_contexts = (
        context_pool,
        slots,
        context_pool[slots].clone(),
        context_candidates,
    )

    # Context count occupies the first entry. Generation request 0 accepts only
    # its golden token (candidate 0); request 1 accepts golden + two drafts.
    module.commit_speculative_states(
        num_accepted_tokens=torch.tensor([1, 1, 3]),
        state_indices=torch.tensor([0, 1, 2]),
        num_contexts=1,
    )

    torch.testing.assert_close(conv_pool[slots], torch.tensor([[10.0], [22.0]]))
    torch.testing.assert_close(context_pool[slots], torch.tensor([[100], [202]]))
    assert module._pending_conv_states is None
    assert module._pending_ngram_contexts is None


def test_ple_mixed_batch_bounds_short_conv_workspace(monkeypatch) -> None:
    """IFB must not pad decode rows to the longest context chunk."""
    channels = 8
    state_len = 9
    module = object.__new__(Qwen4ExpPLE)
    torch.nn.Module.__init__(module)
    module.conv_channels = channels
    module.short_conv_state_len = state_len
    module.short_conv_dilation = 3
    module.conv1d = torch.nn.Conv1d(
        channels,
        channels,
        kernel_size=4,
        groups=channels,
        dilation=module.short_conv_dilation,
        bias=False,
    )
    module._pending_conv_states = None

    # One five-token context followed by two one-token generation requests.
    lengths = torch.tensor([5, 1, 1], dtype=torch.long)
    state_indices = torch.tensor([0, 1, 2], dtype=torch.long)
    input_ids = torch.arange(7, dtype=torch.long)
    metadata = PLEMetadata.build(
        input_ids,
        lengths,
        state_indices,
        is_decode=False,
        eos_token_id=EOS_TOKEN_ID,
        num_contexts=1,
    )
    assert metadata.context_tokens == 5
    values = torch.randn(7, channels)
    initial_state = torch.randn(3, channels, state_len)

    # The original joint-width implementation remains an exact parity oracle
    # when num_contexts is cleared, which disables the split optimization.
    expected_state = initial_state.clone()
    expected = module._short_conv(
        values,
        dataclasses.replace(metadata, num_contexts=0, context_tokens=0),
        expected_state,
    )

    input_shapes = []
    original_conv1d = F.conv1d

    def record_conv1d(input_tensor, *args, **kwargs):
        input_shapes.append(tuple(input_tensor.shape))
        return original_conv1d(input_tensor, *args, **kwargs)

    monkeypatch.setattr(F, "conv1d", record_conv1d)
    actual_state = initial_state.clone()
    actual = module._short_conv(values, metadata, actual_state)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_state, expected_state)
    assert input_shapes == [
        (1, channels, state_len + 5),
        (2, channels, state_len + 1),
    ]


def test_ple_attention_dp_row_shard_preserves_local_token_order(monkeypatch) -> None:
    from tensorrt_llm._torch.modules.qwen4_exp import ple as qwen4_exp_ple

    config = _make_config()
    mapping = SimpleNamespace(tp_size=2, tp_rank=0, cp_size=1, enable_attention_dp=True)
    module = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=32,
        dtype=torch.float32,
        mapping=mapping,
    )
    full_weight = torch.arange(
        module.padded_vocab_size * module.head_dim_per_ngram,
        dtype=torch.float32,
    ).reshape(module.padded_vocab_size, module.head_dim_per_ngram)
    with torch.no_grad():
        module.ngram_embedding.weight.copy_(
            full_weight[module.vocab_start_index : module.vocab_end_index]
        )

    local_ids = torch.tensor(
        [
            [1, module.vocab_end_index + 1] * (NGRAM_HEADS // 2),
            [module.vocab_end_index - 1, module.padded_vocab_size - 1] * (NGRAM_HEADS // 2),
        ],
        dtype=torch.long,
    )
    remote_ids = torch.tensor(
        [[module.vocab_end_index + 3, 7] * (NGRAM_HEADS // 2)],
        dtype=torch.long,
    )
    gathered_ids = torch.cat((local_ids, torch.zeros_like(local_ids[:1]), remote_ids))

    def fake_allgather(input_ids, actual_mapping, dim, sizes):
        assert actual_mapping is mapping
        assert dim == 0
        assert sizes == [3, 1]
        torch.testing.assert_close(input_ids[:2], local_ids)
        torch.testing.assert_close(input_ids[2], torch.zeros_like(input_ids[2]))
        return gathered_ids

    def fake_reducescatter(partial, actual_mapping, dim, sizes):
        assert actual_mapping is mapping
        assert dim == 0
        assert sizes == [3, 1]
        owned = gathered_ids < module.vocab_end_index
        expected_partial = torch.zeros_like(partial)
        expected_partial[owned] = full_weight[gathered_ids[owned]]
        torch.testing.assert_close(partial, expected_partial)
        return full_weight[gathered_ids[:3]]

    monkeypatch.setattr(qwen4_exp_ple, "allgather", fake_allgather)
    monkeypatch.setattr(qwen4_exp_ple, "reducescatter", fake_reducescatter)

    output = module.embed(
        local_ids,
        physical_tokens=3,
        all_rank_num_tokens=[3, 1],
    )

    torch.testing.assert_close(output, full_weight[local_ids])
