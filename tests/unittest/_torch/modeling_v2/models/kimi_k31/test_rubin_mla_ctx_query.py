# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep16's context MLA query forms and the prefill indexer's widening (CPU, from source).

The context call of a dep16 DSA layer hands the attention op a 128-head query, and the op's FP8
FMHA reads it as E4M3. In the BF16 form the padding heads of a BF16 buffer are zeroed, W_UK
writes the real heads' latent columns, and the op ropes the RoPE columns into the buffer and
converts all of it (unit scale, saturated). In the FP8 form (``mla_ctx_fused_q_fp8``) W_UK
writes the same BF16 product, ``rubin_mla_ctx_q_fp8`` converts the real heads' latent columns
into an E4M3 buffer with zero padding heads, the op's RoPE kernel writes the RoPE columns into
that buffer in E4M3 and the op reads no BF16 query (a one-row stand-in carries its shape). Both
forms hand the FMHA the same bytes with the same other arguments.
A prefill step widens the indexer's input with ``rubin_upcast_fp32`` (the bits of ``x.float()``)
into the TF32 projection; a decode step keeps ``x.float()``.
A rank's first prefill layer logs one line naming the query form and the widening that ran; the
later layers and the decode steps log nothing.

The product's functions and constants run from its source; the entries, the indexer and the
attention op are recording stand-ins, the op modeled at the query bytes its FMHA reads.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_PKG_ROOT = Path(importlib.util.find_spec("tensorrt_llm").submodule_search_locations[0]).resolve()
_DEP16 = (
    _PKG_ROOT
    / "_torch/_experimental/modeling_v2/models/kimi_k31/kimi_k31_pro_nvfp4__sm_107__dep16"
    / "modeling.py"
)
_TARGET = "kimi_k31_pro_nvfp4__sm_107__dep16"
_FP8 = torch.float8_e4m3fn
# The constants and functions the context call reads, from the product.
_NAMES = (
    "_K31",
    "_MLA_QUERY_HEADS",
    "_CONSTANTS",
    "_MLA_Q_LORA",
    "_MLA_KV_LORA",
    "_MLA_ROPE",
    "_MLA_NOPE",
    "_MLA_V",
    "_MLA_QK",
    "_MLA_LATENT",
    "_MLA_HEADS",
    "_MLA_OUT_HEADS",
    "_MLA_PAD_HEADS",
    "_MLA_PAD_ROW",
    "_MLA_CTX_FUSED_Q_FP8",
    "_MLA_CONTEXT",
    "_MLA_ATTN",
    "_MLA_ATTN_PAD128",
    "_mla_ctx_query",
    "_mla_ctx_query_fp8",
    "_mla_context",
    "_widen_fp32",
    "_mla_prefill_witness",
    "mla_prefill",
    "mla_generate",
)
# The attention op's arguments that carry the query; the others must be the same in both forms.
_QUERY_ARGS = ("q", "quant_q_buffer", "quant_scale_qkv", "output")


def _nodes(names) -> list:
    """The module-level definitions of ``names`` in the product, in source order."""
    wanted, nodes = set(names), []
    for node in ast.parse(_DEP16.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found = {node.name}
        elif isinstance(node, ast.Assign):
            found = {n.id for t in node.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
        else:
            continue
        if found & wanted:
            nodes.append(node)
            wanted -= found
    assert not wanted, f"the product no longer defines {sorted(wanted)}"
    return nodes


def _e4m3(t: torch.Tensor) -> torch.Tensor:
    """The attention op's conversion of a BF16 value: unit scale, saturated to +-448."""
    return t.float().clamp(-448.0, 448.0).to(_FP8)


class _Logger:
    """The engine logger's ``info_once``: a key logs one line per process (here, per object)."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self._keys: set[str] = set()

    def info_once(self, msg: str, *, key: str) -> None:
        if key not in self._keys:
            self._keys.add(key)
            self.lines.append(msg)


def _namespace(log: list, calls: list, fused: bool | None = None, logger=None) -> dict:
    """The product's context-call code over recording stand-ins (``fused``: the form forced)."""

    def rubin_pad_heads(dst, src, heads):
        log.append(("pad_heads", src is None))
        if src is not None:
            dst[:, :heads] = src
        dst[:, heads:] = 0

    def bmm_out(a, b, out):
        log.append(("bmm", None))
        out.copy_(torch.bmm(a.float(), b.float()).to(out.dtype))

    def rubin_mla_ctx_q_fp8(dst, src, heads):
        log.append(("mla_ctx_q_fp8", None))
        dst[:, heads:] = torch.zeros((), dtype=dst.dtype)
        dst[:, :heads] = _e4m3(src[:, :heads])

    def _idx_prefill(i, q8, w, k8, ks, st, rt):
        log.append(("topk", None))
        return torch.zeros(1, dtype=torch.int32)

    def _idx_remap(i, topk, req_idx, block_table, st, rt):
        return torch.zeros(1, dtype=torch.int32)

    def thop_attention(**kw):
        """The op at its query: the RoPE kernel writes the RoPE columns (the identity table: the
        padded ``q_pe``), then the FMHA reads E4M3 rows; returns the latent columns it read."""
        log.append(("attention", None))
        heads, lora = kw["num_heads"], kw["kv_lora_rank"]
        rows = kw["q"].shape[0]
        if kw["quant_q_buffer"] is None:
            assert kw["quant_scale_qkv"] is None
            query = kw["q"].view(rows, heads, -1).clone()
            query[..., lora:] = kw["q_pe"]
            q8 = _e4m3(query)
        else:
            stand_in = kw["q"]
            assert stand_in.untyped_storage().nbytes() == stand_in.shape[1] * 2, (
                "the BF16 query is a one-row stand-in"
            )
            scale = kw["quant_scale_qkv"]
            assert scale.dtype == torch.float32 and scale.numel() == 1 and float(scale) == 1.0
            q8 = kw["quant_q_buffer"].clone()
            assert q8.dtype == _FP8 and tuple(q8.shape) == (rows, heads, kw["head_size"])
            q8[..., lora:] = _e4m3(kw["q_pe"])
        calls.append((q8.view(torch.uint8), {k: v for k, v in kw.items() if k not in _QUERY_ARGS}))
        out = kw["output"].view(rows, heads, lora)
        out.copy_(q8[..., :lora].float())
        return kw["output"]

    def _mla_expand(v_b_t, latent, out):
        out.copy_(latent.reshape(out.shape[0], -1)[:, : out.shape[1]])

    ns = dict(
        torch=torch,
        MappingProxyType=MappingProxyType,
        rubin_pad_heads=rubin_pad_heads,
        bmm_out=bmm_out,
        rubin_mla_ctx_q_fp8=rubin_mla_ctx_q_fp8,
        _idx_prefill=_idx_prefill,
        _idx_remap=_idx_remap,
        thop_attention=thop_attention,
        _mla_expand=_mla_expand,
        _mla_window=lambda i, st, rt: 4096,
        _TARGET=_TARGET,
        _MlaW=object,
        _Step=object,
        _Rt=object,
        logger=logger,
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(_nodes(_NAMES), [])), str(_DEP16), "exec"), ns
    )
    if fused is not None:
        ns["_MLA_CTX_FUSED_Q_FP8"] = fused
    return ns


def _context_case(rows: int, seed: int):
    """A layer's weights and a context call's inputs at the product's shapes (``rows`` rows)."""
    ns = _namespace([], [])
    g = torch.Generator().manual_seed(seed)
    heads, nope, rope, lora = ns["_MLA_HEADS"], ns["_MLA_NOPE"], ns["_MLA_ROPE"], ns["_MLA_KV_LORA"]
    # Values beyond E4M3's range and below its subnormals included.
    q = (torch.randn(rows, heads * (nope + rope), generator=g) * 40).to(torch.bfloat16)
    k_b = (torch.randn(heads, nope, lora, generator=g) * 0.5).to(torch.bfloat16)
    layer = SimpleNamespace(k_b=k_b, v_b_t=None)
    st = SimpleNamespace(
        dsa=SimpleNamespace(
            sparse_ctx={"sparse_mla_topk": 2048}, req_idx_ctx=None, block_table_ctx=None
        ),
        md=SimpleNamespace(cu_q_seqlens="cu_q", cu_kv_seqlens="cu_kv"),
    )
    rt = SimpleNamespace(
        mla_rope="rope_table", dsa_pools=SimpleNamespace(local={3: 0}, rope_positions=1 << 20)
    )
    return layer, q, st, rt


def _same(a, b) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and torch.equal(a, b)
    return a == b


def _run_context(fused: bool, rows: int = 3, seed: int = 0):
    log, calls = [], []
    ns = _namespace(log, calls, fused)
    layer, q, st, rt = _context_case(rows, seed)
    out = torch.zeros(rows, ns["_MLA_OUT_HEADS"] * ns["_MLA_V"], dtype=torch.bfloat16)
    ns["_mla_context"](layer, 3, q, "latent", "q8", "w", "k8", "ks", out, st, rt)
    return out, log, calls


def test_the_constant_drives_the_form():
    ns = _namespace([], [])
    assert ns["_MLA_CTX_FUSED_Q_FP8"] is bool(ns["_CONSTANTS"]["context"]["mla_ctx_fused_q_fp8"])


@pytest.mark.parametrize("rows", (1, 3, 17))
def test_both_forms_hand_the_fmha_the_same_query(rows):
    out0, log0, calls0 = _run_context(False, rows, rows)
    out1, log1, calls1 = _run_context(True, rows, rows)
    assert len(calls0) == len(calls1) == 1, "one context call per layer"
    (q0, kw0), (q1, kw1) = calls0[0], calls1[0]
    assert torch.equal(q0, q1), "the FMHA would read other query bytes"
    assert kw0.keys() == kw1.keys()
    for key in kw0:
        assert _same(kw0[key], kw1[key]), f"argument {key} differs between the forms"
    assert torch.equal(out0.view(torch.int16), out1.view(torch.int16))
    # BF16: the padding heads of the query zeroed (src None), no conversion; FP8: no zeroing of
    # a BF16 query, one conversion right after W_UK. The RoPE columns are padded in both.
    ops0, ops1 = [op for op, _ in log0], [op for op, _ in log1]
    assert ("pad_heads", True) in log0 and ("pad_heads", True) not in log1
    assert "mla_ctx_q_fp8" not in ops0 and ops1.count("mla_ctx_q_fp8") == 1
    assert ops1.index("mla_ctx_q_fp8") == ops1.index("bmm") + 1, "converted right after W_UK"
    assert ops1.index("mla_ctx_q_fp8") < ops1.index("topk"), "the BF16 buffer free before Top-K"
    assert [op for op in ops0 if op != "pad_heads"] == [
        op for op in ops1 if op not in ("pad_heads", "mla_ctx_q_fp8")
    ]


def test_the_query_scale_is_each_calls_own():
    """quant_scale_qkv is not a layer constant: the context call passes the FP8 form's unit scale
    (None for the BF16 form) and the generation calls (96 heads, or 128 under mla_gen_pad128)
    None, each as a keyword."""
    ns = _namespace([], [])
    assert (
        "quant_scale_qkv" not in ns["_MLA_ATTN"] and "quant_scale_qkv" not in ns["_MLA_ATTN_PAD128"]
    )
    body = ast.parse(_DEP16.read_text()).body
    # the generation calls are _mla_attend's (both decode orders, dsa_logits_first on or off)
    for name, want, count in (("_mla_context", "scale", 1), ("_mla_attend", "None", 2)):
        fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
        calls = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "thop_attention"
        ]
        assert len(calls) == count, (name, len(calls))
        for call in calls:
            kw = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg is not None}
            assert kw["quant_scale_qkv"] == want, name
            assert call.keywords[-1].arg is None, f"{name}: the layer constants are spread last"


def _prefill_namespace(log: list, logger: _Logger) -> dict:
    """``mla_prefill`` and ``mla_generate`` over stand-ins that record the indexer's input."""
    ns = _namespace([], [], logger=logger)

    def rubin_upcast_fp32(x):
        log.append(("upcast", x))
        return x.float()

    def _idx_project(L, x, qr, st, rt, xf=None):
        log.append(("idx_project", xf))
        rows = x.shape[0]
        return (
            torch.zeros(rows, 1),
            torch.zeros(rows, 1),
            torch.zeros(rows, 1),
            torch.zeros(rows, 1),
        )

    ns.update(
        rubin_upcast_fp32=rubin_upcast_fp32,
        _idx_project=_idx_project,
        _mla_project=lambda L, x, st, rt: (x, x),
        rubin_linear=lambda x, w, tf32: x,
        _idx_insert=lambda *a: log.append(("insert", None)),
        _mla_context=lambda *a: log.append(("context", None)),
        _mla_generation=lambda *a: log.append(("generation", None)),
        _mla_output=lambda L, x, out, n: out,
    )
    return ns


_LAYER = SimpleNamespace(q_b=None)


def _step(rows: int, ctx: int, gens: int, cuda_graph: bool = False):
    # An eager step: a CUDA graph's decode step runs the indexer's key half on its own stream
    # (dsa_idx_fork; test_rubin_dep16_dsa_fork.py).
    return SimpleNamespace(
        num_tokens=rows, num_ctx_tokens=ctx, num_generations=gens, cuda_graph=cuda_graph
    )


def test_prefill_widens_the_indexer_input_once():
    log, logger = [], _Logger()
    ns = _prefill_namespace(log, logger)
    x = torch.randn(6, 10).to(torch.bfloat16)
    ns["mla_prefill"](_LAYER, 3, x, _step(6, 4, 2), None)
    upcasts = [a for op, a in log if op == "upcast"]
    (xf,) = [a for op, a in log if op == "idx_project"]
    assert len(upcasts) == 1 and upcasts[0] is x
    assert xf.dtype == torch.float32 and torch.equal(
        xf.view(torch.int32), x.float().view(torch.int32)
    )
    assert [op for op, _ in log if op in ("context", "generation")] == ["context", "generation"]


def test_a_non_contiguous_input_keeps_its_widening():
    log, logger = [], _Logger()
    ns = _prefill_namespace(log, logger)
    x = torch.randn(10, 6).to(torch.bfloat16).t()
    wide = ns["_widen_fp32"](x)
    assert not [op for op, _ in log if op == "upcast"]
    assert torch.equal(wide.view(torch.int32), x.float().view(torch.int32))


def test_decode_keeps_its_widening():
    log, logger = [], _Logger()
    ns = _prefill_namespace(log, logger)
    ns["mla_generate"](_LAYER, 3, torch.randn(4, 10).to(torch.bfloat16), _step(4, 0, 4), None)
    assert [a for op, a in log if op == "idx_project"] == [None], "x.float() inside _idx_project"
    assert not [op for op, _ in log if op == "upcast"]
    assert logger.lines == [], "a decode step logs no prefill line"


def test_the_indexer_projection_takes_the_widened_input():
    """``_idx_project`` hands its TF32 projection ``xf`` when given, else ``x.float()``."""
    seen = []

    def rubin_linear(x, w, tf32):
        seen.append(x)
        return torch.zeros(x.shape[0], 128 + 32)

    ns = dict(
        torch=torch,
        rubin_linear=rubin_linear,
        cublas_mm=lambda a, b: torch.zeros(a.shape[0], 32 * 128),
        rubin_layer_norm=lambda x, w, b, eps: x,
        fused_cat_fp8=lambda a, b, flag: (
            torch.zeros(*a.shape[:-1], 128),
            torch.ones(*a.shape[:-1], 1),
        ),
        _IDX_DIM=128,
        _IDX_HEADS=32,
        _IDX_HALF=64,
        _IDX_K_NORM_EPS=1e-5,
        _IDX_WEIGHT_SCALE=1.0,
        _MlaW=object,
        _Step=object,
        _Rt=object,
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(_nodes(("_idx_project",)), [])),
            str(_DEP16),
            "exec",
        ),
        ns,
    )
    layer = SimpleNamespace(idx_wk_wp=None, idx_wq_b_t=None, idx_k_norm_w=None, idx_k_norm_b=None)
    x = torch.randn(3, 8).to(torch.bfloat16)
    xf = x.float()
    st = SimpleNamespace(cuda_graph=False)
    ns["_idx_project"](layer, x, x, st, None, xf)
    ns["_idx_project"](layer, x, x, st, None)
    assert seen[0] is xf
    assert torch.equal(seen[1].view(torch.int32), x.float().view(torch.int32))


_WITNESS = (
    "Kimi K3.1 Rubin prefill levers: kimi_k31_pro_nvfp4__sm_107__dep16: context query {}, "
    "indexer input widened by {}"
)


@pytest.mark.parametrize("fused", (False, True), ids=("bf16", "fp8"))
def test_a_rank_logs_its_prefill_forms_once(fused):
    log, logger = [], _Logger()
    ns = _prefill_namespace(log, logger)
    ns["_MLA_CTX_FUSED_Q_FP8"] = fused
    for _ in range(2):  # two prefill layers of one rank (one logger per process)
        ns["mla_prefill"](_LAYER, 3, torch.randn(5, 10).to(torch.bfloat16), _step(5, 5, 0), None)
    query = "FP8 (mla_ctx_fused_q_fp8)" if fused else "BF16"
    assert logger.lines == [_WITNESS.format(query, "rubin_upcast_fp32")]


def test_the_line_names_the_widening_that_ran():
    for contiguous, widen in ((True, "rubin_upcast_fp32"), (False, "x.float()")):
        log, logger = [], _Logger()
        ns = _prefill_namespace(log, logger)
        x = torch.randn(6, 10).to(torch.bfloat16)
        ns["_mla_prefill_witness"](x if contiguous else x.t())
        query = "FP8 (mla_ctx_fused_q_fp8)" if ns["_MLA_CTX_FUSED_Q_FP8"] else "BF16"
        assert logger.lines == [_WITNESS.format(query, widen)]
