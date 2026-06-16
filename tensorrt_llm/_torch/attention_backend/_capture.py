# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Env-gated capture of attention-backend forward calls for the test suite.

When ``TRTLLM_ATTN_CAPTURE_DIR`` is set, ``create_attention`` wraps each backend
instance's ``forward`` with :func:`wrap_backend_for_capture`. Every call appends
one JSON line describing the case *shape + config* (no tensor values) to
``{dir}/cases_rank{rank}.jsonl``. These lines deserialize into the test suite's
``BackendCase`` for replay.

The wrapper is intentionally defensive: a capture failure must never perturb the
model's forward pass, so recording errors are swallowed.
"""

import json
import os
import threading
from dataclasses import asdict

_lock = threading.Lock()


def _dtype_str(dt) -> str:
    return str(dt).replace("torch.", "")


def _rank() -> int:
    try:
        from tensorrt_llm._utils import mpi_rank

        return int(mpi_rank())
    except Exception:
        return 0


def _kv_dtype_str(backend, compute_dtype: str) -> str:
    qc = getattr(backend, "quant_config", None)
    if qc is not None:
        try:
            mode = qc.layer_quant_mode
            if mode.has_fp4_kv_cache():
                return "nvfp4"
            if mode.has_fp8_kv_cache():
                return "float8_e4m3fn"
        except Exception:
            pass
    return compute_dtype


def _rope_dict(backend):
    rp = getattr(backend, "rope_params", None)
    if rp is None or getattr(rp, "dim", 0) == 0:
        return None
    try:
        d = asdict(rp)
    except Exception:
        return None
    # Enum -> int (JSON-serializable); tuples -> lists.
    if "scale_type" in d:
        d["scale_type"] = int(rp.scale_type)
    for key in ("short_factor", "long_factor"):
        if isinstance(d.get(key), tuple):
            d[key] = list(d[key])
    # rope_gptj == 1 -> not neox.
    d["is_neox"] = getattr(backend, "position_embedding_type", 2) != 1
    return d


def _attention_mask(forward_args, kwargs):
    if forward_args is not None and getattr(forward_args, "attention_mask", None) is not None:
        return forward_args.attention_mask
    return kwargs.get("attention_mask")


def _window_size(forward_args, kwargs):
    if (
        forward_args is not None
        and getattr(forward_args, "attention_window_size", None) is not None
    ):
        return forward_args.attention_window_size
    return kwargs.get("attention_window_size")


def _mla_dims(backend) -> dict:
    """MLA latent dims from the backend (only when MLA is enabled)."""
    if not getattr(backend, "is_mla_enable", False):
        return {}
    mla = getattr(backend, "mla_params", None)
    return dict(
        v_head_dim=getattr(backend, "v_head_dim", None),
        q_lora_rank=getattr(mla, "q_lora_rank", None),
        kv_lora_rank=getattr(mla, "kv_lora_rank", None),
        qk_nope_head_dim=getattr(mla, "qk_nope_head_dim", None),
        qk_rope_head_dim=getattr(mla, "qk_rope_head_dim", None),
    )


def _build_case_dict(backend, backend_name, q, metadata, forward_args, kwargs):
    mgr = getattr(metadata, "kv_cache_manager", None)
    seq_lens = metadata.seq_lens.tolist() if metadata.seq_lens is not None else []
    num_seqs = len(seq_lens)

    kv_params = getattr(metadata, "kv_cache_params", None)
    num_cached = list(getattr(kv_params, "num_cached_tokens_per_seq", None) or [0] * num_seqs)

    mask = _attention_mask(forward_args, kwargs)
    causal = (mask is None) or (str(getattr(mask, "value", mask)) == "causal")

    compute_dtype = _dtype_str(q.dtype)
    sparse_cfg = getattr(backend, "sparse_attention_config", None)
    page_size = getattr(mgr, "tokens_per_block", None) or 64

    # Cross-attention: record encoder KV lengths when seq_lens_kv differs.
    seq_lens_kv = None
    try:
        if getattr(metadata, "is_cross", False) and metadata.seq_lens_kv is not None:
            seq_lens_kv = metadata.seq_lens_kv.tolist()
    except Exception:
        pass

    rope = _rope_dict(backend)
    # TRTLLM (and other support_fused_rope backends) rotate q/k in-kernel.
    fused_rope = bool(rope) and bool(type(backend).support_fused_rope())

    return dict(
        num_heads=backend.num_heads,
        num_kv_heads=backend.num_kv_heads,
        head_dim=backend.head_dim,
        seq_lens=seq_lens,
        seq_lens_kv=seq_lens_kv,
        num_cached_tokens=num_cached,
        num_contexts=metadata.num_contexts,
        dtype=compute_dtype,
        kv_dtype=_kv_dtype_str(backend, compute_dtype),
        causal=bool(causal),
        sliding_window=_window_size(forward_args, kwargs),
        q_scaling=float(getattr(backend, "q_scaling", 1.0) or 1.0),
        page_size=int(page_size),
        cache="none" if mgr is None else "paged",
        sparse="off" if sparse_cfg is None else "degenerate",
        rope=rope,
        fused_rope=fused_rope,
        is_mla=bool(getattr(backend, "is_mla_enable", False)),
        use_kv_cache_manager_v2=type(mgr).__name__ == "KVCacheManagerV2",
        **_mla_dims(backend),
    )


def wrap_backend_for_capture(backend, backend_name: str, capture_dir: str):
    """Install a recording wrapper on ``backend.forward`` (capture-all to JSONL)."""
    os.makedirs(capture_dir, exist_ok=True)
    out_path = os.path.join(capture_dir, f"cases_rank{_rank()}.jsonl")
    orig_forward = backend.forward

    def forward(q, k, v, metadata, forward_args=None, **kwargs):
        try:
            case = _build_case_dict(backend, backend_name, q, metadata, forward_args, kwargs)
            with _lock:
                with open(out_path, "a") as f:
                    f.write(json.dumps(case) + "\n")
        except Exception:
            # Capture must never break the model forward.
            pass
        return orig_forward(q, k, v, metadata, forward_args=forward_args, **kwargs)

    backend.forward = forward
    return backend
