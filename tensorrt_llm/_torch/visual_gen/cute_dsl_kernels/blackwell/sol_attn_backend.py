"""Model adapters for Sol-Attn.

The kernel-facing API accepts contiguous BF16
``[batch, tokens, heads, 128]`` Q/K/V, ``tau``, ``thresh_type``,
``kv_splits``, and an optional exact KV sink range.

This module only owns model integration:

* a diffusers self-attention dispatch hook for ordinary joint-free sequences;
* HunyuanVideo's padded ``[video, text]`` sequence handling;
* dense text-query rows plus an exact text KV sink for MMDiT;
* optional model-level Morton ordering and dense step/layer guards.

CuTe DSL imports and compilation remain lazy. Ineligible calls delegate to the
original dense attention implementation.

TRT-LLM's own dispatch path (``attention_backend/cute_dsl/sol_attn.py``,
``SolAttnAttention``) consumes only ``_run_sol_attn_bthd`` /
``sol_attn_supported`` from this module. Everything below the "diffusers /
Sana adapter" marker -- the ``make_sol_attn_dispatch`` monkey-patch machinery,
Morton-order helpers, and HunyuanVideo/MMDiT-specific code -- is retained
verbatim from the original diffusers-based port for provenance and is not
called by TRT-LLM today. ``SolAttnStepContext`` in ``sol_attn.py`` is the
production dense-prefix/step counter TRT-LLM actually uses; ``_SolContext``
below is a separate, unused implementation of the same idea from that port
and must not be assumed to be kept in sync with it.
"""

from __future__ import annotations

import functools
import importlib
import math
import os
import sys
from pathlib import Path
from typing import Callable

from tensorrt_llm.logger import logger

HEAD_DIM = 128
DEFAULT_TAU = 1.0
DEFAULT_THRESH_TYPE = "diag"
_DEFAULT_SCALE = HEAD_DIM**-0.5
_PACKAGE_ROOT = Path(__file__).resolve().parent / "sol_attn"
_IMPORT_ROOT = _PACKAGE_ROOT.parent


@functools.lru_cache(maxsize=1)
def _load_sol_attn() -> Callable:
    """Load the integrated package and reject a conflicting preloaded copy."""

    import_root = str(_IMPORT_ROOT)
    loaded = sys.modules.get("sol_attn")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if not loaded_file.is_relative_to(_PACKAGE_ROOT.resolve()):
            raise RuntimeError(
                "a different sol_attn package is already loaded from "
                f"{loaded_file}; import the Sana backend before other copies"
            )
        return loaded.sol_attn

    if import_root not in sys.path:
        sys.path.insert(0, import_root)
    module = importlib.import_module("sol_attn")
    loaded_file = Path(module.__file__).resolve()
    if not loaded_file.is_relative_to(_PACKAGE_ROOT.resolve()):
        raise RuntimeError(f"expected Sol-Attn under {_PACKAGE_ROOT}, got {loaded_file}")
    return module.sol_attn


def sol_attn_supported(q) -> bool:
    """Whether ``q`` can use a CuTe or portable Triton Sol-Attn backend."""

    try:
        import torch
    except Exception:  # pragma: no cover - torch is a runtime dependency
        return False
    if not (hasattr(q, "is_cuda") and q.is_cuda):
        return False
    if q.ndim != 4 or q.shape[-1] != HEAD_DIM or q.dtype != torch.bfloat16:
        return False
    try:
        arch = tuple(torch.cuda.get_device_capability(q.device))
        return arch[0] >= 8
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _cute_runtime_available() -> bool:
    """Whether model dispatch can use one of the optional CuTe kernels."""

    try:
        import cuda.bindings.driver  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _resolve_kv_splits(q, kv_splits: int | str | None) -> int:
    """Resolve the integration-only ``auto`` policy to the public integer API."""

    import torch

    if kv_splits not in (None, "auto"):
        return int(kv_splits)
    arch = tuple(torch.cuda.get_device_capability(q.device))
    if arch == (9, 0) and q.shape[1] >= 65536 and _cute_runtime_available():
        return 4
    return 1


def _dense_bthd(q, k, v):
    import torch

    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    ).transpose(1, 2)


def _run_sol_attn_bthd(
    q,
    k,
    v,
    *,
    tau: float = DEFAULT_TAU,
    thresh_type: str = DEFAULT_THRESH_TYPE,
    kv_splits: int | str | None = "auto",
    sink_start: int | None = None,
    sink_tokens: int = 0,
    dense_fn: Callable | None = None,
):
    """Run Sol-Attn on contiguous BTHD tensors, with a safe dense fallback."""

    q0, k0, v0 = q.contiguous(), k.contiguous(), v.contiguous()

    def dense():
        if dense_fn is not None:
            return dense_fn(q0, k0, v0)
        return _dense_bthd(q0, k0, v0)

    if (
        not sol_attn_supported(q0)
        or k0.shape != q0.shape
        or v0.shape != q0.shape
        or k0.dtype != q0.dtype
        or v0.dtype != q0.dtype
    ):
        return dense()

    try:
        kernel = _load_sol_attn()
        out = kernel(
            q0,
            k0,
            v0,
            tau=float(tau),
            thresh_type=str(thresh_type),
            kv_splits=_resolve_kv_splits(q0, kv_splits),
            sink_start=sink_start,
            sink_tokens=int(sink_tokens),
        )
        _SOL_STATS["kernel_calls"] += 1
        return out
    except Exception as exc:
        if os.environ.get("SOL_ATTN_STRICT", "0") == "1":
            raise
        logger.warning_once(
            f"[sol-attn] kernel raised {type(exc).__name__}: {exc}; falling back to dense "
            "SDPA for this call. Set SOL_ATTN_STRICT=1 to raise instead of silently falling "
            "back.",
            key=(type(exc).__name__, str(exc)),
        )
        return dense()


# --- diffusers / Sana adapter (below): not called by TRT-LLM's dispatch path.
# Retained verbatim from the original port for provenance -- see module
# docstring. TRT-LLM's own dense-prefix counter is SolAttnStepContext in
# ../attention_backend/cute_dsl/sol_attn.py, not the _SolContext below.

_MORTON_CACHE: dict[tuple[int, int, int], tuple] = {}


def _morton3d_perm(grid, device):
    """Return canonical x/y/z-interleaved Morton permutation and its inverse."""

    import torch

    key = tuple(int(x) for x in grid)
    hit = _MORTON_CACHE.get(key)
    if hit is None:
        frames, height, width = key
        total = frames * height * width
        linear = torch.arange(total, dtype=torch.long)
        frame_area = height * width
        z = linear // frame_area
        rem = linear - z * frame_area
        y = rem // width
        x = rem - y * width

        def part1by2(value):
            value = value & 0x1FFFFF
            value = (value | (value << 32)) & 0x1F00000000FFFF
            value = (value | (value << 16)) & 0x1F0000FF0000FF
            value = (value | (value << 8)) & 0x100F00F00F00F00F
            value = (value | (value << 4)) & 0x10C30C30C30C30C3
            value = (value | (value << 2)) & 0x1249249249249249
            return value

        code = part1by2(x) | (part1by2(y) << 1) | (part1by2(z) << 2)
        perm = linear[torch.argsort(code)]
        hit = (perm, torch.argsort(perm))
        _MORTON_CACHE[key] = hit
    return hit[0].to(device), hit[1].to(device)


def install_wan_morton_forward(transformer, grid) -> int:
    """Apply one model-level Morton permutation around the Wan block stack."""

    device = next(transformer.parameters()).device
    perm, inverse = _morton3d_perm(grid, device)
    original_rope = transformer.rope.forward

    def rope_forward(hidden_states):
        freq_cos, freq_sin = original_rope(hidden_states)
        return (
            freq_cos.index_select(1, perm),
            freq_sin.index_select(1, perm),
        )

    def pre_hook(_module, args):
        return (args[0].index_select(1, perm),) + tuple(args[1:])

    def post_hook(_module, _args, output):
        return output.index_select(1, inverse)

    transformer.rope.forward = rope_forward
    transformer.blocks[0].register_forward_pre_hook(pre_hook)
    transformer.blocks[-1].register_forward_hook(post_hook)
    return int(perm.numel())


def _parse_layer_ranges(spec: str) -> frozenset[int]:
    layers: set[int] = set()
    for item in str(spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            layers.update(range(int(start), int(end) + 1))
        else:
            layers.add(int(item))
    return frozenset(layers)


class _SolContext:
    step = -1
    layer = 0
    dense_steps = 0
    dense_layers = frozenset()
    tau = DEFAULT_TAU
    thresh_type = DEFAULT_THRESH_TYPE
    kv_splits: int | str | None = "auto"
    video_len = 0
    grid = None
    morton = False


_SOL_CTX = _SolContext()
_SOL_STATS = {
    "dispatch_calls": 0,
    "kernel_calls": 0,
    "hunyuan_calls": 0,
    "dense_guard_calls": 0,
}


def sol_attn_begin_forward() -> None:
    """Advance the denoising-step clock and reset the layer counter."""

    _SOL_CTX.step += 1
    _SOL_CTX.layer = 0


def reset_sol_attn_state() -> None:
    """Reset the model-integration clock after an untimed warmup generation."""

    _SOL_CTX.step = -1
    _SOL_CTX.layer = 0
    for key in _SOL_STATS:
        _SOL_STATS[key] = 0


def get_sol_attn_stats() -> dict[str, int]:
    """Return lightweight model-integration counters for run validation."""

    return {key: int(value) for key, value in _SOL_STATS.items()}


def _use_dense_layer() -> bool:
    _SOL_STATS["dispatch_calls"] += 1
    layer = _SOL_CTX.layer
    _SOL_CTX.layer += 1
    dense = _SOL_CTX.step < _SOL_CTX.dense_steps or layer in _SOL_CTX.dense_layers
    if dense:
        _SOL_STATS["dense_guard_calls"] += 1
    return dense


_SELF_OP_REGISTERED = False
_HUNYUAN_OP_REGISTERED = False


def _ensure_self_op() -> None:
    global _SELF_OP_REGISTERED
    if _SELF_OP_REGISTERED:
        return
    import torch

    @torch.library.custom_op(
        "sana_sol_attn::self_attention",
        mutates_args=(),
        schema="(Tensor q, Tensor k, Tensor v) -> Tensor",
    )
    def self_attention(q, k, v):
        if _use_dense_layer():
            return _dense_bthd(q, k, v)
        return _run_sol_attn_bthd(
            q,
            k,
            v,
            tau=_SOL_CTX.tau,
            thresh_type=_SOL_CTX.thresh_type,
            kv_splits=_SOL_CTX.kv_splits,
        )

    @self_attention.register_fake
    def _(q, k, v):
        return torch.empty_like(q)

    _SELF_OP_REGISTERED = True


def make_sol_attn_dispatch(
    original_dispatch,
    *,
    tau: float = DEFAULT_TAU,
    thresh_type: str = DEFAULT_THRESH_TYPE,
    kv_splits: int | str | None = "auto",
    dense_steps: int = 0,
    dense_layers: str = "",
):
    """Wrap diffusers self-attention with the released Sol-Attn BTHD API."""

    _SOL_CTX.tau = float(tau)
    _SOL_CTX.thresh_type = str(thresh_type)
    _SOL_CTX.kv_splits = kv_splits
    _SOL_CTX.dense_steps = int(dense_steps)
    _SOL_CTX.dense_layers = _parse_layer_ranges(dense_layers)
    _ensure_self_op()
    import torch

    op = torch.ops.sana_sol_attn.self_attention

    def dispatch(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        attention_kwargs=None,
        *,
        backend=None,
        parallel_config=None,
    ):
        eligible = (
            parallel_config is None
            and attn_mask is None
            and not is_causal
            and dropout_p == 0.0
            and not enable_gqa
            and (scale is None or math.isclose(float(scale), _DEFAULT_SCALE))
            and query.shape == key.shape == value.shape
            and sol_attn_supported(query)
        )
        if eligible:
            return op(
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
            )
        return original_dispatch(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            attention_kwargs,
            backend=backend,
            parallel_config=parallel_config,
        )

    return dispatch


def _dense_hunyuan(q, k, v, key_valid):
    import torch

    batch, _heads, tokens, _dim = q.shape
    mask = torch.zeros(
        batch,
        1,
        1,
        tokens,
        device=q.device,
        dtype=q.dtype,
    )
    mask = mask.masked_fill(
        ~key_valid[:, None, None, :],
        float("-inf"),
    )
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
    )


def _run_mmdit_sol_attn_bthd(
    q,
    k,
    v,
    *,
    video_len: int,
    key_valid,
    tau: float = DEFAULT_TAU,
    thresh_type: str = DEFAULT_THRESH_TYPE,
    kv_splits: int | str | None = "auto",
    grid=None,
    morton: bool = False,
):
    """Run joint MMDiT attention with exact text K/V and dense text Q.

    Inputs use BHSD and the model's native ``[video, padded-text]`` order.
    Padding is cropped before the kernel. Sol-Attn computes the valid joint
    sequence with the text suffix passed as an exact KV sink. Valid text query
    rows are then replaced by dense SDPA; padded query rows remain zero.
    """

    import torch

    q0, k0, v0 = q.contiguous(), k.contiguous(), v.contiguous()
    valid = key_valid.bool()

    def dense():
        return _dense_hunyuan(q0, k0, v0, valid)

    batch, heads, tokens, dim = q0.shape
    video_len = int(video_len)
    if (
        q0.shape != k0.shape
        or q0.shape != v0.shape
        or video_len <= 0
        or video_len >= tokens
        or dim != HEAD_DIM
    ):
        return dense()

    text_valid_per_batch = valid[:, video_len:].sum(dim=1)
    text_tokens = int(text_valid_per_batch[0].item())
    if not bool((text_valid_per_batch == text_tokens).all()):
        return dense()
    effective_tokens = video_len + text_tokens
    expected_valid = torch.arange(tokens, device=q0.device)[None, :] < effective_tokens
    if not bool((valid == expected_valid).all()):
        return dense()

    try:
        joint_index = torch.arange(effective_tokens, device=q0.device)
        if morton and grid is not None and math.prod(int(x) for x in grid) == video_len:
            perm, _inverse = _morton3d_perm(grid, q0.device)
            joint_index = torch.cat(
                [
                    perm,
                    joint_index[video_len:effective_tokens],
                ]
            )
        inverse_joint = torch.argsort(joint_index)

        q_joint = (
            q0[:, :, :effective_tokens].index_select(2, joint_index).transpose(1, 2).contiguous()
        )
        k_joint = (
            k0[:, :, :effective_tokens].index_select(2, joint_index).transpose(1, 2).contiguous()
        )
        v_joint = (
            v0[:, :, :effective_tokens].index_select(2, joint_index).transpose(1, 2).contiguous()
        )
        out_joint = _run_sol_attn_bthd(
            q_joint,
            k_joint,
            v_joint,
            tau=tau,
            thresh_type=thresh_type,
            kv_splits=kv_splits,
            sink_start=video_len,
            sink_tokens=text_tokens,
        ).transpose(1, 2)
        out_joint = out_joint.index_select(2, inverse_joint)

        out = torch.zeros(
            batch,
            heads,
            tokens,
            dim,
            device=q0.device,
            dtype=q0.dtype,
        )
        out[:, :, :video_len] = out_joint[:, :, :video_len]
        if text_tokens:
            out[:, :, video_len:effective_tokens] = (
                torch.nn.functional.scaled_dot_product_attention(
                    q0[:, :, video_len:effective_tokens],
                    k0[:, :, :effective_tokens],
                    v0[:, :, :effective_tokens],
                )
            )
        _SOL_STATS["hunyuan_calls"] += 1
        return out
    except Exception as exc:
        if os.environ.get("SOL_ATTN_STRICT", "0") == "1":
            raise
        print(
            f"[sol-attn:hunyuan] dense fallback: {type(exc).__name__}: {exc}",
            flush=True,
        )
        return dense()


def _ensure_hunyuan_op() -> None:
    global _HUNYUAN_OP_REGISTERED
    if _HUNYUAN_OP_REGISTERED:
        return
    import torch

    @torch.library.custom_op(
        "sana_sol_attn::hunyuan_attention",
        mutates_args=(),
        schema=("(Tensor q, Tensor k, Tensor v, Tensor key_valid) -> Tensor"),
    )
    def hunyuan_attention(q, k, v, key_valid):
        if _use_dense_layer():
            return _dense_hunyuan(q, k, v, key_valid.bool())
        return _run_mmdit_sol_attn_bthd(
            q,
            k,
            v,
            video_len=_SOL_CTX.video_len,
            key_valid=key_valid,
            tau=_SOL_CTX.tau,
            thresh_type=_SOL_CTX.thresh_type,
            kv_splits=_SOL_CTX.kv_splits,
            grid=_SOL_CTX.grid,
            morton=_SOL_CTX.morton,
        )

    @hunyuan_attention.register_fake
    def _(q, k, v, key_valid):
        return torch.empty_like(q)

    _HUNYUAN_OP_REGISTERED = True


def make_mmdit_sol_attn_dispatch(
    original_dispatch,
    *,
    video_len: int,
    tau: float = DEFAULT_TAU,
    thresh_type: str = DEFAULT_THRESH_TYPE,
    kv_splits: int | str | None = "auto",
    dense_steps: int = 0,
    dense_layers: str = "",
    grid=None,
    morton: bool = False,
):
    """Wrap joint MMDiT attention with exact text K/V and dense text Q."""

    _SOL_CTX.video_len = int(video_len)
    _SOL_CTX.tau = float(tau)
    _SOL_CTX.thresh_type = str(thresh_type)
    _SOL_CTX.kv_splits = kv_splits
    _SOL_CTX.dense_steps = int(dense_steps)
    _SOL_CTX.dense_layers = _parse_layer_ranges(dense_layers)
    _SOL_CTX.grid = None if grid is None else tuple(int(value) for value in grid)
    _SOL_CTX.morton = bool(morton)
    _ensure_hunyuan_op()
    import torch

    op = torch.ops.sana_sol_attn.hunyuan_attention
    video_len = int(video_len)

    def dispatch(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
        attention_kwargs=None,
        *,
        backend=None,
        parallel_config=None,
    ):
        eligible = (
            parallel_config is None
            and attn_mask is not None
            and not is_causal
            and dropout_p == 0.0
            and not enable_gqa
            and (scale is None or math.isclose(float(scale), _DEFAULT_SCALE))
            and query.shape == key.shape == value.shape
            and query.shape[1] > video_len
            and sol_attn_supported(query)
        )
        if not eligible:
            return original_dispatch(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                attention_kwargs,
                backend=backend,
                parallel_config=parallel_config,
            )

        key_valid = attn_mask
        if key_valid.dtype != torch.bool:
            key_valid = key_valid > -1.0
        key_valid = key_valid.reshape(key_valid.shape[0], -1)
        if key_valid.shape[1] != query.shape[1]:
            return original_dispatch(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                attention_kwargs,
                backend=backend,
                parallel_config=parallel_config,
            )
        out = op(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            key_valid,
        )
        return out.transpose(1, 2)

    return dispatch


__all__ = [
    "make_mmdit_sol_attn_dispatch",
    "make_sol_attn_dispatch",
]
