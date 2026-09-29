# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified normalizer for ModelOpt-produced quantization configs.

ModelOpt emits ``hf_quant_config.json`` (and the inline
``config.json.quantization_config``) in two on-disk shapes:

- **Legacy** (modelopt 0.x): ``{producer, quantization: {quant_algo,
  kv_cache_quant_algo, exclude_modules, quantized_layers, ...}}``
- **Flat 1.x** (compressed-tensors-style): ``{producer, quant_method="modelopt",
  quant_algo, kv_cache_scheme, ignore, config_groups, ...}``

:func:`read_modelopt_quant_config` collapses either shape into the legacy
"quantization" inner-dict shape, which is what every TensorRT-LLM call site
consumes.
"""

from typing import Any, Dict, Optional

from ..logger import logger


def is_modelopt_quant_config(raw: Any) -> bool:
    """Return True if ``raw`` looks like a modelopt config (either shape).

    Detects either ``producer.name == "modelopt"`` or a ``quant_method``
    starting with ``"modelopt"``.
    """
    if not isinstance(raw, dict):
        return False
    if str(raw.get("quant_method", "")).lower().startswith("modelopt"):
        return True
    return (raw.get("producer") or {}).get("name") == "modelopt"


# Modelopt 1.x supports the same KV-cache algos as 0.x (FP8/NVFP4/INT8),
# but encodes ``kv_cache_scheme`` in two different shapes depending on the
# algo:
#   - FP8         -> dict   {"type": "float", "num_bits": 8}
#   - NVFP4, INT8 -> string "NVFP4" / "INT8"
# Both maps mirror `KV_CACHE_QUANT_ALGO_LIST = [FP8, NVFP4, INT8]`.
_KV_SCHEME_DICT_MAP = {
    ("float", 8): "FP8",
    ("float", 4): "NVFP4",
    ("int", 8): "INT8",
}
_KV_SCHEME_STRING_ALGOS = {"FP8", "NVFP4", "INT8"}

# ModelOpt ships algo spellings that predate the ``QuantAlgo`` enum names.
# Lookup is case-insensitive: the same alias appears lowercase as a top-level
# ``quant_algo`` (modelopt 0.x) and uppercase inside ``quantized_layers``
# (modelopt >= 0.45, e.g. ``nvidia/Kimi-K3-NVFP4``).
_QUANT_ALGO_ALIASES = {
    # Per-block FP8, weight-only. Same on-disk layout as the DeepSeek block
    # recipe (E4M3 weights + one FP32 scale per 128x128 block), so it maps onto
    # FP8_BLOCK_SCALES; ``FP8BlockScalesLinearMethod.load_weights_vanilla``
    # squeezes ModelOpt's two extra singleton scale dimensions.
    "fp8_pb_wo": "FP8_BLOCK_SCALES",
}


def canonicalize_quant_algo(value: Any) -> Any:
    """Map a ModelOpt ``quant_algo`` spelling onto its ``QuantAlgo`` name.

    Case-insensitive. Unknown values pass through untouched so that ``QuantAlgo``
    keeps ownership of rejecting genuinely invalid names.
    """
    if not isinstance(value, str):
        return value
    return _QUANT_ALGO_ALIASES.get(value.lower(), value)


def _canonicalize_algos(result: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize ``quant_algo`` at the top level and inside ``quantized_layers``.

    MIXED_PRECISION checkpoints carry the real algo names per layer, so the
    top-level key alone is not enough. Nested dicts are rebuilt rather than
    mutated in place: ``result`` is only a shallow copy of the caller's config.
    """
    if "quant_algo" in result:
        result["quant_algo"] = canonicalize_quant_algo(result["quant_algo"])
    layers = result.get("quantized_layers")
    if isinstance(layers, dict):
        result["quantized_layers"] = {
            name: (
                {**cfg, "quant_algo": canonicalize_quant_algo(cfg["quant_algo"])}
                if isinstance(cfg, dict) and "quant_algo" in cfg
                else cfg
            )
            for name, cfg in layers.items()
        }
    return result


def _kv_cache_scheme_to_algo(scheme: Any) -> Optional[str]:
    """Translate modelopt 1.x ``kv_cache_scheme`` to a legacy algo name.

    Returns ``None`` (and warns) for any unrecognized shape so a silently
    dropped KV-cache quant setting surfaces in the logs.  A missing scheme
    (``None``) is the documented "no KV-cache quant" case and does not warn.
    """
    if scheme is None:
        return None
    if isinstance(scheme, str):
        algo = scheme.upper()
        if algo in _KV_SCHEME_STRING_ALGOS:
            return algo
    elif isinstance(scheme, dict):
        mapped = _KV_SCHEME_DICT_MAP.get((scheme.get("type"), scheme.get("num_bits")))
        if mapped is not None:
            return mapped
    logger.warning(f"Unrecognized 'kv_cache_scheme' {scheme!r}; KV-cache quant disabled.")
    return None


def read_modelopt_quant_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize either modelopt shape into the legacy ``quantization`` inner dict.

    Raises ``ValueError`` if ``raw`` is not a recognized modelopt config.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict for modelopt quant config, got {type(raw).__name__}")
    if "quantization" in raw:
        # Legacy shape; the inner value must be a dict.
        q = raw["quantization"]
        if not isinstance(q, dict):
            raise ValueError(f"'quantization' must be a dict, got {type(q).__name__}")
        result = dict(q)
    elif is_modelopt_quant_config(raw):
        # Flat → legacy: rename `ignore` → `exclude_modules`, `kv_cache_scheme`
        # → `kv_cache_quant_algo`, drop modelopt-1.x-only metadata.
        _SKIP = {"producer", "quant_method", "ignore", "kv_cache_scheme", "config_groups"}
        result = {k: v for k, v in raw.items() if k not in _SKIP}
        if "ignore" in raw:
            result["exclude_modules"] = raw["ignore"]
        if "kv_cache_scheme" in raw:
            algo = _kv_cache_scheme_to_algo(raw["kv_cache_scheme"])
            if algo is not None:
                result["kv_cache_quant_algo"] = algo
    else:
        raise ValueError(
            f"Not a modelopt quant config (producer={raw.get('producer')!r}, "
            f"quant_method={raw.get('quant_method')!r})"
        )
    return _canonicalize_algos(result)


def warn_if_inline_diverges(
    file_quant: Dict[str, Any], inline_raw: Any, *, source_file: str
) -> None:
    """Warn if ``config.json.quantization_config`` diverges from the file-based config.

    The file remains authoritative; this only logs.  Skips if the inline is
    absent or not recognizably modelopt.
    """
    if not inline_raw or not is_modelopt_quant_config(inline_raw):
        return
    try:
        inline_quant = read_modelopt_quant_config(inline_raw)
    except ValueError as e:
        logger.warning(
            f"Inline modelopt config failed to parse ({e}); "
            f"skipping divergence check against '{source_file}'."
        )
        return
    diffs = []
    for key in ("quant_algo", "kv_cache_quant_algo", "group_size"):
        if file_quant.get(key) != inline_quant.get(key):
            diffs.append(f"{key}: file={file_quant.get(key)!r}, inline={inline_quant.get(key)!r}")
    f_excl = set(file_quant.get("exclude_modules") or [])
    i_excl = set(inline_quant.get("exclude_modules") or [])
    if f_excl != i_excl:
        diffs.append(f"exclude_modules: |file|={len(f_excl)}, |inline|={len(i_excl)}")
    f_ql = len(file_quant.get("quantized_layers") or {})
    i_ql = len(inline_quant.get("quantized_layers") or {})
    if f_ql != i_ql:
        diffs.append(f"quantized_layers count: file={f_ql}, inline={i_ql}")
    if diffs:
        logger.warning(
            f"Inline 'config.json.quantization_config' diverges from "
            f"'{source_file}':\n  - " + "\n  - ".join(diffs)
        )
