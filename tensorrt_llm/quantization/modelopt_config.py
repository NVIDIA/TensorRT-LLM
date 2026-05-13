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
"""Unified parser for ModelOpt-produced quantization configs.

ModelOpt emits ``hf_quant_config.json`` (and the inline
``config.json.quantization_config``) in two on-disk shapes:

1. **Legacy** (modelopt 0.43.x and earlier) -- nested under a ``quantization``
   wrapper, with ``exclude_modules`` for the deny-list and a string
   ``kv_cache_quant_algo``::

       {
           "producer": {...},
           "quantization": {
               "quant_algo": ...,
               "quantized_layers": {...},
               "exclude_modules": [...],
               "kv_cache_quant_algo": ...,
           },
       }

2. **Flat** (modelopt 1.0.x, compressed-tensors-style) -- all fields hoisted
   to the top level, with ``ignore`` for the deny-list and a dict
   ``kv_cache_scheme`` in place of ``kv_cache_quant_algo``::

       {
           "producer": {...},
           "quant_method": "modelopt",
           "quant_algo": ...,
           "quantized_layers": {...},
           "ignore": [...],
           "config_groups": {...},
           "kv_cache_scheme": {...},
       }

This module uses parallel readers: one branch per shape, both producing
a common :class:`ModelOptQuantConfig` struct.  Each caller in TensorRT-LLM
(PyTorch :class:`ModelConfig`, the LLM-API loader, the AutoDeploy
quant-config reader) consumes the struct in whatever shape it needs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..logger import logger


@dataclass
class ModelOptQuantConfig:
    """Common struct produced by either legacy or flat shape readers."""

    quant_algo: Optional[str] = None
    kv_cache_quant_algo: Optional[str] = None
    group_size: Optional[int] = None
    exclude_modules: List[str] = field(default_factory=list)
    # Per-layer overrides: {fully_qualified_name: {"quant_algo": ..., "group_size": ...}}
    quantized_layers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Which shape produced this struct; useful for logging.
    source_format: str = ""

    def to_legacy_inner_dict(self) -> Dict[str, Any]:
        """Render as the legacy ``quantization`` inner dict.

        Used by AutoDeploy whose downstream consumers operate on a dict
        with the legacy field names.
        """
        out: Dict[str, Any] = {
            "quant_algo": self.quant_algo,
            "kv_cache_quant_algo": self.kv_cache_quant_algo,
            "group_size": self.group_size,
            "exclude_modules": list(self.exclude_modules),
            "quantized_layers": dict(self.quantized_layers),
        }
        # Drop keys whose value is "absent" so downstream `.get(..., default)`
        # returns the default rather than an explicit None.
        return {k: v for k, v in out.items() if v is not None}


def _kv_cache_scheme_to_algo(scheme: Any) -> Optional[str]:
    """Translate compressed-tensors ``kv_cache_scheme`` to a string algo."""
    if not isinstance(scheme, dict):
        return None
    if scheme.get("type") == "float" and scheme.get("num_bits") == 8:
        return "FP8"
    return None


def is_modelopt_quant_config(raw: Dict[str, Any]) -> bool:
    """Return True if ``raw`` looks like a ModelOpt-produced quant config.

    Detects either shape via ``producer.name == "modelopt"`` or
    ``quant_method`` starting with ``"modelopt"``.
    """
    if not isinstance(raw, dict):
        return False
    if str(raw.get("quant_method", "")).lower().startswith("modelopt"):
        return True
    if (raw.get("producer") or {}).get("name") == "modelopt":
        return True
    return False


def _read_legacy(raw: Dict[str, Any]) -> ModelOptQuantConfig:
    """Reader for the modelopt 0.43.x nested shape."""
    q = raw.get("quantization")
    if not isinstance(q, dict):
        raise ValueError(
            f"'quantization' must be a dict for legacy ModelOpt config, got {type(q).__name__}."
        )
    return ModelOptQuantConfig(
        quant_algo=q.get("quant_algo"),
        kv_cache_quant_algo=q.get("kv_cache_quant_algo"),
        group_size=q.get("group_size"),
        exclude_modules=list(q.get("exclude_modules") or []),
        # Alias the json.load'd quantized_layers (up to ~50k entries); the raw
        # dict isn't reused after parse, so a defensive copy would be waste.
        quantized_layers=q.get("quantized_layers") or {},
        source_format="legacy",
    )


def _read_flat(raw: Dict[str, Any]) -> ModelOptQuantConfig:
    """Reader for the modelopt 1.0.x flat / compressed-tensors-style shape."""
    return ModelOptQuantConfig(
        quant_algo=raw.get("quant_algo"),
        kv_cache_quant_algo=_kv_cache_scheme_to_algo(raw.get("kv_cache_scheme")),
        group_size=raw.get("group_size"),
        exclude_modules=list(raw.get("ignore") or []),
        quantized_layers=raw.get("quantized_layers") or {},
        source_format="flat",
    )


def parse_modelopt_quant_config(raw: Dict[str, Any]) -> ModelOptQuantConfig:
    """Parse a ModelOpt quant config from either shape.

    Dispatches to :func:`_read_legacy` or :func:`_read_flat` based on the
    presence of the ``"quantization"`` wrapper.  Both branches read their
    native field names directly -- no shape-to-shape conversion occurs.

    Raises ``ValueError`` if ``raw`` is not a recognized ModelOpt config.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a dict for ModelOpt quant config, got {type(raw).__name__}")
    if "quantization" in raw:
        parsed = _read_legacy(raw)
    elif is_modelopt_quant_config(raw):
        parsed = _read_flat(raw)
    else:
        producer = (raw.get("producer") or {}).get("name")
        raise ValueError(
            f"Not a ModelOpt quant config (producer={producer!r}, "
            f"quant_method={raw.get('quant_method')!r}); "
            "missing 'quantization' wrapper and no modelopt marker found."
        )

    # Canonicalize the ``fp8_pb_wo`` alias once at the parse boundary so
    # downstream call sites don't each re-translate it.
    if parsed.quant_algo == "fp8_pb_wo":
        parsed.quant_algo = "FP8_BLOCK_SCALES"
    return parsed


# Fields whose mismatch between ``hf_quant_config.json`` and the inline
# ``config.json.quantization_config`` is worth surfacing.
_INVARIANT_FIELDS = (
    "quant_algo",
    "kv_cache_quant_algo",
    "group_size",
)


def _summarize_quantized_layers(layers: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not layers:
        return None
    histogram: Dict[str, int] = {}
    for entry in layers.values():
        algo = "UNKNOWN"
        if isinstance(entry, dict):
            algo = str(entry.get("quant_algo", "UNKNOWN")).upper()
        histogram[algo] = histogram.get(algo, 0) + 1
    return {"total": len(layers), "per_algo": histogram}


def warn_if_inline_quant_config_differs(
    file_parsed: ModelOptQuantConfig,
    inline_raw: Optional[Dict[str, Any]],
    *,
    source_file: str,
) -> None:
    """Warn if the inline config diverges from the file-based parsed config.

    Compares ``config.json.quantization_config`` to ``file_parsed`` (which
    came from ``hf_quant_config.json``).  The file form remains authoritative;
    this only logs.  If the inline is absent or not a recognizable ModelOpt
    config, no warning is emitted.
    """
    if not inline_raw or not is_modelopt_quant_config(inline_raw):
        return
    try:
        inline_parsed = parse_modelopt_quant_config(inline_raw)
    except ValueError:
        return

    diffs = []
    for key in _INVARIANT_FIELDS:
        lhs = getattr(file_parsed, key)
        rhs = getattr(inline_parsed, key)
        if lhs != rhs:
            diffs.append(f"{key}: hf_quant_config={lhs!r}, inline={rhs!r}")

    lhs_excl = set(file_parsed.exclude_modules)
    rhs_excl = set(inline_parsed.exclude_modules)
    if lhs_excl != rhs_excl:
        only_file = sorted(lhs_excl - rhs_excl)[:3]
        only_inline = sorted(rhs_excl - lhs_excl)[:3]
        diffs.append(
            f"exclude_modules: |hf_quant_config|={len(lhs_excl)}, "
            f"|inline|={len(rhs_excl)}, "
            f"only-in-hf-sample={only_file}, "
            f"only-in-inline-sample={only_inline}"
        )

    file_ql = _summarize_quantized_layers(file_parsed.quantized_layers)
    inline_ql = _summarize_quantized_layers(inline_parsed.quantized_layers)
    if file_ql != inline_ql:
        diffs.append(f"quantized_layers: hf_quant_config={file_ql}, inline={inline_ql}")

    if diffs:
        logger.warning(
            "Inline 'config.json.quantization_config' diverges from "
            f"'{source_file}'; using the latter. Diffs:\n  - " + "\n  - ".join(diffs)
        )
