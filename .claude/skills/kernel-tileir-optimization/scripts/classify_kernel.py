#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

r"""Classify a Triton kernel and optionally apply TileIR optimizations.

Standalone script -- only Python stdlib required.
Outputs structured JSON to stdout.

Usage:
    python classify_kernel.py --file kernel.py
    python classify_kernel.py --code "@triton.jit\\ndef kernel..."
    python classify_kernel.py --file kernel.py --apply-optimizations
    python classify_kernel.py --mock
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

DOT_PATTERNS = [r"tl\.dot\s*\(", r"triton\.dot\s*\("]

NORM_PATTERNS = [
    r"layernorm",
    r"layer_norm",
    r"rmsnorm",
    r"rms_norm",
    r"softmax",
    r"groupnorm",
    r"group_norm",
    r"tl\.sum\s*\([^)]*\)\s*/\s*\w+",  # mean pattern: sum / n
]

REDUCTION_PATTERNS = [
    r"tl\.sum\s*\(",
    r"tl\.max\s*\(",
    r"tl\.min\s*\(",
    r"tl\.argmax\s*\(",
    r"tl\.argmin\s*\(",
    r"tl\.atomic_add\s*\(",
    r"tl\.atomic_max\s*\(",
]

ELEMENTWISE_OPS_PATTERNS = [
    r"\bgelu\b",
    r"\brelu\b",
    r"\bsilu\b",
    r"\bsigmoid\b",
    r"\btanh\b",
    r"tl\.exp\b",
    r"tl\.log\b",
    r"tl\.sin\b",
    r"tl\.cos\b",
    r"tl\.abs\b",
    r"tl\.sqrt\b",
    r"\bdropout\b",
]

# ---------------------------------------------------------------------------
# TileIR autotune config templates
# ---------------------------------------------------------------------------

ELEMENTWISE_CONFIGS = """[
        triton.Config({'BLOCK_SIZE': 256, 'occupancy': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256, 'occupancy': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512, 'occupancy': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024, 'occupancy': 2}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024, 'occupancy': 4}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048, 'occupancy': 4}, num_warps=8, num_stages=4),
        # Extreme config for small inputs
        triton.Config({'BLOCK_SIZE': 256, 'occupancy': 16}, num_warps=2, num_stages=2),
    ]"""

NORM_CONFIGS = """[
        triton.Config({'occupancy': 1}, num_warps=4, num_stages=3),
        triton.Config({'occupancy': 1}, num_warps=8, num_stages=3),
        triton.Config({'occupancy': 2}, num_warps=4, num_stages=3),
        triton.Config({'occupancy': 2}, num_warps=8, num_stages=3),
        triton.Config({'occupancy': 4}, num_warps=4, num_stages=3),
        triton.Config({'occupancy': 4}, num_warps=8, num_stages=3),
    ]"""

DOT_CONFIGS = """[
        # Single CTA configurations
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'occupancy': 1},
            num_stages=4, num_ctas=1),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'occupancy': 2},
            num_stages=4, num_ctas=1),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'occupancy': 2},
            num_stages=4, num_ctas=1),
        # Extended num_stages
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'occupancy': 2},
            num_stages=6, num_ctas=1),
        # 2CTA configurations (critical for Blackwell)
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'occupancy': 2},
            num_stages=4, num_ctas=2),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 128, 'occupancy': 2},
            num_stages=4, num_ctas=2),
        triton.Config(
            {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'occupancy': 2},
            num_stages=6, num_ctas=2),
        # Higher occupancy
        triton.Config(
            {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'occupancy': 4},
            num_stages=4, num_ctas=1),
    ]"""

# ---------------------------------------------------------------------------
# Classification engine
# ---------------------------------------------------------------------------


def _find_matching_patterns(code_lower: str, patterns: list[str]) -> list[str]:
    """Return human-readable indicators for matching patterns."""
    indicators: list[str] = []
    for pattern in patterns:
        if re.search(pattern, code_lower):
            # Make a readable indicator from the pattern
            clean = pattern.replace(r"\s*\(", "(").replace(r"\b", "")
            clean = clean.replace("\\", "")
            indicators.append(f"{clean} found")
    return indicators


def classify_kernel(code: str) -> dict:
    """Classify a Triton kernel to determine TileIR optimization strategy.

    Args:
        code: Triton kernel source code.

    Returns:
        Classification result dict.
    """
    code_lower = code.lower()

    # Check for dot operations (highest priority)
    dot_indicators = _find_matching_patterns(code_lower, DOT_PATTERNS)
    if dot_indicators:
        return {
            "classification": "dot-related",
            "confidence": 0.95,
            "indicators": dot_indicators,
            "tileir_compatible": True,
            "recommendations": [
                "Convert tl.load/tl.store to TMA descriptor loads (MANDATORY)",
                "Add 2CTA configurations (num_ctas=2)",
                "Add occupancy tuning (1, 2, 4)",
                "Extend num_stages range (2, 4, 6)",
                "Use larger block sizes (256x256, 256x128)",
            ],
        }

    # Check for norm-like patterns
    norm_indicators = _find_matching_patterns(code_lower, NORM_PATTERNS)
    if norm_indicators:
        return {
            "classification": "norm-like",
            "confidence": 0.90,
            "indicators": norm_indicators,
            "tileir_compatible": True,
            "recommendations": [
                "Add high occupancy values (2, 4) to configs",
                "Add multiple num_warps options (4, 8)",
                "No TMA needed",
            ],
        }

    # Check for reduction patterns (but not norm)
    reduction_indicators = _find_matching_patterns(code_lower, REDUCTION_PATTERNS)
    elementwise_indicators = _find_matching_patterns(code_lower, ELEMENTWISE_OPS_PATTERNS)

    if reduction_indicators and not elementwise_indicators:
        return {
            "classification": "reduction",
            "confidence": 0.85,
            "indicators": reduction_indicators,
            "tileir_compatible": True,
            "recommendations": [
                "Add high occupancy values (2, 4) to configs",
                "Add multiple num_warps options (4, 8)",
                "No TMA needed",
            ],
        }

    # Default to element-wise
    indicators = elementwise_indicators or ["load/store pattern (default)"]
    return {
        "classification": "element-wise",
        "confidence": 0.75 if not elementwise_indicators else 0.85,
        "indicators": indicators,
        "tileir_compatible": True,
        "recommendations": [
            "Add occupancy tuning (1, 2, 4, 16)",
            "Add num_stages variation (2, 3, 4)",
            "Include extreme configurations for small inputs",
            "No TMA needed",
        ],
    }


# ---------------------------------------------------------------------------
# Optimization application
# ---------------------------------------------------------------------------


def _find_balanced_bracket(text: str, start: int) -> int:
    """Find closing bracket matching the opening one at *start*."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return -1


def apply_optimizations(code: str, classification: str) -> tuple[str, list[str]]:
    """Apply TileIR-specific autotune configs based on classification.

    Args:
        code: Triton kernel source code.
        classification: Kernel type (dot-related, norm-like, element-wise,
            reduction).

    Returns:
        Tuple of (optimized_code, changes_applied).
    """
    cl = classification.lower()
    changes: list[str] = []

    if "dot" in cl:
        new_configs = DOT_CONFIGS
        config_comment = "# TileIR-optimized configs for dot-related kernel (TMA, 2CTA, occupancy)"
        changes.append("Added TileIR dot-related configs (TMA, 2CTA, occupancy)")
        changes.append("Added num_stages=4,6 for deeper pipelining")
        changes.append("Added 2CTA config (num_ctas=2)")
    elif "norm" in cl:
        new_configs = NORM_CONFIGS
        config_comment = "# TileIR-optimized configs for norm-like kernel (high occupancy)"
        changes.append("Added high occupancy configs (1, 2, 4)")
        changes.append("Added num_warps variants (4, 8)")
    elif "reduction" in cl:
        new_configs = NORM_CONFIGS
        config_comment = "# TileIR-optimized configs for reduction kernel (high occupancy)"
        changes.append("Added high occupancy configs (1, 2, 4)")
        changes.append("Added num_warps variants (4, 8)")
    else:
        new_configs = ELEMENTWISE_CONFIGS
        config_comment = "# TileIR configs for element-wise kernel (occupancy + num_stages)"
        changes.append("Added occupancy tuning (1, 2, 4, 16)")
        changes.append("Added num_stages variation (2, 3, 4)")

    # Try to replace existing autotune configs
    configs_match = re.search(r"@triton\.autotune\s*\(\s*configs\s*=\s*\[", code, re.DOTALL)

    if configs_match:
        bracket_start = configs_match.end() - 1
        close_pos = _find_balanced_bracket(code, bracket_start)
        if close_pos == -1:
            close_pos = code.index("]", bracket_start)

        result = (
            code[: configs_match.start()]
            + f"@triton.autotune(\n    {config_comment}\n    configs={new_configs}"
            + code[close_pos + 1 :]
        )
        changes.append("Replaced existing autotune configs")
    else:
        # No autotune found -- add before @triton.jit
        jit_pattern = r"(@triton\.jit\s*\n)"
        autotune_decorator = (
            f"@triton.autotune(\n"
            f"    {config_comment}\n"
            f"    configs={new_configs},\n"
            f"    key=['n_elements'],  # Adjust key based on kernel parameters\n"
            f")\n"
            f"\\1"
        )
        result = re.sub(jit_pattern, autotune_decorator, code)
        if result != code:
            changes.append("Added @triton.autotune decorator with TileIR configs")
        else:
            # Fallback: just prepend a comment
            result = f"# {config_comment}\n{code}"
            changes.append("Added config comment (no @triton.jit found)")

    return result, changes


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _mock_data() -> dict:
    """Return realistic mock classification data."""
    return {
        "classification": "dot-related",
        "confidence": 0.95,
        "indicators": ["tl.dot( found", "block matrix multiply pattern"],
        "tileir_compatible": True,
        "recommendations": [
            "Convert tl.load/tl.store to TMA descriptor loads (MANDATORY)",
            "Add 2CTA configurations (num_ctas=2)",
            "Enable 2CTA parallelism",
            "Use TMA descriptors",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify a Triton kernel for TileIR optimization."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", help="Path to Python file containing the kernel.")
    group.add_argument("--code", help="Kernel code snippet.")
    group.add_argument("--mock", action="store_true", help="Return mock data for testing.")
    parser.add_argument(
        "--apply-optimizations",
        action="store_true",
        help="Apply TileIR optimizations and include optimized code in output.",
    )
    args = parser.parse_args()

    if args.mock:
        data = _mock_data()
        if args.apply_optimizations:
            data["optimized_code"] = "# mock optimized kernel code"
            data["changes_applied"] = ["Added 2CTA config", "Added occupancy tuning"]
    elif args.file:
        if not os.path.exists(args.file):
            print(f"File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file) as f:
            code = f.read()
        data = classify_kernel(code)
        if args.apply_optimizations:
            optimized, changes = apply_optimizations(code, data["classification"])
            data["optimized_code"] = optimized
            data["changes_applied"] = changes
    else:
        code = args.code
        data = classify_kernel(code)
        if args.apply_optimizations:
            optimized, changes = apply_optimizations(code, data["classification"])
            data["optimized_code"] = optimized
            data["changes_applied"] = changes

    json.dump(data, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
