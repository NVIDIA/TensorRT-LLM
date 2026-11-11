# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared checks of the attention metadata extension contracts."""

from typing import Any, Dict, Optional

import torch

# Every production metadata extension and the test module that checks its
# creation configuration with ``check_copy_reproduces_creation_config``; None
# for an extension that takes no creation arguments.
CREATION_CONFIG_COVERAGE: Dict[str, Optional[str]] = {
    "AttentionCoreMetadata": "attention/test_metadata_creation_config.py",
    "VanillaCoreMetadata": "attention/test_metadata_creation_config.py",
    "TrtllmCoreMetadata": "attention/test_metadata_creation_config.py",
    "DraftMetadata": "attention/test_metadata_creation_config.py",
    "SpecDecodingMetadata": "attention/test_metadata_creation_config.py",
    "HelixMetadata": "attention/test_metadata_creation_config.py",
    "FlashMlaMetadata": "attention/test_metadata_creation_config.py",
    "MTPIndexShareExtension": "attention/test_metadata_creation_config.py",
    "Qwen4ExpPleState": "attention/test_metadata_creation_config.py",
    "CrossAttentionMetadata": "attention/test_metadata_creation_config.py",
    "RecurrentMetadata": "attention/test_metadata_creation_config.py",
    "FlashInferCoreMetadata": "attention/test_metadata_creation_config.py",
    "Fp4MlaState": None,
    "DSAMetadata": "attention/test_draft_metadata_reuse.py",
    "GlmTp8DSAMetadata": "attention/test_draft_metadata_reuse.py",
    "DeepseekV4Metadata": "attention/test_draft_metadata_reuse.py",
    "CSA2Metadata": "attention/test_draft_metadata_reuse.py",
    "QSAMetadata": "attention/sparse/qsa/test_qsa_cache_manager_layout.py",
    "RocketMetadata": "attention/sparse/rocketkv/test_rocketkv.py",
    "MiniMaxM3MsaMetadata": "attention/test_metadata_creation_config.py",
    "MiniMaxM3TritonMetadata": "attention/test_metadata_creation_config.py",
}


def _same(expected: Any, actual: Any) -> bool:
    if isinstance(expected, torch.Tensor) or isinstance(actual, torch.Tensor):
        if expected is actual:
            return True
        return (
            isinstance(expected, torch.Tensor)
            and isinstance(actual, torch.Tensor)
            and expected.shape == actual.shape
            and expected.dtype == actual.dtype
            and torch.equal(expected.cpu(), actual.cpu())
        )
    if expected is actual:
        return True
    try:
        return bool(expected == actual)
    except (RuntimeError, TypeError, ValueError):
        return False


def check_copy_reproduces_creation_config(metadata) -> None:
    """A copy built from ``metadata``'s configuration reports that configuration.

    For every attached extension, the copy's ``creation_config()`` equals the
    source's: each creation argument survives ``constructed_copy`` as given.
    """
    copy = metadata.constructed_copy()
    differences = {}
    for extension in metadata._attached_extensions():
        expected = extension.creation_config()
        actual = copy.ext(type(extension)).creation_config()
        names = sorted(
            name
            for name in expected.keys() | actual.keys()
            if not _same(expected.get(name), actual.get(name))
        )
        if names:
            differences[type(extension).__name__] = names
    assert differences == {}, f"creation arguments a constructed copy changed: {differences}"
