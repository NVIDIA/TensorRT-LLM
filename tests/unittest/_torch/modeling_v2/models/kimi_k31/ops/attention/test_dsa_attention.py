# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3.1 DSA backend declarations: the metadata class for ``attention_backend_for`` and the capability
flags, and nothing else."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_attention import (
    K31DSABackend,
    K31PerfDSABackend,
    K31PerfDSAtrtllmAttentionMetadata,
    attention_backend_for,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_metadata import (
    K31DSAtrtllmAttentionMetadata,
)
from tensorrt_llm._torch.attention.backends.interface import AttentionBackend

pytestmark = pytest.mark.cpu_only

#: Everything a backend declaration may define: the metadata class and the capability flags.
_DECLARATIONS = {"Metadata", "support_fused_rope", "support_mla", "support_fp4_kv_cache"}


@pytest.mark.parametrize("backend", [K31DSABackend, K31PerfDSABackend])
def test_backend_is_a_declaration_only(backend):
    """No forward, no constructor, no state: only the metadata class and the flags on top of the interface."""
    for cls in backend.__mro__[: backend.__mro__.index(AttentionBackend)]:
        defined = {
            name for name in vars(cls) if not (name.startswith("__") and name.endswith("__"))
        }
        assert defined <= _DECLARATIONS, f"{cls.__name__} defines {sorted(defined - _DECLARATIONS)}"
        assert "__init__" not in vars(cls)
    assert backend.forward is AttentionBackend.forward
    assert backend.__init__ is AttentionBackend.__init__
    for flag in _DECLARATIONS - {"Metadata"}:
        assert isinstance(getattr(backend, flag)(), bool)


def test_backend_flags_and_metadata():
    assert K31DSABackend.support_fused_rope()
    assert K31DSABackend.support_mla()
    assert not K31DSABackend.support_fp4_kv_cache()
    assert K31DSABackend.Metadata is K31DSAtrtllmAttentionMetadata
    assert K31PerfDSABackend.Metadata is K31PerfDSAtrtllmAttentionMetadata
    assert issubclass(K31PerfDSAtrtllmAttentionMetadata, K31DSAtrtllmAttentionMetadata)


def test_attention_backend_for():
    dsa = SimpleNamespace(algorithm="dsa")
    assert attention_backend_for("TRTLLM", dsa) is K31DSABackend
    assert attention_backend_for("trtllm", dsa, perf=True) is K31PerfDSABackend
    assert attention_backend_for("FLASHINFER", dsa) is None
    assert attention_backend_for("TRTLLM", SimpleNamespace(algorithm="rocket")) is None
    assert attention_backend_for("TRTLLM", None) is None
