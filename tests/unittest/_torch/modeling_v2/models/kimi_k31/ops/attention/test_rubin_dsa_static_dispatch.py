# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family DSA runtime helpers: two backend classes and the metadata classes they carry.

The helpers define no indexer class and no ``nn.Module``: the flat targets run their MLA layers in
``modeling.py``. Each backend class carries its metadata class only; a target picks one by its
metadata settings, and the engine sees a TRTLLM DSA backend and a DSA metadata class (warmup
dispatch, the KV-length refresh rule).
"""

import inspect
import types

import pytest
import torch.nn as nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention import (
    rubin_dsa_backend,
    rubin_dsa_metadata,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.backend import (
    DSATrtllmAttention as EngineBackend,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa.metadata import (
    DSAtrtllmAttentionMetadata as EngineMetadata,
)
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

pytestmark = pytest.mark.cpu_only

BACKENDS = (rubin_dsa_backend.RubinDSABackend, rubin_dsa_backend.RubinDSAHelixBackend)


@pytest.mark.parametrize("module", [rubin_dsa_metadata, rubin_dsa_backend], ids=["meta", "backend"])
def test_no_indexer_class_and_no_module_in_the_helpers(module):
    classes = [
        value
        for value in vars(module).values()
        if inspect.isclass(value) and value.__module__ == module.__name__
    ]
    assert classes
    assert not [c.__name__ for c in classes if "Indexer" in c.__name__]
    assert not [c.__name__ for c in classes if issubclass(c, nn.Module)]


@pytest.mark.parametrize("backend", BACKENDS, ids=lambda c: c.__name__)
def test_a_backend_class_carries_its_metadata_class_only(backend):
    own = set(vars(backend)) - {
        "__module__",
        "__qualname__",
        "__doc__",
        "__firstlineno__",
        "__parameters__",
        "__static_attributes__",
    }
    assert own == {"Metadata"}
    assert issubclass(backend, EngineBackend) and issubclass(backend, TrtllmAttention)
    assert issubclass(backend.Metadata, EngineMetadata)
    assert backend.Metadata.__module__ == rubin_dsa_metadata.__name__


def test_dep16_and_tep8_pick_the_plain_backend():
    cls = rubin_dsa_backend.rubin_dsa_backend_cls(
        kv_lens_refresh_once=False, helix_fused_meta=False
    )
    assert cls is rubin_dsa_backend.RubinDSABackend
    # The engine refreshes the KV lengths after every correction for this metadata.
    assert not getattr(cls.Metadata, "kv_lens_refresh_once", False)
    helix = rubin_dsa_backend.rubin_dsa_backend_cls(
        kv_lens_refresh_once=True, helix_fused_meta=True
    )
    assert helix is rubin_dsa_backend.RubinDSAHelixBackend
    assert helix.Metadata is rubin_dsa_metadata.RubinDSAHelixMetadata
    for args in ((True, False), (False, True)):
        with pytest.raises(ValueError, match="no Kimi K3.1 Rubin DSA backend"):
            rubin_dsa_backend.rubin_dsa_backend_cls(
                kv_lens_refresh_once=args[0], helix_fused_meta=args[1]
            )


def test_attention_backend_for_serves_the_trtllm_dsa_backend_only():
    dsa = types.SimpleNamespace(algorithm="dsa")
    backend = rubin_dsa_backend.RubinDSABackend
    assert rubin_dsa_backend.attention_backend_for("TRTLLM", dsa, backend) is backend
    with pytest.raises(ValueError, match="TRTLLM DSA attention backend only"):
        rubin_dsa_backend.attention_backend_for("FLASHINFER", dsa, backend)
    with pytest.raises(ValueError, match="TRTLLM DSA attention backend only"):
        rubin_dsa_backend.attention_backend_for("TRTLLM", None, backend)
