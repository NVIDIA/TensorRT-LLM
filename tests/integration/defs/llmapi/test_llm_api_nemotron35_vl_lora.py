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
"""Multi-LoRA on Nemotron 3.5 (Super) VL, with images live in the same batch.

The model is `NemotronH_Omni_Reasoning_V3`, served by `NemotronH_Nano_VL_V2`
over a NemotronH backbone: 40 Mamba / 40 latent-MoE / 8 attention layers. LoRA
reaches attention q/k/v/o, both Mamba projections, the shared expert, and the
MoE latent projections. Routed experts take no LoRA, by the target list at
`NemotronHForCausalLM.lora_config()`.

One thing makes a naive version of this test worthless, and it is asserted
against here: an adapter whose keys the loader does not recognize is warned
about and skipped, so it loads clean and does nothing. A run where "LoRA
changed the output" is never checked cannot tell that apart from success.
`assert_adapter_keys_parse` runs on every adapter this test builds, and
`test_nemotron35_vl_lora_adapter.py` pins the same property on CPU.

Adapters are fabricated: no Nemotron VL LoRA checkpoint is published. Base
weights may themselves be random (the SourceOfTruth checkpoint ships a
`random_weights_manifest.json`), which is fine — every assertion here is about
adapters diverging from each other and from base, never about answer quality.
"""

import os
import sys
import tempfile

import pytest

from tensorrt_llm import LLM
from tensorrt_llm._torch.models.modeling_nemotron_nano import NemotronH_Nano_VL_V2
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.inputs import default_multimodal_input_loader
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

from ..conftest import llm_models_root

# The adapter builder lives beside its own CPU tests in the unittest tree; those
# checks run without a GPU and are the gate this file depends on. Importing it
# here keeps one copy rather than letting the two drift.
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "unittest",
            "_torch",
            "modules",
            "tests_lora_modules",
        )
    ),
)

pytestmark = [pytest.mark.threadleak(enabled=False)]

# Adapter ranks. Deliberately unequal: a single-rank set would not exercise the
# per-slot rank handling, and max_lora_rank must cover the largest.
_RANKS = (8, 16, 32)
_MAX_RANK = max(_RANKS)

# Defaults from the known-good serving configuration for this checkpoint.
# Defaults alone do not start it: 40 Mamba layers size their state per sequence,
# so a large max_batch_size consumes the pool and attention gets no blocks.
_TP_SIZE = 2
_MAX_SEQ_LEN = 16384
_MAX_BATCH_SIZE = 8
_KV_CACHE_CONFIG = KvCacheConfig(
    free_gpu_memory_fraction=0.6,
    # The backbone is mostly Mamba, whose recurrent state cannot be replayed
    # from a block hash.
    enable_block_reuse=False,
)


def _model_dir() -> str:
    """Checkpoint under test.

    The NVFP4 build is the default: it fits one B200 where the 232 GiB bf16
    SourceOfTruth needs two or more. Override to point at another.
    """
    override = os.environ.get("NEMOTRON35_VL_MODEL_DIR")
    if override:
        return override
    return f"{llm_models_root()}/nemotron_3.5_super_nvfp4_experts_no_fp8_kv"


def _media_dir() -> str:
    override = os.environ.get("NEMOTRON35_VL_MEDIA_DIR")
    if override:
        return override
    return f"{llm_models_root()}/multimodals/test_data"


@pytest.fixture(scope="module")
def model_dir() -> str:
    """Resolve the checkpoint, distinguishing "not configured" from "wrong path".

    An explicit NEMOTRON35_VL_MODEL_DIR that does not exist is a configuration
    error and fails; an absent default only means this checkpoint has not been
    staged into LLM_MODELS_ROOT yet, which skips.
    """
    override = os.environ.get("NEMOTRON35_VL_MODEL_DIR")
    path = _model_dir()
    if os.path.isdir(path):
        return path
    if override:
        raise FileNotFoundError(
            f"NEMOTRON35_VL_MODEL_DIR is set to {override!r}, which is not a directory"
        )
    pytest.skip(f"checkpoint not staged: {path}")


# Adapter weight scale. 0.02 -- the scale the Qwen3-MoE routed-expert test uses
# -- was measured on this model to leave greedy output byte-identical to base,
# which reads exactly like "LoRA was never applied" and is why an earlier run of
# this test was misdiagnosed. 0.2 moves every token of a 16-token completion on
# both the text and the image path. The assertions below only distinguish
# adapters from each other and from base, so a large scale costs nothing.
_ADAPTER_STD = 0.2


def _build_adapters(tmpdir: str, model_dir: str) -> list:
    """One adapter per rank, each with its own seed so outputs must differ."""
    from nemotron35_vl_lora_utils import assert_adapter_keys_parse, create_nemotron35_lora_adapter

    paths = []
    for seed, rank in enumerate(_RANKS):
        path = create_nemotron35_lora_adapter(
            os.path.join(tmpdir, f"lora-r{rank}"),
            model_dir,
            lora_rank=rank,
            seed=seed,
            std=_ADAPTER_STD,
        )
        # Cheap, and it converts a silent no-op into a failure at the point the
        # adapter is written rather than at the point its effect is missing.
        assert_adapter_keys_parse(path)
        paths.append(path)
    return paths


def _lora_config(lora_paths: list):
    """Target set from the model class, cache sized for every adapter at once.

    `lora_dir` must list every adapter: engine setup reads `lora_dir[0]` to
    recover the shared-expert intermediate size, and an empty list leaves the
    shared-expert modules mis-sized.
    """
    config = NemotronH_Nano_VL_V2.lora_config(_model_dir())
    config.lora_dir = list(lora_paths)
    config.max_lora_rank = _MAX_RANK
    config.max_loras = len(lora_paths)
    config.max_cpu_loras = len(lora_paths)
    return config


def _image_prompts(llm, model_dir: str, count: int) -> list:
    """`count` copies of one image prompt, so only the adapter varies."""
    media = os.path.join(_media_dir(), "seashore.png")
    if not os.path.exists(media):
        pytest.skip(f"test image not found: {media}")
    return default_multimodal_input_loader(
        tokenizer=llm.tokenizer,
        model_dir=model_dir,
        model_type="NemotronH_Nano_VL_V2",
        modality="image",
        prompts=["Describe this image in one sentence."] * count,
        media=[media] * count,
        image_data_format="pt",
        device="cpu",
    )


def _tokens(outputs) -> list:
    return [list(o.outputs[0].token_ids) for o in outputs]


@pytest.mark.skip_less_device_memory(140000)
def test_nemotron35_vl_multi_lora_with_images(model_dir):
    """Three adapters and a no-LoRA row in one batch, every row carrying an image.

    TP2 is both the functional and the sharding gate, and the sharding reaches
    LoRA directly: with `num_key_value_heads=2` a rank holds one KV head, so the
    attn_k / attn_v adapter shapes change, while the MoE latent projections are
    replicated rather than sharded.

    CUDA graphs are off. Building `CudaGraphLoraManager` against a model that
    carries a vision tower fails during worker init, before any forward pass, on
    every checkpoint in this family; that belongs to the manager rather than to
    this model and is tracked on its own. `cuda_graph_config=None` keeps this
    test on what it is here to cover, which is per-request adapter routing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        lora_paths = _build_adapters(tmpdir, model_dir)
        llm = LLM(
            model=model_dir,
            lora_config=_lora_config(lora_paths),
            tensor_parallel_size=_TP_SIZE,
            kv_cache_config=_KV_CACHE_CONFIG,
            cuda_graph_config=None,
            max_batch_size=_MAX_BATCH_SIZE,
            max_seq_len=_MAX_SEQ_LEN,
            trust_remote_code=True,
        )
        try:
            sampling_params = SamplingParams(max_tokens=24, temperature=0.0, top_k=1)

            base_inputs = _image_prompts(llm, model_dir, 1)
            base_tokens = _tokens(llm.generate(base_inputs, sampling_params, lora_request=None))[0]
            assert base_tokens, "base model produced no tokens"

            requests = [None] + [
                LoRARequest(f"nemotron35-vl-lora-{i}", i, path) for i, path in enumerate(lora_paths)
            ]
            inputs = _image_prompts(llm, model_dir, len(requests))
            out = _tokens(llm.generate(inputs, sampling_params, lora_request=requests))

            # The no-LoRA row must be untouched by its neighbours: if an adapter
            # leaked across slots this is where it shows.
            assert out[0] == base_tokens, (
                "the no-LoRA row in a mixed batch diverged from the standalone "
                "base run; an adapter leaked across slots"
            )

            for i, tokens in enumerate(out[1:]):
                assert tokens, f"adapter {i} produced no tokens"
                assert tokens != base_tokens, (
                    f"adapter {i} (rank {_RANKS[i]}) produced output identical to "
                    "base; it parsed but was never applied"
                )

            # Distinct adapters must give distinct output, or one adapter is
            # being applied to every row.
            for i in range(len(_RANKS)):
                for j in range(i + 1, len(_RANKS)):
                    assert out[1 + i] != out[1 + j], (
                        f"adapters {i} (rank {_RANKS[i]}) and {j} (rank {_RANKS[j]}) "
                        "produced identical output; per-request routing is not "
                        "selecting the right adapter"
                    )
        finally:
            llm.shutdown()
