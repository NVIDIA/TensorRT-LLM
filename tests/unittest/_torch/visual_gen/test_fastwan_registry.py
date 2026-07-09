#112 SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Registry and defaults tests for FastWanPipeline.

No GPU or checkpoint required.

Run:
    pytest tests/unittest/_torch/visual_gen/test_fastwan_registry.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.wan.defaults import get_fastwan_default_params
from tensorrt_llm._torch.visual_gen.models.wan.pipeline_fastwan import FastWanPipeline
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


FASTWAN_HF_ID = "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers"
REGISTRY_KEY = "WanDMDPipeline"


class TestFastWanRegistration:
    def test_registry_key_exists(self):
        assert REGISTRY_KEY in PIPELINE_REGISTRY

    def test_registry_maps_to_fastwan_class(self):
        assert PIPELINE_REGISTRY[REGISTRY_KEY].pipeline_cls is FastWanPipeline

    def test_hf_id_in_registry(self):
        hf_ids = PIPELINE_REGISTRY[REGISTRY_KEY].hf_ids
        assert FASTWAN_HF_ID in hf_ids


class TestFastWanDefaultParams:
    def test_num_inference_steps_is_3(self):
        assert get_fastwan_default_params()["num_inference_steps"] == 3

    def test_guidance_scale_is_cfg_free(self):
        assert get_fastwan_default_params()["guidance_scale"] == 1.0

    def test_frame_rate_is_24(self):
        assert get_fastwan_default_params()["frame_rate"] == 24.0

    def test_canonical_resolution(self):
        params = get_fastwan_default_params()
        assert params["height"] == 704
        assert params["width"] == 1280


class TestFastWanImageNotSupported:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="nvtx_range decorator needs CUDA")
    def test_image_raises_not_implemented(self):
        pipe = object.__new__(FastWanPipeline)
        with pytest.raises(NotImplementedError):
            pipe.forward(prompt="test", seed=0, image=object())
