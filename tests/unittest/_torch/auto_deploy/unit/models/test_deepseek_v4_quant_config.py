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

import json
import struct
from pathlib import Path

import pytest

from tensorrt_llm._torch.auto_deploy.models.deepseek_v4_quant import (
    BF16_OR_F32,
    FINEGRAINED_FP8_LINEAR,
    INTEGER_METADATA,
    PACKED_MXFP4_EXPERT,
    SKIPPED_MTP,
    UNKNOWN,
    DeepSeekV4QuantConfigError,
    classify_deepseek_v4_checkpoint,
)
from tensorrt_llm._torch.auto_deploy.models.quant_config_reader import HFQuantConfigReader


def _deepseek_v4_config() -> dict[str, object]:
    return {
        "model_type": "deepseek_v4",
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }


def _deepseek_v4_tensor_metadata() -> dict[str, dict[str, object]]:
    return {
        "embed.weight": {"dtype": "BF16", "shape": [32000, 4096]},
        "head.weight": {"dtype": "BF16", "shape": [32000, 4096]},
        "hc_head_base": {"dtype": "F32", "shape": [2]},
        "layers.0.attn.attn_sink": {"dtype": "F32", "shape": [64]},
        "layers.0.attn.compressor.wkv.weight": {"dtype": "BF16", "shape": [1024, 4096]},
        "layers.0.attn.indexer.compressor.wkv.weight": {
            "dtype": "BF16",
            "shape": [1024, 4096],
        },
        "layers.0.attn.indexer.weights_proj.weight": {
            "dtype": "BF16",
            "shape": [128, 4096],
        },
        "layers.0.attn.indexer.wq_b.scale": {"dtype": "F8_E8M0", "shape": [64, 8]},
        "layers.0.attn.indexer.wq_b.weight": {"dtype": "F8_E4M3", "shape": [8192, 1024]},
        "layers.0.attn.kv_norm.weight": {"dtype": "BF16", "shape": [1024]},
        "layers.0.attn.q_norm.weight": {"dtype": "BF16", "shape": [1024]},
        "layers.0.attn.wq_a.scale": {"dtype": "F8_E8M0", "shape": [8, 32]},
        "layers.0.attn.wq_a.weight": {"dtype": "F8_E4M3", "shape": [1024, 4096]},
        "layers.0.attn_norm.weight": {"dtype": "BF16", "shape": [4096]},
        "layers.0.ffn.experts.0.w1.scale": {"dtype": "F8_E8M0", "shape": [2048, 128]},
        "layers.0.ffn.experts.0.w1.weight": {"dtype": "I8", "shape": [2048, 2048]},
        "layers.0.ffn.gate.bias": {"dtype": "F32", "shape": [256]},
        "layers.0.ffn.gate.tid2eid": {"dtype": "I64", "shape": [1024, 6]},
        "layers.0.ffn.gate.weight": {"dtype": "BF16", "shape": [256, 4096]},
        "layers.0.ffn.shared_experts.w1.scale": {"dtype": "F8_E8M0", "shape": [16, 32]},
        "layers.0.ffn.shared_experts.w1.weight": {"dtype": "F8_E4M3", "shape": [2048, 4096]},
        "layers.0.ffn_norm.weight": {"dtype": "BF16", "shape": [4096]},
        "layers.0.hc_attn_base": {"dtype": "F32", "shape": [2]},
        "layers.0.hc_ffn_fn": {"dtype": "F32", "shape": [2]},
        "mtp.0.attn.wq_a.weight": {"dtype": "F8_E4M3", "shape": [1024, 4096]},
        "norm.weight": {"dtype": "BF16", "shape": [4096]},
    }


def _write_deepseek_v4_checkpoint_fixture(tmp_path: Path) -> None:
    config = _deepseek_v4_config()
    tensor_metadata = _deepseek_v4_tensor_metadata()
    safetensors_name = "model-00001-of-00001.safetensors"

    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {name: safetensors_name for name in tensor_metadata},
            }
        ),
        encoding="utf-8",
    )
    _write_safetensors_header(tmp_path / safetensors_name, tensor_metadata)


def _write_safetensors_header(path: Path, tensor_metadata: dict[str, dict[str, object]]) -> None:
    header = {
        name: {
            "dtype": metadata["dtype"],
            "shape": metadata["shape"],
            "data_offsets": [0, 0],
        }
        for name, metadata in tensor_metadata.items()
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes)


def test_deepseek_v4_hf_reader_returns_normalized_quant_config(tmp_path: Path) -> None:
    _write_deepseek_v4_checkpoint_fixture(tmp_path)

    result = HFQuantConfigReader.from_file(str(tmp_path))

    assert result is not None
    reader, extra_model_kwargs = result
    assert extra_model_kwargs == {}
    qcfg = reader.get_config()
    assert qcfg["quant_method"] == "deepseek_v4_fp8"
    assert qcfg["hf_quant_method"] == "fp8"
    assert qcfg["linear_quant_method"] == "finegrained_fp8"
    assert qcfg["expert_quant_method"] == "mxfp4"
    assert qcfg["scale_fmt"] == "ue8m0"
    assert qcfg["weight_block_size"] == [128, 128]
    assert qcfg["expert_block_size"] == 32
    assert qcfg["fmt"] == "e4m3"
    assert qcfg["activation_scheme"] == "dynamic"
    assert {
        "embed",
        "head",
        "*.ffn.gate",
        "*.attn.compressor",
        "*.norm",
        "*.hc_*",
        "*.attn_sink",
        "mtp.*",
    }.issubset(set(qcfg["exclude_modules"]))


def test_deepseek_v4_classifier_groups_quantized_and_excluded_tensors() -> None:
    classification = classify_deepseek_v4_checkpoint(
        _deepseek_v4_config(), _deepseek_v4_tensor_metadata()
    )
    categories = classification["categories"]

    assert "layers.0.attn.wq_a.weight" in categories[FINEGRAINED_FP8_LINEAR]
    assert "layers.0.attn.wq_a.scale" in categories[FINEGRAINED_FP8_LINEAR]
    assert "layers.0.attn.indexer.wq_b.weight" in categories[FINEGRAINED_FP8_LINEAR]
    assert "layers.0.ffn.shared_experts.w1.weight" in categories[FINEGRAINED_FP8_LINEAR]
    assert "layers.0.ffn.experts.0.w1.weight" in categories[PACKED_MXFP4_EXPERT]
    assert "layers.0.ffn.experts.0.w1.scale" in categories[PACKED_MXFP4_EXPERT]
    assert "layers.0.ffn.gate.tid2eid" in categories[INTEGER_METADATA]
    assert "mtp.0.attn.wq_a.weight" in categories[SKIPPED_MTP]

    for key in (
        "embed.weight",
        "head.weight",
        "norm.weight",
        "hc_head_base",
        "layers.0.attn.attn_sink",
        "layers.0.attn.compressor.wkv.weight",
        "layers.0.attn.indexer.compressor.wkv.weight",
        "layers.0.ffn.gate.weight",
        "layers.0.ffn.gate.bias",
        "layers.0.hc_attn_base",
    ):
        assert key in categories[BF16_OR_F32]
        assert key not in categories[FINEGRAINED_FP8_LINEAR]
        assert key not in categories[PACKED_MXFP4_EXPERT]


def test_deepseek_v4_classifier_fails_on_unknown_keys_unless_waived() -> None:
    tensor_metadata = _deepseek_v4_tensor_metadata()
    tensor_metadata["layers.0.unexpected.weight"] = {"dtype": "BF16", "shape": [16, 16]}

    with pytest.raises(DeepSeekV4QuantConfigError, match="Unknown DeepSeek V4"):
        classify_deepseek_v4_checkpoint(_deepseek_v4_config(), tensor_metadata)

    classification = classify_deepseek_v4_checkpoint(
        _deepseek_v4_config(), tensor_metadata, allow_unknown=True
    )
    assert "layers.0.unexpected.weight" in classification["categories"][UNKNOWN]


def test_deepseek_v4_classifier_validates_scale_shapes() -> None:
    tensor_metadata = _deepseek_v4_tensor_metadata()
    tensor_metadata["layers.0.attn.wq_a.scale"] = {"dtype": "F8_E8M0", "shape": [7, 32]}

    with pytest.raises(DeepSeekV4QuantConfigError, match="expected \\[8, 32\\]"):
        classify_deepseek_v4_checkpoint(_deepseek_v4_config(), tensor_metadata)


def test_non_deepseek_fp8_config_uses_generic_hf_behavior() -> None:
    reader = HFQuantConfigReader()
    reader.read_config(
        {
            "model_type": "llama",
            "quantization_config": {
                "quant_method": "fp8",
                "exclude_modules": ["custom"],
            },
        }
    )

    qcfg = reader.get_config()
    assert qcfg["quant_method"] == "fp8"
    assert qcfg["exclude_modules"] == ["custom", "lm_head", "model.embed_tokens"]
    assert "linear_quant_method" not in qcfg
    assert "checkpoint_classification" not in qcfg
