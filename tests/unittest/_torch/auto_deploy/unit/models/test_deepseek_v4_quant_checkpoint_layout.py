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
import math
import os
import struct
from pathlib import Path

import pytest

from tensorrt_llm._torch.auto_deploy.models.custom.modeling_deepseek_v4 import (
    DeepseekV4PackedMxfp4ExpertsCheckpointLayout,
)
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.auto_deploy.models.quant_checkpoint_layout import (
    FineGrainedFP8CheckpointLayout,
    QuantCheckpointLayoutRegistry,
    QuantizedCheckpointLayout,
)
from tensorrt_llm._torch.auto_deploy.models.quant_config_reader import (
    HFQuantConfigReader,
    autodetect_quant_config_reader,
)

_DEEPSEEK_V4_FLASH_ENV_VARS = ("DEEPSEEK_V4_FLASH_MODEL_DIR", "DEEPSEEK_V4_MODEL_DIR")


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
        "layers.0.ffn.experts.0.w2.scale": {"dtype": "F8_E8M0", "shape": [4096, 64]},
        "layers.0.ffn.experts.0.w2.weight": {"dtype": "I8", "shape": [4096, 1024]},
        "layers.0.ffn.experts.0.w3.scale": {"dtype": "F8_E8M0", "shape": [2048, 128]},
        "layers.0.ffn.experts.0.w3.weight": {"dtype": "I8", "shape": [2048, 2048]},
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


def _set_fp8_scale_shapes(
    tensor_metadata: dict[str, dict[str, object]],
    weight_block_size: tuple[int, int],
) -> None:
    for weight_name in (
        "layers.0.attn.indexer.wq_b.weight",
        "layers.0.attn.wq_a.weight",
        "layers.0.ffn.shared_experts.w1.weight",
    ):
        scale_name = weight_name.removesuffix(".weight") + ".scale"
        weight_shape = tensor_metadata[weight_name]["shape"]
        assert isinstance(weight_shape, list)
        tensor_metadata[scale_name]["shape"] = [
            math.ceil(weight_shape[0] / weight_block_size[0]),
            math.ceil(weight_shape[1] / weight_block_size[1]),
        ]


def _write_deepseek_v4_checkpoint_fixture(
    tmp_path: Path,
    tensor_metadata: dict[str, dict[str, object]] | None = None,
    config: dict[str, object] | None = None,
) -> None:
    tensor_metadata = tensor_metadata or _deepseek_v4_tensor_metadata()
    config = config or _deepseek_v4_config()
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


def _write_deepseek_v4_config_only_fixture(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(_deepseek_v4_config()), encoding="utf-8")


def _write_deepseek_v4_index_without_shards_fixture(tmp_path: Path) -> None:
    safetensors_name = "model-00001-of-00001.safetensors"

    _write_deepseek_v4_config_only_fixture(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 0},
                "weight_map": {
                    "layers.0.ffn.experts.0.w1.weight": safetensors_name,
                },
            }
        ),
        encoding="utf-8",
    )


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


def _assert_deepseek_v4_checkpoint_layout(checkpoint_layout: object) -> None:
    assert isinstance(checkpoint_layout, QuantizedCheckpointLayout)
    assert isinstance(checkpoint_layout.finegrained_fp8, FineGrainedFP8CheckpointLayout)
    assert any(
        isinstance(consumer, DeepseekV4PackedMxfp4ExpertsCheckpointLayout)
        for consumer in checkpoint_layout.checkpoint_consumers
    )


def _deepseek_v4_flash_checkpoint_or_skip() -> Path:
    candidates: list[Path] = []
    for env_var in _DEEPSEEK_V4_FLASH_ENV_VARS:
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value))

    models_root = os.environ.get("LLM_MODELS_ROOT")
    if models_root:
        root = Path(models_root)
        candidates.extend(
            (
                root / "DeepSeek-V4-Flash",
                root / "DeepSeek-V4" / "DeepSeek-V4-Flash",
                root / "deepseek-ai" / "DeepSeek-V4-Flash",
            )
        )

    for candidate in candidates:
        has_safetensors = (candidate / "model.safetensors.index.json").is_file() or any(
            candidate.glob("*.safetensors")
        )
        if (candidate / "config.json").is_file() and has_safetensors:
            return candidate

    pytest.skip(
        "DeepSeek-V4-Flash checkpoint not found; set DEEPSEEK_V4_FLASH_MODEL_DIR "
        "or LLM_MODELS_ROOT to enable this metadata validation."
    )


def test_deepseek_v4_checkpoint_layout_is_registered() -> None:
    checkpoint_layout = QuantCheckpointLayoutRegistry.build_from_config(_deepseek_v4_config())

    assert checkpoint_layout is not None
    _assert_deepseek_v4_checkpoint_layout(checkpoint_layout)


def test_deepseek_v4_checkpoint_layout_uses_quant_config_layout_fields() -> None:
    config = _deepseek_v4_config()
    qconf = config["quantization_config"]
    assert isinstance(qconf, dict)
    qconf["scale_fmt"] = "UE8M0"
    qconf["weight_block_size"] = [64, 128]

    checkpoint_layout = QuantCheckpointLayoutRegistry.build_from_config(config)

    assert checkpoint_layout is not None
    _assert_deepseek_v4_checkpoint_layout(checkpoint_layout)
    assert checkpoint_layout.finegrained_fp8 is not None
    assert checkpoint_layout.finegrained_fp8.scale_fmt == "ue8m0"
    assert checkpoint_layout.finegrained_fp8.weight_block_size == (64, 128)


def test_deepseek_v4_checkpoint_layout_requires_weight_block_size() -> None:
    config = _deepseek_v4_config()
    qconf = config["quantization_config"]
    assert isinstance(qconf, dict)
    del qconf["weight_block_size"]

    with pytest.raises(ValueError, match="requires weight_block_size"):
        QuantCheckpointLayoutRegistry.build_from_config(config)


def test_deepseek_v4_checkpoint_layout_rejects_unsupported_scale_fmt() -> None:
    config = _deepseek_v4_config()
    qconf = config["quantization_config"]
    assert isinstance(qconf, dict)
    qconf["scale_fmt"] = "float32"

    with pytest.raises(ValueError, match="scale_fmt='ue8m0'"):
        QuantCheckpointLayoutRegistry.build_from_config(config)


def test_deepseek_v4_hf_reader_returns_normalized_quant_config(tmp_path: Path) -> None:
    _write_deepseek_v4_checkpoint_fixture(tmp_path)

    result = HFQuantConfigReader.from_file(str(tmp_path))

    assert result is not None
    reader, extra_model_kwargs = result
    assert extra_model_kwargs == {"ad_use_mxfp4_experts": True}
    qcfg = reader.get_config()
    assert qcfg["quant_method"] == "fp8"
    assert qcfg["scale_fmt"] == "ue8m0"
    assert qcfg["weight_block_size"] == [128, 128]
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
    _assert_deepseek_v4_checkpoint_layout(qcfg["checkpoint_layout"])


def test_deepseek_v4_hf_reader_uses_quant_config_weight_block_size_for_metadata(
    tmp_path: Path,
) -> None:
    config = _deepseek_v4_config()
    qconf = config["quantization_config"]
    assert isinstance(qconf, dict)
    qconf["weight_block_size"] = [64, 128]
    tensor_metadata = _deepseek_v4_tensor_metadata()
    _set_fp8_scale_shapes(tensor_metadata, (64, 128))
    _write_deepseek_v4_checkpoint_fixture(tmp_path, tensor_metadata, config)

    result = HFQuantConfigReader.from_file(str(tmp_path))

    assert result is not None
    reader, _ = result
    qcfg = reader.get_config()
    assert qcfg["weight_block_size"] == [64, 128]
    checkpoint_layout = qcfg["checkpoint_layout"]
    _assert_deepseek_v4_checkpoint_layout(checkpoint_layout)
    assert checkpoint_layout.finegrained_fp8 is not None
    assert checkpoint_layout.finegrained_fp8.weight_block_size == (64, 128)


def test_deepseek_v4_hf_reader_skips_config_only_prefetch_without_metadata(
    tmp_path: Path,
) -> None:
    _write_deepseek_v4_config_only_fixture(tmp_path)

    hf_result = HFQuantConfigReader.from_file(str(tmp_path), require_checkpoint_metadata=False)
    autodetect_result = autodetect_quant_config_reader(
        str(tmp_path), require_checkpoint_metadata=False
    )

    assert hf_result is None
    assert autodetect_result is None


def test_deepseek_v4_hf_reader_skips_index_without_shards_in_relaxed_mode(
    tmp_path: Path,
) -> None:
    _write_deepseek_v4_index_without_shards_fixture(tmp_path)

    hf_result = HFQuantConfigReader.from_file(str(tmp_path), require_checkpoint_metadata=False)
    autodetect_result = autodetect_quant_config_reader(
        str(tmp_path), require_checkpoint_metadata=False
    )

    assert hf_result is None
    assert autodetect_result is None


def test_deepseek_v4_hf_reader_reports_missing_index_shard_in_strict_mode(
    tmp_path: Path,
) -> None:
    _write_deepseek_v4_index_without_shards_fixture(tmp_path)

    with pytest.raises(ValueError, match="safetensors file not found"):
        HFQuantConfigReader.from_file(str(tmp_path))

    with pytest.raises(ValueError, match="safetensors file not found"):
        autodetect_quant_config_reader(str(tmp_path))


def test_deepseek_v4_hf_reader_requires_metadata_in_strict_mode(tmp_path: Path) -> None:
    _write_deepseek_v4_config_only_fixture(tmp_path)

    with pytest.raises(ValueError, match="requires safetensors metadata"):
        HFQuantConfigReader.from_file(str(tmp_path))

    with pytest.raises(ValueError, match="requires safetensors metadata"):
        autodetect_quant_config_reader(str(tmp_path))


def test_deepseek_v4_hf_reader_preserves_malformed_metadata_errors_in_relaxed_mode(
    tmp_path: Path,
) -> None:
    _write_deepseek_v4_config_only_fixture(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")

    with pytest.raises(ValueError, match="Invalid safetensors header"):
        HFQuantConfigReader.from_file(str(tmp_path), require_checkpoint_metadata=False)


def test_deepseek_v4_prefetch_checkpoint_skips_quant_reader_without_weights(
    tmp_path: Path,
) -> None:
    _write_deepseek_v4_config_only_fixture(tmp_path)
    factory = AutoModelForCausalLMFactory(model=str(tmp_path))

    fetched_dir = factory._prefetch_checkpoint(str(tmp_path), skip_prefetch_weights=True)

    assert fetched_dir == str(tmp_path)
    assert factory._quant_config_reader is None


@pytest.mark.parametrize(
    ("tensor_name", "bad_shape", "expected_shape"),
    (
        ("layers.0.attn.wq_a.scale", [7, 32], r"(?:\(8, 32\)|\[8, 32\])"),
        ("layers.0.ffn.experts.0.w1.scale", [2048, 127], r"(?:\(2048, 128\)|\[2048, 128\])"),
    ),
)
def test_deepseek_v4_layout_validates_scale_shapes(
    tmp_path: Path,
    tensor_name: str,
    bad_shape: list[int],
    expected_shape: str,
) -> None:
    tensor_metadata = _deepseek_v4_tensor_metadata()
    tensor_metadata[tensor_name] = {"dtype": "F8_E8M0", "shape": bad_shape}
    _write_deepseek_v4_checkpoint_fixture(tmp_path, tensor_metadata)

    with pytest.raises(ValueError, match=f"expected {expected_shape}"):
        HFQuantConfigReader.from_file(str(tmp_path))


def test_deepseek_v4_layout_requires_fp8_companion_scale_metadata(tmp_path: Path) -> None:
    tensor_metadata = _deepseek_v4_tensor_metadata()
    del tensor_metadata["layers.0.attn.wq_a.scale"]
    _write_deepseek_v4_checkpoint_fixture(tmp_path, tensor_metadata)

    with pytest.raises(ValueError, match=r"wq_a\.weight is missing companion scale"):
        HFQuantConfigReader.from_file(str(tmp_path))


def test_deepseek_v4_layout_skips_mtp_metadata_without_companion_scale(
    tmp_path: Path,
) -> None:
    tensor_metadata = _deepseek_v4_tensor_metadata()
    tensor_metadata["mtp.0.layers.0.attn.wq_a.weight"] = {
        "dtype": "F8_E4M3",
        "shape": [1024, 4096],
    }
    _write_deepseek_v4_checkpoint_fixture(tmp_path, tensor_metadata)

    result = HFQuantConfigReader.from_file(str(tmp_path))

    assert result is not None
    reader, _ = result
    assert "mtp.*" in reader.get_config()["exclude_modules"]


def test_deepseek_v4_layout_requires_complete_mxfp4_expert_projection_metadata(
    tmp_path: Path,
) -> None:
    tensor_metadata = _deepseek_v4_tensor_metadata()
    del tensor_metadata["layers.0.ffn.experts.0.w2.weight"]
    del tensor_metadata["layers.0.ffn.experts.0.w2.scale"]
    _write_deepseek_v4_checkpoint_fixture(tmp_path, tensor_metadata)

    with pytest.raises(ValueError, match="missing required projection w2"):
        HFQuantConfigReader.from_file(str(tmp_path))


def test_autodetect_preserves_deepseek_v4_metadata_errors(tmp_path: Path) -> None:
    tensor_metadata = _deepseek_v4_tensor_metadata()
    tensor_metadata["layers.0.attn.wq_a.scale"] = {"dtype": "F8_E8M0", "shape": [7, 32]}
    _write_deepseek_v4_checkpoint_fixture(tmp_path, tensor_metadata)

    with pytest.raises(ValueError, match=r"expected (?:\(8, 32\)|\[8, 32\])"):
        autodetect_quant_config_reader(str(tmp_path))


def test_deepseek_v4_flash_real_checkpoint_metadata_validates_when_available() -> None:
    checkpoint_dir = _deepseek_v4_flash_checkpoint_or_skip()

    result = HFQuantConfigReader.from_file(str(checkpoint_dir))

    assert result is not None
    reader, extra_model_kwargs = result
    assert extra_model_kwargs == {"ad_use_mxfp4_experts": True}
    qcfg = reader.get_config()
    _assert_deepseek_v4_checkpoint_layout(qcfg["checkpoint_layout"])
    assert qcfg["scale_fmt"] == "ue8m0"
    assert qcfg["expert_quant_method"] == "mxfp4"


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
    assert "checkpoint_layout" not in qcfg
