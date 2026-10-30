# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.model_loader import _validate_encoder_attention_backend_support
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.cpu_only

_PUBLIC_BACKENDS = frozenset({"VANILLA", "TRTLLM", "FLASHINFER"})


def test_all_multimodal_encoder_backends_are_declared_and_documented() -> None:
    from tensorrt_llm._torch.models._arch_index import (
        MODEL_ARCH_TO_MODULE,
        MULTIMODAL_MODEL_TYPE_TO_MODULE,
        is_builtin_zoo_module,
    )
    from tensorrt_llm._torch.models.modeling_gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration,
    )
    from tensorrt_llm._torch.models.modeling_mllama import MllamaForConditionalGeneration
    from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
    from tensorrt_llm.inputs.registry import (
        INPUT_PROCESSOR_REGISTRY,
        MULTIMODAL_PLACEHOLDER_REGISTRY,
    )

    # Import the lazy zoo before enumerating registries, including mixin models
    # that may not register an input processor yet.
    for module in sorted(set(MODEL_ARCH_TO_MODULE.values())):
        import_module(f"tensorrt_llm._torch.models.{module}")
    encoders_by_module = {}
    for model_type in MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_model_types():
        module = MULTIMODAL_MODEL_TYPE_TO_MODULE[model_type]
        modalities = MULTIMODAL_PLACEHOLDER_REGISTRY.get_registered_modalities(model_type)
        # Aliases and variants in a provider share logical encoders. Union their
        # modalities so adding an encoder to any alias cannot escape this check.
        encoders_by_module.setdefault(module, set()).update(
            "vision" if modality in ("image", "video") else modality for modality in modalities
        )
    models = set(INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type)
    models.update(
        model for model in MODEL_CLASS_MAPPING.values() if issubclass(model, MultimodalModelMixin)
    )
    models.add(MllamaForConditionalGeneration)
    rows = ["| Model | Logical encoder | Supported backends |", "| --- | --- | --- |"]
    for model in sorted(models, key=lambda model: model.__name__):
        if not is_builtin_zoo_module(model.__module__):
            continue
        assert "supported_encoder_attention_backends" in vars(model), model.__name__
        declared = model.supported_encoder_attention_backends
        if model is Gemma4UnifiedForConditionalGeneration:
            expected = set()  # Encoder-free projections have no attention backend.
        elif model is MllamaForConditionalGeneration:
            expected = {"vision"}  # Registered stub; vision input is not implemented.
        else:
            expected = encoders_by_module[model.__module__.rsplit(".", 1)[1]]
        assert set(declared) == expected, model.__name__
        for encoder in sorted(declared) or [None]:
            backends = declared[encoder] if encoder else frozenset()
            assert isinstance(backends, frozenset), (model.__name__, encoder)
            assert backends <= _PUBLIC_BACKENDS, (model.__name__, encoder)
            status = (
                ", ".join(
                    f"`{backend}`"
                    for backend in ("VANILLA", "TRTLLM", "FLASHINFER")
                    if backend in backends
                )
                or "Not supported"
            )
            label = f"`{encoder}`" if encoder else "None (encoder-free)"
            rows.append(f"| `{model.__name__}` | {label} | {status} |")

    doc = Path(__file__).resolve().parents[4] / "docs/source/features/multi-modality.md"
    table = doc.read_text().split("<!-- encoder-attention-backends:start -->\n", 1)[1]
    table = table.split("\n<!-- encoder-attention-backends:end -->", 1)[0]
    assert table == "\n".join(rows), "Update the encoder backend support table from declarations"


@pytest.mark.parametrize("backend", sorted(_PUBLIC_BACKENDS))
def test_public_backend_names_parse_case_insensitively(backend: str) -> None:
    config = MultimodalConfig(encoder_attention_backend={"vision": backend.lower()})
    assert config.encoder_attention_backend == {"vision": backend}
    assert MultimodalConfig.model_validate_json(config.model_dump_json()) == config


@pytest.mark.parametrize(
    "backend",
    ["SDPA", "FLASH_ATTENTION_2", "eager", "unknown", "VANILLA\n", "", None, 1, True, [], {}],
)
def test_rejects_non_public_backend_names_and_non_strings(backend: object) -> None:
    with pytest.raises(ValidationError, match="encoder_attention_backend.vision"):
        MultimodalConfig(encoder_attention_backend={"vision": backend})


@pytest.mark.parametrize("value", ["VANILLA", ["VANILLA"], 1])
def test_rejects_backend_settings_without_encoder_names(value: object) -> None:
    with pytest.raises(ValidationError, match="encoder_attention_backend"):
        MultimodalConfig(encoder_attention_backend=value)


@pytest.mark.parametrize("overrides", [None, {}])
def test_omitted_encoder_backends_preserve_defaults(overrides: dict | None) -> None:
    config = ModelConfig(
        attn_backend="FLASHINFER",
        multimodal_config=MultimodalConfig(encoder_attention_backend=overrides),
    )
    assert config.get_encoder_attention_backend("vision", "TRTLLM") == "TRTLLM"
    assert config.get_encoder_attention_backend("audio", "VANILLA") == "VANILLA"
    assert config.with_encoder_attention_backend("vision") is config
    assert config.with_encoder_attention_backend("vision", "TRTLLM").attn_backend == "TRTLLM"
    assert config.with_encoder_attention_backend("audio", "VANILLA").attn_backend == "VANILLA"
    assert config.attn_backend == "FLASHINFER"


def test_encoder_override_preserves_other_encoders_decoder_and_shared_state() -> None:
    config = ModelConfig(
        attn_backend="TRTLLM",
        multimodal_config=MultimodalConfig(encoder_attention_backend={"vision": "VANILLA"}),
    )
    config.extra_attrs["shared"] = object()
    config._frozen = True
    vision = config.with_encoder_attention_backend("vision")
    assert vision is not config
    assert vision.attn_backend == "VANILLA"
    assert config.get_encoder_attention_backend("audio", "FLASHINFER") == "FLASHINFER"
    assert config.with_encoder_attention_backend("audio", "TRTLLM") is config
    assert vision.extra_attrs is config.extra_attrs
    assert vision.multimodal_config is config.multimodal_config
    assert vision._frozen
    assert config.attn_backend == "TRTLLM"


def test_explicit_default_backend_does_not_copy_config() -> None:
    config = ModelConfig(
        attn_backend="TRTLLM",
        multimodal_config=MultimodalConfig(encoder_attention_backend={"vision": "TRTLLM"}),
    )
    assert config.with_encoder_attention_backend("vision", "VANILLA") is config


class _EncoderModel:
    supported_encoder_attention_backends = {"vision": _PUBLIC_BACKENDS, "audio": frozenset()}


@pytest.mark.parametrize("backend", sorted(_PUBLIC_BACKENDS))
def test_loader_accepts_declared_public_backends(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("tensorrt_llm._torch.flashinfer_utils.IS_FLASHINFER_AVAILABLE", True)
    _validate_encoder_attention_backend_support(
        _EncoderModel, MultimodalConfig(encoder_attention_backend={"vision": backend})
    )


@pytest.mark.parametrize(
    ("model_cls", "encoder", "message"),
    [
        (_EncoderModel, "video", "video"),
        (_EncoderModel, "audio", "audio.*cannot be changed"),
        (None, "vision", "vision"),
    ],
)
def test_loader_rejects_unknown_and_fixed_encoders(
    model_cls: type | None, encoder: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_encoder_attention_backend_support(
            model_cls, MultimodalConfig(encoder_attention_backend={encoder: "VANILLA"})
        )


def test_loader_rejects_backend_outside_encoder_support() -> None:
    class VanillaEncoderModel:
        supported_encoder_attention_backends = {"vision": frozenset({"VANILLA"})}

    with pytest.raises(ValueError, match="vision.*TRTLLM"):
        _validate_encoder_attention_backend_support(
            VanillaEncoderModel,
            MultimodalConfig(encoder_attention_backend={"vision": "TRTLLM"}),
        )


def test_explicit_flashinfer_does_not_silently_fall_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("tensorrt_llm._torch.flashinfer_utils.IS_FLASHINFER_AVAILABLE", False)
    with pytest.raises(ValueError, match="FlashInfer.*unavailable"):
        _validate_encoder_attention_backend_support(
            _EncoderModel,
            MultimodalConfig(encoder_attention_backend={"vision": "FLASHINFER"}),
        )


@pytest.mark.parametrize("model_cls", [None, _EncoderModel])
@pytest.mark.parametrize("overrides", [None, {}])
def test_loader_keeps_omitted_settings_unchanged(
    model_cls: type | None, overrides: dict | None
) -> None:
    _validate_encoder_attention_backend_support(model_cls, None)
    _validate_encoder_attention_backend_support(
        model_cls, MultimodalConfig(encoder_attention_backend=overrides)
    )


@pytest.mark.parametrize(
    ("module", "model", "expected"),
    [
        (
            "gemma4mm",
            "Gemma4ForConditionalGeneration",
            {"vision": _PUBLIC_BACKENDS, "audio": frozenset()},
        ),
        ("gemma4_unified", "Gemma4UnifiedForConditionalGeneration", {}),
        (
            "nemotron_h_multimodal",
            "NemotronHMultimodalModel",
            {"vision": _PUBLIC_BACKENDS, "audio": frozenset()},
        ),
        ("glm5_next_vision", "Glm5NextVLM", {"vision": frozenset({"TRTLLM"})}),
        ("kimi_k25", "KimiK25ForConditionalGeneration", {"vision": _PUBLIC_BACKENDS}),
        ("kimi_k3_vl", "KimiK3ForConditionalGeneration", {"vision": _PUBLIC_BACKENDS}),
        ("step3p7vl", "Step3p7VLForConditionalGeneration", {"vision": _PUBLIC_BACKENDS}),
        ("minicpmv4_6", "MiniCPMV4_6Model", {"vision": _PUBLIC_BACKENDS}),
        ("mistral", "Mistral3VLM", {"vision": _PUBLIC_BACKENDS}),
        ("deepseekv41_multimodal", "DeepseekV41ForConditionalGeneration", {"vision": frozenset()}),
        ("minimaxm3", "MiniMaxM3VLForConditionalGeneration", {"vision": frozenset()}),
    ],
)
def test_model_encoder_backend_declarations(module: str, model: str, expected: dict) -> None:
    model_cls = getattr(import_module(f"tensorrt_llm._torch.models.modeling_{module}"), model)
    assert model_cls.supported_encoder_attention_backends == expected
    for encoder, backends in expected.items():
        assert backends <= _PUBLIC_BACKENDS
        if not backends:
            with pytest.raises(ValueError, match=f"{encoder}.*cannot be changed"):
                _validate_encoder_attention_backend_support(
                    model_cls, MultimodalConfig(encoder_attention_backend={encoder: "VANILLA"})
                )


@pytest.mark.parametrize(
    ("module", "model", "encoder"),
    [
        ("vila", "VilaModel", "vision"),
        ("phi4mm", "Phi4MMForCausalLM", "vision"),
        ("phi4mm", "Phi4MMForCausalLM", "audio"),
    ],
)
def test_hf_encoders_reject_backend_overrides(module: str, model: str, encoder: str) -> None:
    model_cls = getattr(import_module(f"tensorrt_llm._torch.models.modeling_{module}"), model)
    with pytest.raises(ValueError, match=f"{encoder}.*cannot be changed"):
        _validate_encoder_attention_backend_support(
            model_cls, MultimodalConfig(encoder_attention_backend={encoder: "VANILLA"})
        )


@pytest.mark.parametrize("disabled", [False, True])
def test_loader_rejects_absent_or_disabled_encoder(disabled: bool) -> None:
    config = SimpleNamespace(vision_config=object() if disabled else None)
    with pytest.raises(ValueError, match="vision.*not enabled"):
        _validate_encoder_attention_backend_support(
            _EncoderModel,
            MultimodalConfig(encoder_attention_backend={"vision": "VANILLA"}),
            pretrained_config=config,
            disable_mm_encoder=disabled,
        )


def test_loader_accepts_enabled_encoder() -> None:
    _validate_encoder_attention_backend_support(
        _EncoderModel,
        MultimodalConfig(encoder_attention_backend={"vision": "VANILLA"}),
        pretrained_config=SimpleNamespace(vision_config=object()),
    )


def test_model_config_rejects_invalid_backend_even_without_pydantic_validation() -> None:
    config = ModelConfig(
        multimodal_config=MultimodalConfig.model_construct(
            encoder_attention_backend={"vision": "SDPA"}
        )
    )
    with pytest.raises(ValueError, match="Invalid attention backend.*SDPA"):
        config.get_encoder_attention_backend("vision", "TRTLLM")


@pytest.mark.parametrize("override", [None, "VANILLA", "TRTLLM", "FLASHINFER"])
def test_gemma4_preserves_default_towers_and_changes_only_vision(override: str | None) -> None:
    from transformers import Gemma4Config

    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4ForConditionalGeneration

    config = ModelConfig(
        pretrained_config=Gemma4Config(text_config={}, vision_config={}, audio_config={}),
        attn_backend="FLASHINFER",
        multimodal_config=MultimodalConfig(
            encoder_attention_backend={"vision": override} if override else None
        ),
    )
    text = Gemma4ForConditionalGeneration.get_sub_model_config(config, "text_config")
    vision = Gemma4ForConditionalGeneration.get_sub_model_config(config, "vision_config")
    audio = Gemma4ForConditionalGeneration.get_sub_model_config(config, "audio_config")
    assert text.attn_backend == "FLASHINFER"
    assert vision.attn_backend == (override or "TRTLLM")
    assert audio.attn_backend == "TRTLLM"
    assert text.extra_attrs is vision.extra_attrs is audio.extra_attrs is config.extra_attrs
