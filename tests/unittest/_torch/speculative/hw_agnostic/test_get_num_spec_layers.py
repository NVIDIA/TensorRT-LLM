from tensorrt_llm._torch.speculative.utils import (
    get_num_spec_layers,
    update_spec_config_from_draft_model_config,
    update_spec_config_from_loaded_model,
)
from tensorrt_llm.llmapi.llm_args import Eagle3DecodingConfig, MTPDecodingConfig


class _DraftPretrainedConfig:

    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers


def test_get_num_spec_layers_eagle3_one_model_uses_draft_hidden_layers():
    spec_config = Eagle3DecodingConfig(
        max_draft_len=8,
        speculative_model="/path/to/draft",
        num_eagle_layers=8,
    )
    spec_config._num_draft_hidden_layers = 1

    assert get_num_spec_layers(spec_config) == 1


def test_get_num_spec_layers_eagle3_one_model_defaults_to_one():
    spec_config = Eagle3DecodingConfig(
        max_draft_len=8,
        speculative_model="/path/to/draft",
        num_eagle_layers=8,
    )

    assert get_num_spec_layers(spec_config) == 1


def test_get_num_spec_layers_mtp_eagle_one_model():
    spec_config = MTPDecodingConfig(max_draft_len=1)
    assert get_num_spec_layers(spec_config) == 1


def test_update_spec_config_from_draft_model_config():
    spec_config = Eagle3DecodingConfig(
        max_draft_len=8,
        speculative_model="/path/to/draft",
    )

    update_spec_config_from_draft_model_config(spec_config,
                                               _DraftPretrainedConfig(3))

    assert spec_config._num_draft_hidden_layers == 3
    assert get_num_spec_layers(spec_config) == 3


def test_update_spec_config_from_loaded_model():
    spec_config = Eagle3DecodingConfig(
        max_draft_len=8,
        speculative_model="/path/to/draft",
    )

    class _Model:

        class _Config:

            num_hidden_layers = 32

        config = _Config()
        draft_config = type("DraftConfig", (), {
            "pretrained_config": _DraftPretrainedConfig(2),
        })()

    update_spec_config_from_loaded_model(spec_config, _Model())

    assert spec_config._num_draft_hidden_layers == 2
    assert get_num_spec_layers(spec_config) == 2
