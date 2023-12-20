import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Union

import pytest


@contextmanager
def prepend_to_sys_path(path: Union[str, os.PathLike]) -> None:
    sys.path = [str(path)] + sys.path
    try:
        yield
    finally:
        sys.path = sys.path[1:]


with prepend_to_sys_path(Path(__file__).parent / '../../../../../examples/gpt'):
    from utils.nemo import nemo_config_to_ini_config


class TestRotaryParametersSetting:
    nemo_configs = {
        "learned_absolute": {
            "position_embedding_type": "learned_absolute",
            "rotary_percentage": 0.0,
            "seq_len_interpolation_factor": 4.0,
            "rotary_base": 10000,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "relative": {
            "position_embedding_type": "relative",
            "rotary_percentage": 0.0,
            "seq_len_interpolation_factor": None,
            "rotary_base": 10000,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "nemo_rotary_pct_default_is_used": {
            "position_embedding_type": "rope",
            "seq_len_interpolation_factor": None,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "nemo_rotary_base_default_is_used": {
            "position_embedding_type": "rope",
            "seq_len_interpolation_factor": None,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "scaling_is_set": {
            "position_embedding_type": "rope",
            "seq_len_interpolation_factor": 3.5,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "wrong_seq_len_interpolation_factor_value": {
            "position_embedding_type": "rope",
            "seq_len_interpolation_factor": 1.0,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "no_scaling": {
            "position_embedding_type": "rope",
            "seq_len_interpolation_factor": None,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "rotary_base": {
            "position_embedding_type": "rope",
            "rotary_base": 9999,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "rotary_percentage_equals_0": {
            "position_embedding_type": "rope",
            "rotary_base": 9999,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
            "rotary_percentage": 0.0,
        },
        "rotary_percentage_gt_1": {
            "position_embedding_type": "rope",
            "rotary_base": 9999,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
            "rotary_percentage": 1.1,
        },
        "good_rotary_percentage": {
            "position_embedding_type": "rope",
            "rotary_base": 9999,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
            "rotary_percentage": 0.4,
        },
    }

    @pytest.mark.parametrize("nemo_config_name",
                             ["learned_absolute", "relative"])
    def test_no_rope(self, nemo_config_name):
        nemo_config = self.nemo_configs[nemo_config_name]
        vocab_size = 103
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101,
                                               vocab_size, "float32")
        assert float(ini_config["gpt"]["rotary_pct"]) == 0.0
        assert "rotary_scaling" not in ini_config["gpt"]
        assert "rotary_base" not in ini_config["gpt"]
        assert "n_head" in ini_config["gpt"]
        assert "n_positions" in ini_config["gpt"]
        assert int(ini_config["gpt"]["vocab_size"]) == vocab_size

    def test_nemo_rotary_pct_default_is_used(self):
        nemo_config = self.nemo_configs["nemo_rotary_pct_default_is_used"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        assert float(ini_config["gpt"]["rotary_pct"]) == 1.0

    def test_rotary_base_default(self):
        nemo_config = self.nemo_configs["nemo_rotary_base_default_is_used"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        assert int(ini_config["gpt"]["rotary_base"]) == 10000

    def test_scaling_is_set(self):
        nemo_config = self.nemo_configs["scaling_is_set"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        assert float(ini_config["gpt"]["rotary_scaling_factor"]
                     ) == nemo_config["seq_len_interpolation_factor"]
        assert ini_config["gpt"]["rotary_scaling_type"] == "linear"

    def test_wrong_seq_len_interpolation_factor_value(self):
        nemo_config = self.nemo_configs[
            "wrong_seq_len_interpolation_factor_value"]
        with pytest.raises(ValueError):
            nemo_config_to_ini_config(nemo_config, 100, 101, 103, "float32")

    def test_no_scaling(self):
        nemo_config = self.nemo_configs["no_scaling"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        assert "rotary_scaling" not in ini_config["gpt"]

    def test_rotary_base(self):
        nemo_config = self.nemo_configs["rotary_base"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        assert int(
            ini_config["gpt"]["rotary_base"]) == nemo_config["rotary_base"]

    def test_rotary_percentage_equals_0(self):
        nemo_config = self.nemo_configs["rotary_percentage_equals_0"]
        with pytest.raises(ValueError):
            nemo_config_to_ini_config(nemo_config, 100, 101, 103, "float32")

    def test_rotary_percentage_gt_1(self):
        nemo_config = self.nemo_configs["rotary_percentage_gt_1"]
        with pytest.raises(ValueError):
            nemo_config_to_ini_config(nemo_config, 100, 101, 103, "float32")

    def test_good_rotary_percentage(self):
        nemo_config = self.nemo_configs["good_rotary_percentage"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        assert float(
            ini_config["gpt"]["rotary_pct"]) == nemo_config["rotary_percentage"]
