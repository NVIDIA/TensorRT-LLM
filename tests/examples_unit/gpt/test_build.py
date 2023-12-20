import argparse
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Union


@contextmanager
def prepend_to_sys_path(path: Union[str, os.PathLike]) -> None:
    sys.path = [str(path)] + sys.path
    try:
        yield
    finally:
        sys.path = sys.path[1:]


# Using 2 separate context managers instead of 1 to avoid pre-commit isort problem.
# In "Build TRT-LLM" job pre-commit is run twice for 2 consecutive builds.
# If 1 context manager is used the first pre-commit check is passed for the first
# build but the second pre-commit check is failed with isort.
with prepend_to_sys_path(Path(__file__).parent / '../../../examples/gpt'):
    from build import override_args_from_model_dir

with prepend_to_sys_path(Path(__file__).parent / '../../../examples/gpt'):
    from utils.nemo import nemo_config_to_ini_config


class TestOverridingOfRotaryParameters:
    nemo_configs = {
        "rotary_base_overriding": {
            "position_embedding_type": "rope",
            "rotary_percentage": 1.0,
            "seq_len_interpolation_factor": 4.0,
            "rotary_base": 8888,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "rotary_scaling_overriding": {
            "position_embedding_type": "rope",
            "rotary_percentage": 1.0,
            "seq_len_interpolation_factor": 3.33333,
            "rotary_base": 10000,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "rotary_pct_overriding": {
            "position_embedding_type": "rope",
            "rotary_percentage": 0.3,
            "seq_len_interpolation_factor": 3.33333,
            "rotary_base": 10000,
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
        "no_overriding": {
            "position_embedding_type": "learned_absolute",
            "max_position_embedding": 1024,
            "num_attention_heads": 48,
        },
    }

    @staticmethod
    def create_args_with_model_dir(model_dir) -> argparse.Namespace:
        args = argparse.Namespace()
        args.model_dir = model_dir
        return args

    def test_rotary_base_overriding(self):
        nemo_config = self.nemo_configs["rotary_base_overriding"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        with tempfile.TemporaryDirectory() as model_dir:
            with open(Path(model_dir) / "config.ini", "w") as f:
                ini_config.write(f)
            args = self.create_args_with_model_dir(model_dir)
            args.rotary_base = 1111
            override_args_from_model_dir(args)
            assert args.rotary_base == nemo_config["rotary_base"]

    def test_rotary_scaling_overriding(self):
        nemo_config = self.nemo_configs["rotary_scaling_overriding"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        with tempfile.TemporaryDirectory() as model_dir:
            model_dir = Path(model_dir)
            with open(model_dir / "config.ini", "w") as f:
                ini_config.write(f)
            args = self.create_args_with_model_dir(model_dir)
            args.scaling = "Scaling?"
            override_args_from_model_dir(args)
            assert args.rotary_scaling == [
                "linear",
                str(nemo_config["seq_len_interpolation_factor"])
            ]

    def test_rotary_pct_overriding(self):
        nemo_config = self.nemo_configs["rotary_pct_overriding"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        with tempfile.TemporaryDirectory() as model_dir:
            with open(Path(model_dir) / "config.ini", "w") as f:
                ini_config.write(f)
            args = self.create_args_with_model_dir(model_dir)
            args.rotary_pct = 'foo'
            override_args_from_model_dir(args)
            assert args.rotary_pct == nemo_config["rotary_percentage"]

    def test_no_overriding(self):
        nemo_config = self.nemo_configs["no_overriding"]
        ini_config = nemo_config_to_ini_config(nemo_config, 100, 101, 103,
                                               "float32")
        with tempfile.TemporaryDirectory() as model_dir:
            with open(Path(model_dir) / "config.ini", "w") as f:
                ini_config.write(f)
            args = self.create_args_with_model_dir(model_dir)
            args.rotary_scaling = 'foo'
            args.rotary_pct = "bar"
            args.rotary_base = "baz"
            override_args_from_model_dir(args)
            assert args.rotary_scaling == "foo"
            assert args.rotary_pct == 0.0
            assert args.rotary_base == "baz"
