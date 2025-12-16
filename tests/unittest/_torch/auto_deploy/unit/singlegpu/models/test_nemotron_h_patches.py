import functools
import types

import pytest
import torch
from _model_test_utils import _hf_model_dir_or_hub_id
from transformers import AutoConfig

from tensorrt_llm._torch.auto_deploy.models.modeling_nemotron_h import NemotronHForCausalLM
from tensorrt_llm._torch.auto_deploy.models.patches.nemotron_h import (
    _from_config_original,
    _nemotron_h_moe_forward,
)

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


def skip_on_no_hf_access(func):
    """Decorator for skipping tests that fail due to HF access issues.

    This allows us to share the same test code for CI (where access may be restricted, especially for private
    repositories) and locally.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except OSError as e:
            if "not a valid model identifier" in str(e):
                pytest.skip("Test skipped due to (no) HF access.")
            raise

    return wrapper


def _load_nemotron_moe_layer(model_name_or_path: str, custom_model_cls=None):
    """
    Build a tiny NemotronH model (1 layer, small dims) and return the first NemotronHMOE module.
    """
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    cfg.use_cache = False

    cfg.torch_dtype = "bfloat16"
    cfg.hidden_size = 32
    cfg.intermediate_size = 64
    cfg.moe_intermediate_size = 64
    cfg.moe_shared_expert_intermediate_size = 64
    cfg.mamba_head_dim = 40
    cfg.mamba_num_heads = 4
    cfg.n_groups = 2
    cfg.num_attention_heads = 4
    cfg.num_hidden_layers = 9
    cfg.num_key_value_heads = 2
    cfg.ssm_state_size = 32

    if custom_model_cls is None:
        model = _from_config_original(cfg, trust_remote_code=True)
    else:
        model = custom_model_cls._from_config(cfg)
    model.eval()

    nemotron_moe = None
    for _, mod in model.named_modules():
        if type(mod).__name__ == "NemotronHMOE":
            nemotron_moe = mod
            break

    if nemotron_moe is None:
        raise RuntimeError("NemotronHMOE layer not found. Check your model id or config.")

    _set_gate_weights(nemotron_moe)

    return nemotron_moe


def _set_gate_weights(module):
    # This helper function is necessary because the `weight` parameter of the `NemotronHTopkRouter`
    # is initialized as `torch.empty` in the original model code, which no manner of random seed
    # setting will have any effect on. We therefore set it like the below to ensure the
    # reproducibility of the tests.
    for _, mod in module.named_modules():
        if type(mod).__name__ == "NemotronHTopkRouter":
            if hasattr(mod, "weight"):
                mod.weight = torch.nn.Parameter(torch.randn_like(mod.weight))


@pytest.mark.parametrize(
    "model_name",
    [
        _hf_model_dir_or_hub_id(
            "NVIDIA-Nemotron-Nano-31B-A3-v3", "nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3"
        ),
    ],
)
@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
@skip_on_no_hf_access
def test_nemotronh_moe_patch_forward(model_name, B, S, dtype):
    device = "cuda"

    module = _load_nemotron_moe_layer(model_name)
    module.to(device)

    H = module.config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    ref = module(x)

    module.forward = types.MethodType(_nemotron_h_moe_forward, module)
    test = module(x)

    rtol = 0.05
    atol = 0.05

    torch.testing.assert_close(test, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "model_name",
    [
        _hf_model_dir_or_hub_id(
            "NVIDIA-Nemotron-Nano-31B-A3-v3", "nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3"
        ),
    ],
)
@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
@skip_on_no_hf_access
def test_nemotronh_moe_custom_implementation(model_name, B, S, dtype):
    device = "cuda"

    module = _load_nemotron_moe_layer(model_name)
    module.to(device)

    H = module.config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    ref = module(x)

    new_module = _load_nemotron_moe_layer(model_name, custom_model_cls=NemotronHForCausalLM)
    new_module.to(device)
    new_module.load_state_dict(module.state_dict())

    test = new_module(x)

    rtol = 0.05
    atol = 0.05

    torch.testing.assert_close(test, ref, rtol=rtol, atol=atol)
