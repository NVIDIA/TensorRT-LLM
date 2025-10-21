import types

import pytest
import torch
from _model_test_utils import _hf_model_dir_or_hub_id
from transformers import AutoConfig

from tensorrt_llm._torch.auto_deploy.models.patches.nemotron_h import (
    _from_config_original,
    _nemotron_h_moe_forward,
)

torch.manual_seed(42)


def _load_nemotron_moe_layer(model_name_or_path: str):
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

    model = _from_config_original(cfg, trust_remote_code=True)
    model.eval()

    nemotron_moe = None
    for name, mod in model.named_modules():
        if type(mod).__name__ == "NemotronHMOE":
            nemotron_moe = mod
            break

    if nemotron_moe is None:
        raise RuntimeError("NemotronHMOE layer not found. Check your model id or config.")

    return nemotron_moe


@pytest.mark.parametrize(
    "model_name",
    [
        _hf_model_dir_or_hub_id(
            "NVIDIA-Nemotron-Nano-31B-A3-v3", "nvidia/NVIDIA-Nemotron-Nano-31B-A3-v3"
        ),
    ],
)
@pytest.mark.parametrize("B,S", [(2, 6), (1, 8)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_nemotronh_moe_patch_forward(model_name, B, S, dtype):
    pytest.skip("Skipping due to permission issue")
    device = "cuda"

    module = _load_nemotron_moe_layer(model_name)
    module.to(device)

    H = module.config.hidden_size
    x = torch.randn(B, S, H, device=device, dtype=dtype)

    with torch.no_grad():
        ref = module(x)

    module.forward = types.MethodType(_nemotron_h_moe_forward, module)
    with torch.no_grad():
        test = module(x)

    rtol = 0.05
    atol = 0.05

    torch.testing.assert_close(test, ref, rtol=rtol, atol=atol)
