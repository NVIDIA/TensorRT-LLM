import importlib.util
import sys
import types
from unittest import mock

import pytest
import torch
from _model_test_utils import get_small_model_config
from torch.export import Dim
from transformers import AutoConfig, AutoModelForCausalLM
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs
from tensorrt_llm._torch.auto_deploy.models.custom.modeling_nemotron_h import NemotronHForCausalLM
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

_BATCH_AND_SEQUENCE_TEST_CASES = ((2, 6), (1, 8))


@pytest.fixture(scope="function", autouse=True)
def set_seed():
    torch.manual_seed(42)


@pytest.fixture(autouse=True)
def stub_mamba_ssm_if_missing():
    """Stub `mamba_ssm` package.

    The `modeling_nemotron_h.py` code in all recent nemotron checkpoints have a hard dependency
    on `mamba_ssm.ops.triton.layernorm_gated.rmsnorm_fn`. This fixture stubs it, such that we
    at least can get past the import stage of the remote modeling code.
    """
    module = "mamba_ssm"
    submodule = f"{module}.ops.triton.layernorm_gated"

    if importlib.util.find_spec(module) is not None:
        yield
        return

    stub_mod = types.ModuleType(submodule)
    stub_mod.rmsnorm_fn = None

    with mock.patch.dict(sys.modules, {submodule: stub_mod}):
        yield


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
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
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
        llm_models_root() / "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    ],
)
@pytest.mark.parametrize("B,S", _BATCH_AND_SEQUENCE_TEST_CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.no_grad()
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


@pytest.mark.parametrize(
    "model_dir,model_on_meta_during_export",
    [
        ("nvidia/NVIDIA-Nemotron-Nano-12B-v2", True),
        ("nvidia/NVIDIA-Nemotron-Nano-12B-v2", False),
    ],
)
def test_custom_model_implementation_can_be_exported(
    model_dir: str,
    model_on_meta_during_export: bool,
):
    # NOTE: set to False if you want to locally test the full model.
    use_small_config: bool = True

    common_kwargs = {
        "world_size": 0,
        "runtime": "demollm",
        "model_factory": "AutoModelForCausalLM",
        "max_seq_len": 512,
        "transforms": {
            "insert_cached_attention": {"backend": "flashinfer"},
            "compile_model": {"backend": "torch-simple"},
        },
    }

    if use_small_config:
        llm_args = get_small_model_config(model_dir, **common_kwargs)["args"]
    else:
        llm_args = {
            "model": model_dir,
            **common_kwargs,
            "model_kwargs": {
                "dtype": "bfloat16",
            },
        }
    llm_args = LlmArgs(**llm_args)

    factory = llm_args.create_factory()
    model = factory.build_model("meta")
    tokenizer = factory.init_tokenizer()

    # 1. Export wants min batch size of 2 (to avoid specialization during export).
    # 2. Can't get `padding` / `truncation` to work without other steps so just use the prompts
    #    with the same tokenized length in order for the tokenizer not to complain when creating
    #    the tensor.
    message = [
        "Mamba is a snake with the following properties:",
        "Tiger is a cat with the following properties:",
    ]
    inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False).to("cuda")

    input_ids = inputs["input_ids"]
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
        input_ids.shape[0], 1
    )
    batch_size_dynamic = Dim.DYNAMIC
    seq_len_dynamic = Dim.DYNAMIC
    dynamic_shapes = (
        {0: batch_size_dynamic, 1: seq_len_dynamic},
        {0: batch_size_dynamic, 1: seq_len_dynamic},
    )

    def _run_torch_export_to_gm():
        return torch_export_to_gm(
            model,
            args=tuple(),
            kwargs={"input_ids": input_ids, "position_ids": position_ids},
            dynamic_shapes=dynamic_shapes,
        )

    if model_on_meta_during_export:
        gm = _run_torch_export_to_gm()
        factory.load_or_random_init(gm, device="cuda")
        move_to_device(gm, "cuda")
        factory._to_maybe_random(model, "cuda")
        # In order to ensure the `_minus_A` (non-persistent buffer) is correct, we need to run the
        # model's load state pre/post hooks by loading the state dicts after initialization.
        # NOTE: this is done under the hood by `torch_export_to_gm`, so we only need this in this
        # `if` clause.
        model.load_state_dict(gm.state_dict())
        gm.load_state_dict(model.state_dict())
    else:
        factory.load_or_random_init(model, device="cuda")
        gm = _run_torch_export_to_gm()
        move_to_device(gm, "cuda")

    # let's do a comparison of every state dict item between the model and the gm
    torch.testing.assert_close(model.state_dict(), gm.state_dict(), rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        dict(model.named_buffers()), dict(gm.named_buffers()), rtol=0.0, atol=0.0
    )

    with torch.inference_mode():
        out_original = model(input_ids=input_ids, position_ids=position_ids)
        out_gm = gm(input_ids=input_ids, position_ids=position_ids)

    atol, rtol = 1e-3, 1e-3
    torch.testing.assert_close(
        out_gm,
        out_original,
        rtol=rtol,
        atol=atol,
    )
