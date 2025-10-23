import pytest
import torch
import torch.export as te
from _model_test_utils import get_small_model_config
from torch.export import Dim

from tensorrt_llm._torch.auto_deploy.export import apply_export_patches, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.llm_args import AutoDeployConfig
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device

# NOTE: find example inputs with the same tokenization length to avoid seq concat.
EXAMPLE_INPUT = "Mamba is a snake with the following properties:"
EXAMPLE_INPUT2 = "Tiger is a cat with the following properties:"


@pytest.mark.parametrize(
    "model_dir,run_verify_generation",
    [
        pytest.param("ibm-ai-platform/Bamba-9B-v2", True),
        pytest.param("nvidia/NVIDIA-Nemotron-Nano-12B-v2", False),
    ],
)
def test_bamba_patches(model_dir: str, run_verify_generation: bool):
    # NOTE: set to False if you want to locally test the full model
    use_small_config: bool = True

    model_on_meta_during_export = True
    use_cache: bool = True
    export_func: str = "torch_export_to_gm"

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
        llm_args["model_kwargs"]["use_cache"] = use_cache
    else:
        llm_args = {
            "model": model_dir,
            **common_kwargs,
            "model_kwargs": {
                "use_cache": use_cache,
                "dtype": "bfloat16",
            },
        }
    llm_args = AutoDeployConfig(**llm_args)

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    factory = llm_args.create_factory()
    model = factory.build_model("meta")
    tokenizer = factory.init_tokenizer()

    # 1. Export wants min batch size of 2 (to avoid specialization during export).
    # 2. Can't get `padding` / `truncation` to work without other steps so just use the same prompt
    #    twice in order for the tokenizer not to complain when creating the tensor.
    message = [EXAMPLE_INPUT] * 2
    inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False).to("cuda")

    input_ids = inputs["input_ids"]
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
        input_ids.shape[0], 1
    )
    dynamic_shapes = (
        {0: Dim("batch_size", min=0, max=8), 1: Dim("seq_len", min=0, max=512)},
        {
            0: Dim("batch_size", min=0, max=8),
            1: Dim("seq_len", min=0, max=512),
        },
    )

    def _run_torch_export_to_gm():
        return torch_export_to_gm(
            model,
            args=tuple(),
            kwargs={"input_ids": input_ids, "position_ids": position_ids},
            dynamic_shapes=dynamic_shapes,
            patch_list=[
                "bamba",
                "autocast_noop",  # For "unsupported scalarType".
            ],
        )

    def _run_torch_export():
        with apply_export_patches(patch_list=["bamba", "autocast_noop"]):
            with torch.inference_mode():
                ep = te.export(
                    model,
                    args=(),
                    kwargs={"input_ids": input_ids, "position_ids": position_ids},
                    dynamic_shapes=dynamic_shapes,
                    strict=False,
                )
            egm = ep.module()
        return egm

    def _run_export():
        if export_func == "torch_export_to_gm":
            return _run_torch_export_to_gm()
        else:
            return _run_torch_export()

    if model_on_meta_during_export:
        gm = _run_export()
        factory.load_or_random_init(gm, device="cuda")
        move_to_device(gm, "cuda")
        factory._to_maybe_random(model, "cuda")
        model.load_state_dict(gm.state_dict())
    else:
        factory.load_or_random_init(model, device="cuda")
        gm = _run_export()
        move_to_device(gm, "cuda")

    if run_verify_generation:
        _verify_generation(factory, model, tokenizer)

    # let's do a comparison of every state dict item between the model and the gm
    torch.testing.assert_close(model.state_dict(), gm.state_dict(), rtol=0.0, atol=0.0)

    outputs_for_comparison = {}
    with torch.inference_mode():
        out_original = model(input_ids=input_ids, position_ids=position_ids)
        with apply_export_patches(patch_list=["bamba"]):
            outputs_for_comparison["patched"] = model(
                input_ids=input_ids, position_ids=position_ids
            )

    with torch.inference_mode():
        outputs_for_comparison["gm"] = gm(input_ids=input_ids, position_ids=position_ids)

    atol, rtol = 1e-3, 1e-3
    for comp, outs in outputs_for_comparison.items():
        print(f"====== COMPARISON ({comp}) ======")
        torch.testing.assert_close(
            outs,
            out_original,
            rtol=rtol,
            atol=atol,
        )


def _verify_generation(factory, model, tokenizer):
    print("====== WITHOUT PATCH ======")
    _generate(tokenizer, model)
    with apply_export_patches(patch_list=["bamba"]):
        print("====== WITH PATCH ======")
        _generate(tokenizer, model)


def _generate(tokenizer, model):
    messages = [
        EXAMPLE_INPUT,
        EXAMPLE_INPUT2,
    ]
    for msg in messages:
        num_tokens = tokenizer(
            msg, return_tensors="pt", return_token_type_ids=False
        ).input_ids.shape[1]
        print(f"{msg=}, {num_tokens=}")
    inputs = tokenizer(messages, return_tensors="pt", return_token_type_ids=False).to(model.device)
    response = model.generate(**inputs, max_new_tokens=64)
    print("\n".join(tokenizer.batch_decode(response, skip_special_tokens=True)))


if __name__ == "__main__":
    test_bamba_patches()
