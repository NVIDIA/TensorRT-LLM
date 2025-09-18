import torch  # noqa
from torch.export import Dim  # noqa
from transformers import AutoTokenizer

from tensorrt_llm._torch.auto_deploy.export import apply_export_patches, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.llm_args import AutoDeployConfig
from tensorrt_llm._torch.auto_deploy.transformations._graph import move_to_device  # noqa

MODEL_DIR = (
    "/home/scratch.williamz_gpu/code/trtc/builder/hf_cache/hub/models--ibm-ai-platform--Bamba-9B-v2/"
    "snapshots/b42852dc9eb96c8ae3359dc8df0e4c3f5c37eb21"
)
EXAMPLE_INPUT = "Mamba is a snake with the following properties:"


def test_bamba_patches():
    llm_args = AutoDeployConfig(
        **{
            "model": MODEL_DIR,
            "world_size": 0,
            "runtime": "demollm",
            "compile_backend": "torch-simple",
            "attn_backend": "flashinfer",
            "model_factory": "AutoModelForCausalLM",
            "model_kwargs": {
                "use_cache": True,
            },
            "max_seq_len": 512,
        },
    )

    factory = llm_args.create_factory()
    model = factory.build_model("meta")
    factory.load_or_random_init(model, device="cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    _verify_generation(factory, model, tokenizer)

    # 1. Export wants min batch size of 2 (to avoid specialization during export).
    # 2. Can't get `padding` / `truncation` to work without other steps so just use the same prompt
    #    twice in order for the tokenizer not to complain when creating the tensor.
    message = [EXAMPLE_INPUT] * 2
    inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False).to(model.device)

    input_ids = inputs["input_ids"]
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
        input_ids.shape[0], 1
    )

    outputs_for_comparison = {}
    with torch.inference_mode():
        out_original = model(input_ids=input_ids, position_ids=position_ids)
        with apply_export_patches(patch_list=["bamba"]):
            outputs_for_comparison["patched"] = model(
                input_ids=input_ids, position_ids=position_ids
            )

    dynamic_shapes = (
        {0: Dim("batch_size", min=0, max=8), 1: Dim("seq_len", min=0, max=512)},
        {
            0: Dim("batch_size", min=0, max=8),
            1: Dim("seq_len", min=0, max=512),
        },
    )
    print("====== EXPORTING GRAPH MODULE ======")
    gm = torch_export_to_gm(
        model,
        args=(input_ids, position_ids),
        dynamic_shapes=dynamic_shapes,
        patch_list=[
            "bamba",
            # For "unsupported scalarType".
            "autocast_noop",
        ],
    )
    move_to_device(gm, model.device)

    with torch.inference_mode():
        outputs_for_comparison["gm"] = gm(input_ids, position_ids)

    atol, rtol = 1e-3, 1e-3
    for comp, outs in outputs_for_comparison.items():
        print(f"====== COMPARISON ({comp}) ======")
        try:
            torch.testing.assert_close(
                outs,
                out_original,
                rtol=rtol,
                atol=atol,
            )
            print("Passed!")
        except AssertionError as e:
            print(e)
            diff = torch.abs(outs.logits - out_original.logits)
            print(f"abs diff: {diff}")
            print(f"average diff: {diff.mean()}")


# NOTE: remember to augment `_simple_forward` to pass `*args, **kwargs`.
def _verify_generation(factory, model, tokenizer):
    # print("====== WITHOUT PATCH ======")
    # _generate(tokenizer, model)
    with apply_export_patches(patch_list=["bamba"]):
        print("====== WITH PATCH ======")
        _generate(tokenizer, model)


def _generate(tokenizer, model):
    message = [EXAMPLE_INPUT]
    inputs = tokenizer(message, return_tensors="pt", return_token_type_ids=False).to(model.device)
    response = model.generate(**inputs, max_new_tokens=64)
    print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])


# def _get_example_inputs(llm_args, factory, device):
#     batch_size = min(2, llm_args.max_batch_size)
#     seq_len = min(4, llm_args.max_seq_len)
#     inputs = {"input_ids": torch.ones(batch_size, seq_len, dtype=torch.int)}
#     for key, value in inputs.items():
#         if isinstance(value, torch.Tensor):
#             dtype = torch.bfloat16 if isinstance(value, torch.FloatTensor) else None
#             inputs[key] = value.to(device=device, dtype=dtype)
#
#     return inputs
