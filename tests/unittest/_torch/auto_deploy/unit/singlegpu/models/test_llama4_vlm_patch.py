import torch
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig
from PIL import Image

from tensorrt_llm._torch.auto_deploy import LlmArgs
from tensorrt_llm._torch.auto_deploy.export import apply_export_patches, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device


def test_build_run_llama4_vlm():
    atol = 1e-3
    rtol = 1e-3

    experiment_config = get_small_model_config("meta-llama/Llama-4-Scout-17B-16E-Instruct")
    experiment_config["args"]["model_kwargs"]["_attn_implementation"] = "eager"
    experiment_config = ExperimentConfig(**experiment_config)
    llm_args: LlmArgs = experiment_config.args

    factory = llm_args.create_factory()
    model = factory.build_model("cuda")
    processor = factory.init_processor()

    img1 = Image.new("RGB", (16, 16), color=(128, 128, 128))
    img2 = Image.new("RGB", (16, 16), color=(64, 64, 64))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img1},
                {"type": "image", "image": img2},
                {
                    "type": "text",
                    "text": "Describe what you see in the two images and their differences.",
                },
            ],
        },
    ]

    inputs = (
        processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        .to(model.device)
        .to(torch.bfloat16)
    )

    # get relevant inputs
    input_ids = inputs["input_ids"]
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
        input_ids.shape[0], 1
    )
    pixel_values = inputs["pixel_values"]

    def _run_with_and_without_image(model, use_patch=True):
        with apply_export_patches(patch_configs={"hf_llama4_vision": {"enabled": use_patch}}):
            with torch.inference_mode():
                out_no_images = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    pixel_values=torch.zeros_like(pixel_values) if use_patch else None,
                )
                out_with_images = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    pixel_values=pixel_values,
                )
            return {"no_images": out_no_images.logits, "with_images": out_with_images.logits}

    # Get output pre-patch
    out_original = _run_with_and_without_image(model, use_patch=False)

    # Get output post-patch
    outputs_for_comparison = {}
    outputs_for_comparison["model_with_patch"] = _run_with_and_without_image(model)

    # Export to GM
    gm = torch_export_to_gm(
        model,
        kwargs={
            "input_ids": input_ids,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
        },
        patch_configs={
            "transformers_sdpa_mask": {},
            "autocast_noop": {},
            "torch_where": {},
            "tensor_meta_device": {},
            "sdpa_kernel_noop": {},
            "sdpa": {},
            "hf_llama4_vision": {"enabled": True},
        },
    )
    move_to_device(gm, model.device)

    # Get the output post export
    outputs_for_comparison["gm"] = _run_with_and_without_image(gm)

    # Run comparisons to out_original with no patch now...
    for comp, outs in outputs_for_comparison.items():
        torch.testing.assert_close(
            outs,
            out_original,
            rtol=rtol,
            atol=atol,
            msg=lambda m: f"Comparison for {comp} failed:\n{m}",
        )
