import torch
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig

from tensorrt_llm._torch.auto_deploy import LlmArgs
from tensorrt_llm._torch.auto_deploy.export import apply_export_patches, torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device


def test_build_run_mistral3_vlm():
    experiment_config = get_small_model_config("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
    experiment_config = ExperimentConfig(**experiment_config)
    llm_args: LlmArgs = experiment_config.args

    factory = llm_args.create_factory()
    model = factory.build_model("cuda")

    inputs = factory.get_example_inputs_with_images()
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            dtype = torch.bfloat16 if isinstance(value, torch.FloatTensor) else None
            inputs[key] = value.to(device=model.device, dtype=dtype)

    # get relevant inputs
    input_ids = inputs["input_ids"]
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).repeat(
        input_ids.shape[0], 1
    )
    pixel_values = inputs["pixel_values"]
    image_sizes = inputs["image_sizes"]

    def _run_with_and_without_image(model, use_patch=True):
        with apply_export_patches(
            patch_configs={
                "hf_mistral3": {"enabled": use_patch},
                "hf_pixtral_vit": {"enabled": use_patch},
            }
        ):
            with torch.inference_mode():
                out_no_images = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    pixel_values=torch.zeros_like(pixel_values) if use_patch else None,
                    image_sizes=image_sizes if use_patch else None,
                )
                out_with_images = model(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                )
            return {"no_images": out_no_images.logits, "with_images": out_with_images.logits}

    # Get output pre-patch.
    out_original = _run_with_and_without_image(model, use_patch=False)

    # Get output post-patch.
    outputs_for_comparison = {}
    # TODO(2ez4bz): Figure out why the patches do not work outside of `torch_export_to_gm`.
    # outputs_for_comparison["model_with_patch"] = _run_with_and_without_image(model)

    gm = torch_export_to_gm(
        model,
        args=(),
        kwargs={
            "input_ids": input_ids,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
        },
        patch_configs={
            "transformers_sdpa_mask": {},
            "autocast_noop": {},
            "torch_where": {},
            "tensor_meta_device": {},
            "sdpa_kernel_noop": {},
            "sdpa": {},
            "hf_mistral3": {"enabled": True},
            "hf_pixtral_vit": {"enabled": True},
        },
    )
    move_to_device(gm, model.device)

    outputs_for_comparison["gm"] = _run_with_and_without_image(gm)

    atol, rtol = 1e-3, 1e-3
    for comp, outs in outputs_for_comparison.items():
        torch.testing.assert_close(
            outs,
            out_original,
            rtol=rtol,
            atol=atol,
        )
