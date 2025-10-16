import argparse
from typing import Any

import modelopt.torch.opt as mto
import torch
from diffusers import DiffusionPipeline

from tensorrt_llm._torch.auto_deploy.compile import CompileBackendRegistry
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transformations._graph import load_buffers_and_params
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

torch._dynamo.config.cache_size_limit = 100


# TODO: Reuse the cache context from the original model
class TransformerWrapper(torch.nn.Module):
    def __init__(self, compiled_model, config):
        super().__init__()
        self.model = compiled_model
        self.config = config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def cache_context(self, *args, **kwargs):
        # Return a no-op context manager since the compiled model
        # doesn't support this feature
        from contextlib import contextmanager

        @contextmanager
        def noop_context():
            yield

        return noop_context()


def generate_image(pipe: DiffusionPipeline, prompt: str, image_name: str) -> None:
    """Generate an image using the given pipeline and prompt."""
    seed = 42
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    image.save(image_name)
    ad_logger.info(f"Image generated saved as {image_name}")


@torch.inference_mode()
def benchmark_model(model, generate_dummy_inputs, benchmarking_runs=200, warmup_runs=25) -> float:
    """Returns the latency of the model in seconds."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    input_data = generate_dummy_inputs()

    for _ in range(warmup_runs):
        _ = model(**input_data)

    torch.cuda.synchronize()

    torch.cuda.profiler.cudart().cudaProfilerStart()
    start_event.record()
    for _ in range(benchmarking_runs):
        _ = model(**input_data)
    end_event.record()
    end_event.synchronize()
    torch.cuda.profiler.cudart().cudaProfilerStop()

    return start_event.elapsed_time(end_event) / benchmarking_runs / 1000


def generate_dummy_inputs(
    device: str = "cuda", model_dtype: torch.dtype = torch.bfloat16
) -> dict[str, Any]:
    """Generate dummy inputs for the flux transformer."""
    assert model_dtype in [torch.bfloat16, torch.float16], (
        "Model dtype must be either bfloat16 or float16"
    )
    dummy_input = {}
    text_maxlen = 512
    dummy_input["hidden_states"] = torch.randn(1, 4096, 64, dtype=model_dtype, device=device)
    dummy_input["timestep"] = torch.tensor(data=[1.0] * 1, dtype=model_dtype, device=device)
    dummy_input["guidance"] = torch.full((1,), 3.5, dtype=torch.float32, device=device)
    dummy_input["pooled_projections"] = torch.randn(1, 768, dtype=model_dtype, device=device)
    dummy_input["encoder_hidden_states"] = torch.randn(
        1, text_maxlen, 4096, dtype=model_dtype, device=device
    )
    dummy_input["txt_ids"] = torch.randn(text_maxlen, 3, dtype=torch.float32, device=device)
    dummy_input["img_ids"] = torch.randn(4096, 3, dtype=torch.float32, device=device)
    dummy_input["joint_attention_kwargs"] = {}
    dummy_input["return_dict"] = False

    return dummy_input


def execution_device_getter(self):
    return torch.device("cuda")  # Always return CUDA


def execution_device_setter(self, value):
    self.__dict__["_execution_device"] = torch.device("cuda")  # Force CUDA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="black-forest-labs/FLUX.1-dev",
        help="The model to use for inference.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="The prompt to use for inference.",
    )
    parser.add_argument(
        "--hf_inference",
        action="store_true",
        help="Whether to generate image with the base hf model in addition to autodeploy generation",
    )
    parser.add_argument(
        "--restore_from", type=str, help="The quantized checkpoint path to restore the model from"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Whether to benchmark the model",
    )
    parser.add_argument(
        "--skip_image_generation",
        action="store_true",
        help="Whether to skip image generation",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=1,
        help="The max batch size to use for the model",
    )
    args = parser.parse_args()
    DiffusionPipeline._execution_device = property(execution_device_getter, execution_device_setter)
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    if args.hf_inference:
        if not args.skip_image_generation:
            ad_logger.info("Generating image with the torch pipeline")
            generate_image(pipe, args.prompt, "hf_mars_horse.png")
        if args.benchmark:
            latency = benchmark_model(pipe.transformer, generate_dummy_inputs)
            ad_logger.info(f"HuggingFace Latency: {latency} seconds")
    model = pipe.transformer
    if args.restore_from:
        ad_logger.info(f"Restoring model from {args.restore_from}")
        mto.restore(model, args.restore_from)
    flux_config = pipe.transformer.config
    flux_kwargs = generate_dummy_inputs()

    gm = torch_export_to_gm(model, args=(), kwargs=flux_kwargs, clone=True)

    if args.restore_from:
        # Load quantized weights from the restored model into the exported graph module
        quant_state_dict = model.state_dict()
        load_buffers_and_params(
            gm, quant_state_dict, strict_missing=False, strict_unexpected=False, clone=False
        )

    compiler_cls = CompileBackendRegistry.get("torch-opt")
    gm = compiler_cls(gm, args=(), max_batch_size=args.max_batch_size, kwargs=flux_kwargs).compile()

    del model
    fx_model = TransformerWrapper(gm, flux_config)
    pipe.transformer = fx_model
    if not args.skip_image_generation:
        ad_logger.info("Generating image with the exported auto-deploy model")
        generate_image(pipe, args.prompt, "autodeploy_mars_horse_gm.png")

    if args.benchmark:
        latency = benchmark_model(fx_model, generate_dummy_inputs)
        ad_logger.info(f"AutoDeploy Latency: {latency} seconds")


if __name__ == "__main__":
    main()
