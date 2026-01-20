import argparse
import os
from collections import abc

import modelopt.torch.opt as mto
import torch
import yaml
from diffusers import DiffusionPipeline
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from tensorrt_llm._torch.auto_deploy.compile import CompileBackendRegistry
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils._graph import load_buffers_and_params
from tensorrt_llm._torch.auto_deploy.utils.logger import ad_logger

torch._dynamo.config.cache_size_limit = 100

dtype_map = {
    "Half": torch.float16,
    "BFloat16": torch.bfloat16,
    "Float": torch.float32,
}


def load_config(config_path):
    """Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Dictionary with export, optimizer, and compile configurations.
    """
    if not config_path:
        raise ValueError("Config path is required. Use --config to specify a YAML config file.")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    ad_logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Validate required sections
    required_sections = ["export", "optimizer", "compile"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config file missing required section: {section}")

    return config


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


def clip_model():
    """Load CLIP model for image-text similarity evaluation."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, processor


def compute_clip_similarity(image_path: str, prompt: str, clip_model_and_processor) -> float:
    """Compute CLIP similarity score between generated image and text prompt.

    Args:
        image_path: Path to the generated image
        prompt: Text prompt used to generate the image
        clip_model_and_processor: Tuple of (CLIP model, CLIP processor)

    Returns:
        Similarity score between 0 and 1
    """
    model, processor = clip_model_and_processor
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # Normalize embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # Compute cosine similarity
        similarity = (image_embeds @ text_embeds.T).squeeze().item()

    return similarity


@torch.inference_mode()
def generate_image(pipe: DiffusionPipeline, prompt: str, image_name: str) -> None:
    """Generate an image using the given pipeline and prompt."""
    image = pipe(
        prompt,
        output_type="pil",
        num_inference_steps=30,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]
    image.save(image_name)
    ad_logger.info(f"Image generated saved as {image_name}")


@torch.inference_mode()
def benchmark_backbone_standalone(
    pipe, num_warmup=10, num_benchmark=100, model_name="flux-dev", model_dtype="Half"
):
    """Benchmark the backbone model directly without running the full pipeline."""
    backbone = pipe.transformer if hasattr(pipe, "transformer") else pipe.unet

    # Generate dummy inputs for the backbone
    dummy_inputs = _gen_dummy_inp_flux(backbone)

    # Warmup
    ad_logger.info(f"Warming up: {num_warmup} iterations")
    for _ in tqdm(range(num_warmup), desc="Warmup"):
        _ = backbone(**dummy_inputs)

    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    ad_logger.info(f"Benchmarking: {num_benchmark} iterations")
    times = []
    for _ in tqdm(range(num_benchmark), desc="Benchmark"):
        torch.cuda.profiler.cudart().cudaProfilerStart()
        start_event.record()
        _ = backbone(**dummy_inputs)
        end_event.record()
        torch.cuda.synchronize()
        torch.cuda.profiler.cudart().cudaProfilerStop()
        times.append(start_event.elapsed_time(end_event))

    avg_latency = sum(times) / len(times)
    times = sorted(times)
    p50 = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    p99 = times[int(len(times) * 0.99)]

    ad_logger.info(f"\nBackbone-only inference latency ({model_dtype}):")
    ad_logger.info(f"  Average: {avg_latency:.2f} ms")
    ad_logger.info(f"  P50: {p50:.2f} ms")
    ad_logger.info(f"  P95: {p95:.2f} ms")
    ad_logger.info(f"  P99: {p99:.2f} ms")

    return avg_latency


def torch_to(data, *args, **kwargs):
    """Try to recursively move the data to the specified args/kwargs."""
    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list)):
        return type(data)([torch_to(val, *args, **kwargs) for val in data])
    elif isinstance(data, abc.Mapping):
        return {k: torch_to(val, *args, **kwargs) for k, val in data.items()}
    return data


def _gen_dummy_inp_flux(backbone, min_bs=1):
    cfg = backbone.config
    text_maxlen = 512
    img_dim = 4096

    dtype = torch.bfloat16
    dummy_input = {
        "hidden_states": torch.randn(min_bs, img_dim, cfg.in_channels, dtype=dtype),
        "encoder_hidden_states": torch.randn(
            min_bs, text_maxlen, cfg.joint_attention_dim, dtype=dtype
        ),
        "pooled_projections": torch.randn(min_bs, cfg.pooled_projection_dim, dtype=dtype),
        "timestep": torch.ones(1, dtype=dtype),
        "img_ids": torch.randn(img_dim, 3, dtype=torch.float32),
        "txt_ids": torch.randn(text_maxlen, 3, dtype=torch.float32),
        "return_dict": False,
        "joint_attention_kwargs": {},
    }
    if cfg.guidance_embeds:  # flux-dev
        dummy_input["guidance"] = torch.full((1,), 3.5, dtype=torch.float32)

    dummy_input = torch_to(dummy_input, device="cuda")

    return dummy_input


def execution_device_getter(self):
    return torch.device("cuda")


def execution_device_setter(self, value):
    self.__dict__["_execution_device"] = torch.device("cuda")


def main(argv=None):
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
    parser.add_argument(
        "--image_path",
        type=str,
        default="output.png",
        help="Path to save the generated image (default: output.png)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "flux_transforms.yaml"),
        help="Path to YAML config file for export, optimizer, and compile settings (default: flux_transforms.yaml)",
    )
    args = parser.parse_args(argv)

    # Validate max_batch_size
    if args.max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {args.max_batch_size}")

    DiffusionPipeline._execution_device = property(execution_device_getter, execution_device_setter)
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    # Load CLIP model for similarity evaluation if generating images
    clip_model_processor = None
    if not args.skip_image_generation:
        ad_logger.info("Loading CLIP model for similarity evaluation")
        clip_model_processor = clip_model()

    if args.hf_inference:
        if not args.skip_image_generation:
            ad_logger.info("Generating image with the torch pipeline")
            hf_image_path = f"hf_{args.image_path}"
            generate_image(pipe, args.prompt, hf_image_path)

            # Compute CLIP similarity score
            similarity = compute_clip_similarity(hf_image_path, args.prompt, clip_model_processor)
            ad_logger.info(f"CLIP similarity score (HF): {similarity:.4f}")
        if args.benchmark:
            ad_logger.info("Benchmarking HuggingFace model")
            latency = benchmark_backbone_standalone(pipe, model_dtype="BFloat16")
            ad_logger.info(f"HuggingFace Average Inference Latency: {latency:.2f} ms")
    model = pipe.transformer
    flux_config = pipe.transformer.config
    flux_kwargs = _gen_dummy_inp_flux(model, min_bs=args.max_batch_size)

    # Load config from YAML
    config = load_config(args.config)

    # Restore quantizers
    if args.restore_from:
        ad_logger.info(f"Restoring model from {args.restore_from}")
        try:
            mto.restore(model, args.restore_from)
            quant_state_dict = model.state_dict()
            load_buffers_and_params(
                model, quant_state_dict, strict_missing=False, strict_unexpected=False, clone=False
            )
        except Exception as e:
            ad_logger.error(f"Failed to restore model from {args.restore_from}: {e}")
            raise

    # Export to graph module with config params
    ad_logger.info("Exporting model to graph module...")
    export_config = config["export"]
    gm = torch_export_to_gm(
        model,
        args=(),
        kwargs=flux_kwargs,
        clone=export_config.get("clone", False),
        strict=export_config.get("strict", False),
    )

    # Apply inference optimizer fusions
    optimizer_config = config.get("optimizer")
    if optimizer_config:
        ad_logger.info("Applying inference optimizer fusions (FP8 and FP4)...")
        optimizer = InferenceOptimizer(factory=None, config=optimizer_config)
        gm = optimizer(cm=None, mod=gm)
        ad_logger.info("Inference optimizer fusions applied successfully")
    else:
        ad_logger.info("No optimizer transforms configured, skipping optimizer fusions")

    # Compile model with config params
    compile_config = config["compile"]
    backend = compile_config.get("backend", "torch-opt")
    cuda_graph_batch_sizes = compile_config.get("cuda_graph_batch_sizes", None)

    # Validate backend availability
    if not CompileBackendRegistry.has(backend):
        available = CompileBackendRegistry.list()
        raise ValueError(f"Backend '{backend}' not found. Available backends: {available}")

    ad_logger.info(f"Compiling model with backend: {backend}")
    compiler_cls = CompileBackendRegistry.get(backend)
    gm = compiler_cls(
        gm,
        args=(),
        max_batch_size=args.max_batch_size,
        kwargs=flux_kwargs,
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
    ).compile()

    del model
    fx_model = TransformerWrapper(gm, flux_config)
    pipe.transformer = fx_model
    if not args.skip_image_generation:
        ad_logger.info("Generating image with the exported auto-deploy model")
        generate_image(pipe, args.prompt, args.image_path)

        # Compute CLIP similarity score
        similarity = compute_clip_similarity(args.image_path, args.prompt, clip_model_processor)
        ad_logger.info(f"CLIP similarity score (AutoDeploy): {similarity:.4f}")

    if args.benchmark:
        ad_logger.info("Benchmarking AutoDeploy model")
        latency = benchmark_backbone_standalone(pipe, model_dtype="BFloat16")
        ad_logger.info(f"AutoDeploy Average Inference Latency: {latency:.2f} ms")


if __name__ == "__main__":
    main()
