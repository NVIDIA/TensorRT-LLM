import argparse
import time

import numpy as np
import torch
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

import tensorrt_llm

world_size = tensorrt_llm.mpi_world_size()
rank = tensorrt_llm.mpi_rank()


def parseArgs():
    parser = argparse.ArgumentParser(
        description='run SDXL with the UNet TensorRT engine.')
    parser.add_argument('--model_dir',
                        type=str,
                        default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=233)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument(
        '--prompt',
        type=str,
        default=
        "masterpiece, gouache painting, 1girl, distant view, lone boat, willow trees"
    )
    parser.add_argument('--engine_dir',
                        type=str,
                        default=None,
                        help='engine directory')
    parser.add_argument('--num-warmup-runs', type=int, default=3)
    parser.add_argument('--avg-runs', type=int, default=10)
    parser.add_argument("--ignore_ratio",
                        type=float,
                        default=0.2,
                        help="Ignored ratio of the slowest and fastest steps")
    parser.add_argument("--output",
                        type=str,
                        default="output.png",
                        help="Output file name")
    return parser.parse_args()


if __name__ == "__main__":
    args = parseArgs()
    model_dir = args.model_dir
    size = args.size
    seed = args.seed
    prompt = args.prompt
    num_inference_steps = args.num_inference_steps
    engine_dir = f'sdxl_s{size}_w{world_size}' if args.engine_dir is None else args.engine_dir
    num_warmup_runs = args.num_warmup_runs
    avg_runs = args.avg_runs
    output_file = args.output

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipeline.set_progress_bar_config(disable=rank != 0)
    pipeline.prepare(engine_dir, size)
    pipeline.to('cuda')

    # warm up
    for i in range(num_warmup_runs):
        image = pipeline(
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            height=size,
            width=size).images[0]

    latency_list = []
    for i in range(avg_runs):
        st = time.time()
        image = pipeline(
            num_inference_steps=num_inference_steps,
            prompt=prompt,
            generator=torch.Generator(device="cuda").manual_seed(seed),
            height=size,
            width=size,
        ).images[0]
        ed = time.time()
        latency_list.append(ed - st)

    latency_list = sorted(latency_list)
    ignored_count = int(args.ignore_ratio * len(latency_list) / 2)
    if ignored_count > 0:
        latency_list = latency_list[ignored_count:-ignored_count]

    if rank == 0:
        print(f"Avg latency: {np.sum(latency_list) / len(latency_list):.5f} s")
        image.save(output_file)
