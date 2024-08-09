import argparse
import time

import numpy as np
import torch
from pipeline_stable_diffusion_xl import StableDiffusionXLPipeline

import tensorrt_llm

world_size = tensorrt_llm.mpi_world_size()
rank = tensorrt_llm.mpi_rank()

parser = argparse.ArgumentParser(
    description='run SDXL with the UNet TensorRT engine.')
parser.add_argument('--size', type=int, default=1024)
parser.add_argument('--seed', type=int, default=233)
parser.add_argument('--num_inference_steps', type=int, default=50)
parser.add_argument(
    '--prompt',
    type=str,
    default=
    "masterpiece, gouache painting, 1girl, distant view, lone boat, willow trees"
)
parser.add_argument('--model_dir',
                    type=str,
                    default=None,
                    help='model directory')

args = parser.parse_args()
size = args.size
seed = args.seed
prompt = args.prompt
num_inference_steps = args.num_inference_steps
model_dir = f'sdxl_s{size}_w{world_size}' if args.model_dir is None else args.model_dir

pipeline = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
pipeline.set_progress_bar_config(disable=rank != 0)
pipeline.prepare(f'sdxl_s{size}_w{world_size}', size)
pipeline.to('cuda')

li = []
for i in range(10):
    st = time.time()
    image = pipeline(num_inference_steps=num_inference_steps,
                     prompt=prompt,
                     generator=torch.Generator(device="cuda").manual_seed(seed),
                     height=size,
                     width=size).images[0]
    ed = time.time()
    li.append(ed - st)

if rank == 0:
    print(f'Avg latency: {np.sum(li[-7:]) / 7.0}s')
    image.save(f"output.png")
