import argparse
import os

import tensorrt as trt
import torch
from diffusers import DiffusionPipeline

import tensorrt_llm
from tensorrt_llm.builder import Builder
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.unet.pp.unet_pp import DistriUNetPP
from tensorrt_llm.models.unet.unet_2d_condition import UNet2DConditionModel
from tensorrt_llm.models.unet.weights import load_from_hf_unet
from tensorrt_llm.network import net_guard

parser = argparse.ArgumentParser(description='build the UNet TensorRT engine.')
parser.add_argument('--model_dir',
                    type=str,
                    default='stabilityai/stable-diffusion-xl-base-1.0')
parser.add_argument('--size', type=int, default=1024, help='image size')
parser.add_argument('--output_dir',
                    type=str,
                    default=None,
                    help='output directory')

args = parser.parse_args()

model_dir = args.model_dir
size = args.size
sample_size = size // 8

world_size = tensorrt_llm.mpi_world_size()
rank = tensorrt_llm.mpi_rank()
output_dir = f'sdxl_s{size}_w{world_size}' if args.output_dir is None else args.output_dir
if rank == 0 and not os.path.exists(output_dir):
    os.makedirs(output_dir)

device_per_batch = world_size // 2 if world_size > 1 else 1
batch_group = 2 if world_size > 1 else 1

# Use tp_size to indicate the size of patch parallelism
# Use pp_size to indicate the size of batch parallelism
mapping = Mapping(world_size=world_size,
                  rank=rank,
                  tp_size=device_per_batch,
                  pp_size=batch_group)

torch.cuda.set_device(tensorrt_llm.mpi_rank())

tensorrt_llm.logger.set_level('verbose')
builder = Builder()
builder_config = builder.create_builder_config(
    name='UNet2DConditionModel',
    precision='float16',
    timing_cache='model.cache',
    profiling_verbosity='detailed',
    tensor_parallel=world_size,
    precision_constraints=
    None,  # do not use obey or the precision error will be too large
)

pipeline = DiffusionPipeline.from_pretrained(model_dir,
                                             torch_dtype=torch.float16)
model = UNet2DConditionModel(
    sample_size=sample_size,
    in_channels=4,
    out_channels=4,
    center_input_sample=False,
    flip_sin_to_cos=True,
    freq_shift=0,
    down_block_types=("DownBlock2D", "CrossAttnDownBlock2D",
                      "CrossAttnDownBlock2D"),
    up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    block_out_channels=(320, 640, 1280),
    layers_per_block=2,
    downsample_padding=1,
    mid_block_scale_factor=1.0,
    act_fn="silu",
    norm_num_groups=32,
    norm_eps=1e-5,
    cross_attention_dim=2048,
    attention_head_dim=[5, 10, 20],
    addition_embed_type="text_time",
    addition_time_embed_dim=256,
    projection_class_embeddings_input_dim=2816,
    transformer_layers_per_block=[1, 2, 10],
    use_linear_projection=True,
    dtype=trt.float16,
)

load_from_hf_unet(pipeline.unet, model)
model = DistriUNetPP(model, mapping)

# Module -> Network
network = builder.create_network()
network.plugin_config.to_legacy_setting()
if mapping.world_size > 1:
    network.plugin_config.set_nccl_plugin('float16')

with net_guard(network):
    # Prepare
    network.set_named_parameters(model.named_parameters())

    # Forward
    sample = tensorrt_llm.Tensor(
        name='sample',
        dtype=trt.float16,
        shape=[2, 4, sample_size, sample_size],
    )
    timesteps = tensorrt_llm.Tensor(
        name='timesteps',
        dtype=trt.float16,
        shape=[
            1,
        ],
    )
    encoder_hidden_states = tensorrt_llm.Tensor(
        name='encoder_hidden_states',
        dtype=trt.float16,
        shape=[2, 77, 2048],
    )
    text_embeds = tensorrt_llm.Tensor(
        name='text_embeds',
        dtype=trt.float16,
        shape=[2, 1280],
    )
    time_ids = tensorrt_llm.Tensor(
        name='time_ids',
        dtype=trt.float16,
        shape=[2, 6],
    )

    output = model(sample, timesteps, encoder_hidden_states, text_embeds,
                   time_ids)

    # Mark outputs
    output_dtype = trt.float16
    output.mark_output('pred', output_dtype)

# Network -> Engine
engine = builder.build_engine(network, builder_config)
assert engine is not None, 'Failed to build engine.'

engine_name = f'sdxl_unet_s{size}_w{world_size}_r{rank}.engine'
engine_path = os.path.join(output_dir, engine_name)
with open(engine_path, 'wb') as f:
    f.write(engine)
builder.save_config(builder_config, os.path.join(output_dir, 'config.json'))
