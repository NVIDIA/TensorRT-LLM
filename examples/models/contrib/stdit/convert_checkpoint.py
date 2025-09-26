import argparse
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from vae import get_vae

import tensorrt_llm
from tensorrt_llm._utils import release_gc
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models import STDiT3Model

PRETRAINED_STDIT_PATH = "hpcai-tech/OpenSora-STDiT-v3"


def pixel_size_to_latent_size(args):
    vae = get_vae(
        from_pretrained=args.vae_type,
        micro_frame_size=args.vae_micro_frame_size,
        micro_batch_size=args.vae_micro_batch_size,
    ).eval()
    spatial_patch_size = vae.spatial_vae.patch_size
    temporal_patch_size = vae.temporal_vae.patch_size
    vae_out_channels = vae.out_channels
    pixel_size = (args.num_frames, args.height, args.width)
    latent_size = vae.get_latent_size(pixel_size)
    return {
        'in_channels': vae_out_channels,
        'latent_size': latent_size,
        'spatial_patch_size': spatial_patch_size,
        'temporal_patch_size': temporal_patch_size,
    }


def size_str_to_list(repr):
    return [int(it) for it in repr.split('x')] if 'x' in repr else [int(repr)]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timm_ckpt', type=str, default=None)
    parser.add_argument('--output_dir',
                        type=str,
                        default='tllm_checkpoint',
                        help='The path to save the TensorRT LLM checkpoint')
    parser.add_argument('--caption_channels',
                        type=int,
                        default=4096,
                        help='The channel of input of caption embedder')
    parser.add_argument('--depth',
                        type=int,
                        default=28,
                        help='The number of STDiT blocks')
    parser.add_argument('--input_sq_size',
                        type=int,
                        default=512,
                        help='Base spatial position embedding size')
    parser.add_argument('--stdit_type',
                        type=str,
                        default="STDiT3",
                        choices=["STDiT3"])
    parser.add_argument('--stdit_patch_size',
                        type=str,
                        default='1x2x2',
                        help='The patch size of stdit for patchify')
    parser.add_argument('--width',
                        type=int,
                        default=640,
                        help='The width of image size')
    parser.add_argument('--height',
                        type=int,
                        default=360,
                        help='The height of image size')
    parser.add_argument('--num_frames',
                        type=int,
                        default=102,
                        help='The frames of generated video')
    parser.add_argument('--vae_type',
                        type=str,
                        default="hpcai-tech/OpenSora-VAE-v1.2",
                        choices=["hpcai-tech/OpenSora-VAE-v1.2"])
    parser.add_argument('--vae_micro_frame_size',
                        type=int,
                        default=17,
                        help='The micro_frame_size for vae')
    parser.add_argument('--vae_micro_batch_size',
                        type=int,
                        default=4,
                        help='The micro_batch_size for vae')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=1152,
                        help='The hidden size of STDiT')
    parser.add_argument('--num_heads',
                        type=int,
                        default=16,
                        help='The number of heads of attention module')
    parser.add_argument(
        '--mlp_ratio',
        type=float,
        default=4.0,
        help=
        'The ratio of hidden size compared to input hidden size in MLP layer')
    parser.add_argument(
        '--class_dropout_prob',
        type=float,
        default=0.1,
        help='The probability to drop class token when training')
    parser.add_argument('--model_max_length',
                        type=int,
                        default=300,
                        help='The max number of tokens (default: 300)')
    parser.add_argument('--text_encoder_type',
                        type=str,
                        default="DeepFloyd/t5-v1_1-xxl",
                        choices=["DeepFloyd/t5-v1_1-xxl"])
    parser.add_argument('--learn_sigma',
                        type=bool,
                        default=True,
                        help='Whether the model learn sigma')
    parser.add_argument('--tp_size',
                        type=int,
                        default=1,
                        help='N-way tensor parallelism size')
    parser.add_argument('--cp_size',
                        type=int,
                        default=1,
                        help='Context parallelism size')
    parser.add_argument('--pp_size',
                        type=int,
                        default=1,
                        help='N-way pipeline parallelism size')
    parser.add_argument('--dtype',
                        type=str,
                        default='float16',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--disable_qk_norm',
                        action='store_true',
                        help='Disable norm for qk in attention')
    parser.add_argument('--fp8',
                        action='store_true',
                        help='Whether use FP8 for layers')
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='The number of workers for converting checkpoint in parallel')
    parser.add_argument('--log_level', type=str, default='info')
    args = parser.parse_args()
    return args


def convert_and_save_model(args):
    # [NOTE] PP is not supported yet.
    world_size = args.tp_size * args.cp_size
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    def convert_and_save_rank(args, rank):
        mapping = Mapping(world_size=world_size,
                          rank=rank,
                          tp_size=args.tp_size,
                          cp_size=args.cp_size)
        # Process args
        runtime_config = {
            'architecture': "STDiT3",
            'checkpoint_path': os.path.abspath(args.timm_ckpt),
            'caption_channels': args.caption_channels,
            'num_hidden_layers': args.depth,
            'width': args.width,
            'height': args.height,
            'num_frames': args.num_frames,
            'hidden_size': args.hidden_size,
            'stdit_patch_size': size_str_to_list(args.stdit_patch_size),
            'input_sq_size': args.input_sq_size,
            'num_attention_heads': args.num_heads,
            'model_max_length': args.model_max_length,
            'mlp_ratio': args.mlp_ratio,
            'class_dropout_prob': args.class_dropout_prob,
            'learn_sigma': args.learn_sigma,
            'qk_norm': (not args.disable_qk_norm),
            'stdit_type': args.stdit_type,
            'vae_type': args.vae_type,
            'text_encoder_type': args.text_encoder_type,
        }
        runtime_config.update(pixel_size_to_latent_size(args))
        tik = time.time()
        stdit = STDiT3Model.from_pretrained(os.path.dirname(args.timm_ckpt),
                                            args.dtype,
                                            mapping=mapping,
                                            **runtime_config)
        stdit.save_checkpoint(args.output_dir, save_config=True)
        print(f'Total time of reading and converting: {time.time()-tik:.3f} s')
        tik = time.time()
        del stdit
        print(f'Total time of saving checkpoint: {time.time()-tik:.3f} s')

    execute(args.workers, [convert_and_save_rank] * world_size, args)
    release_gc()


def execute(workers, func, args):
    if workers == 1:
        for rank, f in enumerate(func):
            f(args, rank)
    else:
        with ThreadPoolExecutor(max_workers=workers) as p:
            futures = [p.submit(f, args, rank) for rank, f in enumerate(func)]
            exceptions = []
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    traceback.print_exc()
                    exceptions.append(e)
            assert len(
                exceptions
            ) == 0, "Checkpoint conversion failed, please check error log."


def main():
    print(tensorrt_llm.__version__)
    args = parse_arguments()
    logger.set_level(args.log_level)

    assert args.pp_size == 1, "PP is not supported yet."

    tik = time.time()

    if args.timm_ckpt is None:
        print(
            f"No pretrained checkpoint provided, use default checkpoint from Huggingface instead."
        )
        args.timm_ckpt = "./pretrained_ckpt/model.safetensors"
        if not os.path.exists(args.timm_ckpt):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=PRETRAINED_STDIT_PATH,
                              local_dir=os.path.dirname(args.timm_ckpt))

    convert_and_save_model(args)

    tok = time.time()
    t = time.strftime('%H:%M:%S', time.gmtime(tok - tik))
    print(f'Total time of converting checkpoints: {t}')


if __name__ == '__main__':
    main()
