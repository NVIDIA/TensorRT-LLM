import argparse
import glob
import json
import os

import torch
from pipeline_tllm import TllmOpenSoraPipeline, TllmSTDiT
from safetensors.torch import load_file
from scheduler import RFlowScheduler
from text_encoder import CaptionEmbedder, T5Encoder
from utils import DataProcessor
from vae import get_vae

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch


class RuntimeConfig(dict):

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(
            f"'RuntimeConfig {self}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def main(cfg):
    tensorrt_llm.logger.set_level(cfg.log_level)

    torch.set_grad_enabled(False)
    dtype = cfg.get("dtype", "float16")
    device = torch.device('cuda')

    config_file = os.path.join(cfg.tllm_model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    ## Build modules
    model_config = config.get("pretrained_config")
    stdit = TllmSTDiT(config, cfg.tllm_model_dir, debug_mode=cfg.debug_mode)
    # HACK: for classifier-free guidance
    ckpt_path = model_config['checkpoint_path']
    if os.path.isdir(ckpt_path):
        ckpt_files = glob.glob(model_config['checkpoint_path'] + "/*")
    else:
        ckpt_files = [ckpt_path]
    pretrained_weights = None
    for file in ckpt_files:
        if file.endswith('.safetensors'):
            pretrained_weights = load_file(file)
        elif file.endswith('.pt'):
            pretrained_weights = dict(torch.load(file, weights_only=True))
        if pretrained_weights is not None:
            break
    if pretrained_weights is None:
        raise FileNotFoundError
    y_embedder = CaptionEmbedder(
        in_channels=model_config.get('caption_channels'),
        hidden_size=model_config.get('hidden_size'),
        uncond_prob=model_config.get('class_dropout_prob'),
        act_layer=torch.nn.GELU(approximate="tanh"),
        token_num=model_config.get('model_max_length'),
    )
    for name, param in y_embedder.named_parameters():
        param.data = pretrained_weights['y_embedder.' + name]
    y_embedder.y_embedding = pretrained_weights['y_embedder.y_embedding']
    reuse_y_embedding = pretrained_weights['y_embedder.y_embedding']
    text_encoder = T5Encoder(
        from_pretrained=cfg.text_encoder,
        model_max_length=model_config.get('model_max_length'),
        caption_channels=model_config.get('caption_channels'),
        y_embedding=reuse_y_embedding,  # HACK: for classifier-free guidance
        hidden_size=model_config.get('hidden_size'),
        device=device,
    )
    vae = get_vae(
        from_pretrained=cfg.vae,
        micro_frame_size=17,
        micro_batch_size=4,
    ).to(dtype=str_dtype_to_torch(dtype), device=device).eval()
    scheduler = RFlowScheduler(
        use_timestep_transform=True,
        num_timesteps=cfg.num_timesteps,
        num_sampling_steps=cfg.num_sampling_steps,
    )
    pipe = TllmOpenSoraPipeline(
        stdit=stdit,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        num_sampling_steps=cfg.num_sampling_steps,
        num_timesteps=cfg.num_timesteps,
        cfg_scale=cfg.cfg_scale,
        align=cfg.get("align", None),
        aes=cfg.get("aes", None),
        flow=cfg.get("flow", None),
        camera_motion=cfg.get("camera_motion", None),
        resolution=cfg.get("resolution", None),
        aspect_ratio=cfg.get("aspect_ratio", None),
        num_frames=cfg.get("num_frames", None),
        fps=cfg.get("fps", 30),
        save_fps=cfg.get("save_fps",
                         cfg.get("fps", 30) // cfg.get("frame_interval", 1)),
        diffusion_model_type=cfg.get("diffusion_model_type", None),
        condition_frame_length=cfg.get("condition_frame_length", 5),
        condition_frame_edit=cfg.get("condition_frame_edit", 0.0),
        use_timestep_transform=True,
        video_save_dir=cfg.get("video_save_dir", "sample_outputs"),
        dtype=dtype,
        device=device,
        seed=cfg.get('seed', 0),
    )

    # load prompts
    prompts = cfg.get("prompt", None)
    start_idx = cfg.get("start_index", 0)
    if prompts is None:
        if cfg.get("prompt_path", None) is not None:
            prompts = DataProcessor.load_prompts(cfg.prompt_path, start_idx,
                                                 cfg.get("end_index", None))
        else:
            prompts = [cfg.get("prompt_generator", "")
                       ] * 1_000_000  # endless loop

    pipe(prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prompt',
                        nargs='*',
                        help="Text prompt(s) to guide video generation")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resolution", type=str, default="360p")
    parser.add_argument("--aspect-ratio", type=str, default="9:16")
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2946901)
    parser.add_argument("--tllm_model_dir",
                        type=str,
                        default='./engine_outputs/')
    parser.add_argument("--video_save_dir",
                        type=str,
                        default='./sample_outputs/')
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument("--debug_mode", action='store_true')
    args = parser.parse_args()

    runtime_config = dict(
        num_frames=102,
        fps=30,
        frame_interval=1,
        save_fps=30,
        diffusion_model_type="STDiT3",
        text_encoder="DeepFloyd/t5-v1_1-xxl",
        vae="hpcai-tech/OpenSora-VAE-v1.2",
        cfg_scale=7.0,
        dtype="float16",
        condition_frame_length=5,
        align=5,
        aes=6.5,
        flow=None,
        prompt="A scene from disaster movie.",
    )
    runtime_config.update(vars(args))

    main(RuntimeConfig(runtime_config))
