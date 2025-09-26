import os
import time
from functools import wraps
from typing import List

import numpy as np
import torch
import torch.distributed
from cuda import cudart
from scheduler import timestep_transform
from utils import DataProcessor, print_progress_bar

import tensorrt_llm
from tensorrt_llm._utils import (str_dtype_to_torch, str_dtype_to_trt,
                                 torch_dtype_to_trt, trt_dtype_to_torch)
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.session import Session, TensorInfo


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


class TllmSTDiT():

    def __init__(self,
                 config,
                 tllm_model_dir,
                 debug_mode=True,
                 stream: torch.cuda.Stream = None):
        self.dtype = config['pretrained_config']['dtype']

        self.depth = config['pretrained_config']['num_hidden_layers']

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        gpus_per_node = config['pretrained_config']['mapping']['gpus_per_node']
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(world_size=world_size,
                                            rank=rank,
                                            cp_size=cp_size,
                                            tp_size=tp_size,
                                            pp_size=1,
                                            gpus_per_node=gpus_per_node)

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(tllm_model_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()

        assert engine_buffer is not None

        self.session = Session.from_serialized_engine(engine_buffer)

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        expected_tensor_names = [
            'x', 'timestep', 'y', 'mask', 'x_mask', 'fps', 'height', 'width',
            'output'
        ]

        if self.mapping.tp_size > 1:
            self.buffer, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))
            self.inputs['all_reduce_workspace'] = self.all_reduce_workspace
            expected_tensor_names += ['all_reduce_workspace']

        self.latent_size = config['pretrained_config']['latent_size']
        self.patch_size = config['pretrained_config']['stdit_patch_size']
        self.in_channels = config['pretrained_config']['in_channels']
        self.caption_channels = config['pretrained_config']['caption_channels']
        self.model_max_length = config['pretrained_config']['model_max_length']
        self.config = config['pretrained_config']

        self.max_cattn_seq_len = int(
            np.prod([
                np.ceil(d / p)
                for d, p in zip(self.latent_size, self.patch_size)
            ]))

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _get_extra_inputs_for_attention_plugin(self, batch_size, max_seq_len,
                                               max_encoder_seq_len):
        host_max_attention_window_sizes = torch.tensor([max_seq_len] *
                                                       self.depth,
                                                       dtype=torch.int32).cpu()
        host_sink_token_length = torch.tensor([0], dtype=torch.int32).cpu()
        context_lengths = torch.full([batch_size],
                                     max_seq_len,
                                     dtype=torch.int32).cuda()
        host_context_lengths = torch.full([batch_size],
                                          max_seq_len,
                                          dtype=torch.int32).cpu()
        host_request_types = torch.zeros([batch_size], dtype=torch.int32).cpu()
        perf_knob_tensor_size = 16
        host_runtime_perf_knobs = torch.tensor([-1] * perf_knob_tensor_size,
                                               dtype=torch.int64,
                                               device='cpu')
        host_context_progress = torch.tensor([0],
                                             dtype=torch.int64,
                                             device='cpu')
        cross_encoder_input_lengths = torch.full([batch_size],
                                                 max_encoder_seq_len,
                                                 dtype=torch.int32,
                                                 device='cuda')
        cross_encoder_max_input_length = torch.empty((max_encoder_seq_len, ),
                                                     dtype=torch.int32,
                                                     device='cuda')

        extra_inputs = {
            'host_max_attention_window_sizes': host_max_attention_window_sizes,
            'host_sink_token_length': host_sink_token_length,
            'context_lengths': context_lengths,
            'host_context_lengths': host_context_lengths,
            'encoder_input_lengths': cross_encoder_input_lengths,
            'encoder_max_input_length': cross_encoder_max_input_length,
            'host_request_types': host_request_types,
            'host_runtime_perf_knobs': host_runtime_perf_knobs,
            'host_context_progress': host_context_progress
        }
        return extra_inputs

    def _setup(self, batch_size, max_encoder_seq_len):
        input_info = [
            TensorInfo(name='x',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(batch_size, self.in_channels, *self.latent_size)),
            TensorInfo(name='timestep',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(batch_size, )),
            TensorInfo(name='y',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(batch_size, 1, self.model_max_length,
                              self.caption_channels)),
            TensorInfo(name='mask',
                       dtype=str_dtype_to_trt('int32'),
                       shape=(1, self.model_max_length)),
            TensorInfo(name='x_mask',
                       dtype=str_dtype_to_trt('bool'),
                       shape=(batch_size, self.latent_size[0])),
            TensorInfo(name='fps',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(1, )),
            TensorInfo(name='height',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(1, )),
            TensorInfo(name='width',
                       dtype=str_dtype_to_trt(self.dtype),
                       shape=(1, )),
        ]

        extra_inputs = self._get_extra_inputs_for_attention_plugin(
            batch_size=batch_size,
            max_seq_len=self.max_cattn_seq_len,
            max_encoder_seq_len=max_encoder_seq_len,
        )
        input_info += [
            tensorrt_llm.runtime.TensorInfo(name,
                                            torch_dtype_to_trt(tensor.dtype),
                                            tensor.shape)
            for name, tensor in extra_inputs.items()
        ]

        output_info = self.session.infer_shapes(input_info)
        for t_info in output_info:
            self.outputs[t_info.name] = torch.empty(tuple(t_info.shape),
                                                    dtype=trt_dtype_to_torch(
                                                        t_info.dtype),
                                                    device=self.device)
        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    @cuda_stream_guard
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, y: torch.Tensor,
                mask: torch.Tensor, x_mask: torch.Tensor, fps: torch.Tensor,
                height: torch.Tensor, width: torch.Tensor, y_lens: List[int]):
        """
        Forward pass of STDiT.
        x: (N, C, F, H, W)
        timestep: (N,)
        y: ()
        mask: ()
        x_mask: (N * 2, )
        fps: (1)
        height: (1)
        width: (1)
        """
        self._setup(x.shape[0], max_encoder_seq_len=np.max(y_lens))
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        inputs = {
            'x': x,
            "timestep": timestep,
            'y': y,
            "mask": mask,
            "x_mask": x_mask,
            "fps": fps,
            "height": height,
            "width": width,
        }
        self.inputs.update(**inputs)
        extra_inputs = self._get_extra_inputs_for_attention_plugin(
            batch_size=x.shape[0],
            max_seq_len=self.max_cattn_seq_len,
            max_encoder_seq_len=np.max(y_lens),
        )
        self.inputs.update(**extra_inputs)

        self.session.set_shapes(self.inputs)
        ok = self.session.run(self.inputs, self.outputs,
                              self.stream.cuda_stream)

        if not ok:
            raise RuntimeError('Executing TRT engine failed!')
        debug_tensors = {}
        for name in list(self.outputs.keys()):
            if name != 'output':
                debug_tensors[name] = self.outputs.pop(name)
        if len(debug_tensors) == 0:
            return self.outputs['output']
        else:
            return self.outputs['output'], debug_tensors


class TllmOpenSoraPipeline():

    def __init__(
            self,
            stdit,
            text_encoder,
            vae,
            scheduler,
            num_sampling_steps=30,
            num_timesteps=1000,
            cfg_scale=4.0,
            align=None,
            aes=None,
            flow=None,
            camera_motion=None,
            image_size=None,
            resolution=None,
            aspect_ratio=None,
            num_frames=None,
            fps=None,
            save_fps=None,
            diffusion_model_type=None,
            condition_frame_length=None,
            condition_frame_edit=None,
            use_discrete_timesteps=False,
            use_timestep_transform=False,
            video_save_dir='./samples/',
            dtype='float16',
            device=torch.device('cuda'),
            seed=None,
            **kwargs,
    ):
        self.stdit = stdit
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.data_processor = DataProcessor(text_encoder=text_encoder, vae=vae)

        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.align = align
        self.aes = aes
        self.flow = flow
        self.camera_motion = camera_motion

        if image_size is None:
            assert (
                resolution is not None and aspect_ratio is not None
            ), "resolution and aspect_ratio must be provided if image_size is not provided"
            image_size = self.data_processor.get_image_size(
                resolution, aspect_ratio)
        self.image_size = image_size
        self.num_frames = self.data_processor.get_num_frames(num_frames)
        self.input_size = (num_frames, *image_size)
        self.latent_size = self.data_processor.get_latent_size(self.input_size)

        self.fps = fps
        self.save_fps = save_fps
        self.diffusion_model_type = diffusion_model_type
        self.condition_frame_length = condition_frame_length
        self.condition_frame_edit = condition_frame_edit

        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.video_save_dir = video_save_dir
        self.seed = self.data_processor.set_random_seed(seed)
        self.dtype = dtype
        self.device = device

        self.prompt_as_path = kwargs.get("prompt_as_path", False)
        self.watermark = kwargs.get("watermark", False)

    def __call__(
        self,
        prompts,
        batch_size=1,
        num_sample=1,
        loop=1,
        reference_path=None,
        mask_strategy=None,
        sample_name=None,
        start_idx=0,
    ):
        reference_path = [""] * len(
            prompts) if reference_path is None else reference_path
        mask_strategy = [""] * len(
            prompts) if mask_strategy is None else mask_strategy
        assert len(reference_path) == len(
            prompts), "Length of reference must be the same as prompts"
        assert len(mask_strategy) == len(
            prompts), "Length of mask_strategy must be the same as prompts"

        for i in range(0, len(prompts), batch_size):
            # == prepare batch prompts ==
            batch_prompts = prompts[i:i + batch_size]
            ms = mask_strategy[i:i + batch_size]
            refs = reference_path[i:i + batch_size]

            # == get json from prompts ==
            batch_prompts, refs, ms = self.data_processor.extract_json_from_prompts(
                batch_prompts, refs, ms)
            original_batch_prompts = batch_prompts

            # == get reference for condition ==
            refs = self.data_processor.collect_references_batch(
                refs, self.image_size)

            # == multi-resolution info ==
            model_args = self.data_processor.prepare_multi_resolution_info(
                self.diffusion_model_type, len(batch_prompts), self.image_size,
                self.num_frames, self.fps, self.device,
                str_dtype_to_torch(self.dtype))

            # == Iter over number of sampling for one prompt ==
            for k in range(num_sample):
                # == prepare save paths ==
                save_paths = [
                    self.data_processor.get_save_path_name(
                        self.video_save_dir,
                        sample_name=sample_name,
                        sample_idx=start_idx + idx,
                        prompt=original_batch_prompts[idx],
                        prompt_as_path=self.prompt_as_path,
                        num_sample=num_sample,
                        k=k,
                    ) for idx in range(len(batch_prompts))
                ]

                # NOTE: Skip if the sample already exists
                # This is useful for resuming sampling VBench
                if self.prompt_as_path and \
                    all(os.path.exists(path) for path in save_paths):
                    continue

                # == process prompts step by step ==
                # 0. split prompt
                # each element in the list is [prompt_segment_list, loop_idx_list]
                batched_prompt_segment_list = []
                batched_loop_idx_list = []
                for prompt in batch_prompts:
                    prompt_segment_list, loop_idx_list = self.data_processor.split_prompt(
                        prompt)
                    batched_prompt_segment_list.append(prompt_segment_list)
                    batched_loop_idx_list.append(loop_idx_list)

                # [NOTE] Skip refining prompt by OpenAI
                for idx, prompt_segment_list in enumerate(
                        batched_prompt_segment_list):
                    batched_prompt_segment_list[
                        idx] = self.data_processor.append_score_to_prompts(
                            prompt_segment_list,
                            aes=self.aes,
                            flow=self.flow,
                            camera_motion=self.camera_motion,
                        )

                # 3. clean prompt with T5
                for idx, prompt_segment_list in enumerate(
                        batched_prompt_segment_list):
                    batched_prompt_segment_list[idx] = [
                        self.data_processor.text_preprocessing(prompt)
                        for prompt in prompt_segment_list
                    ]

                # 4. merge to obtain the final prompt
                batch_prompts = []
                for prompt_segment_list, loop_idx_list in zip(
                        batched_prompt_segment_list, batched_loop_idx_list):
                    batch_prompts.append(
                        self.data_processor.merge_prompt(
                            prompt_segment_list, loop_idx_list))

                # == Iter over loop generation ==
                video_clips = []
                for loop_i in range(loop):
                    # == get prompt for loop i ==
                    batch_prompts_loop = self.data_processor.extract_prompts_loop(
                        batch_prompts, loop_i)

                    # == add condition frames for loop ==
                    if loop_i > 0:
                        refs, ms = self.data_processor.append_generated(
                            video_clips[-1], refs, ms, loop_i,
                            self.condition_frame_length,
                            self.condition_frame_edit)

                    # == sampling ==
                    torch.manual_seed(self.seed)
                    noise = torch.randn(len(batch_prompts),
                                        self.vae.out_channels,
                                        *self.latent_size,
                                        device=self.device,
                                        dtype=str_dtype_to_torch(self.dtype))
                    masks = self.data_processor.apply_mask_strategy(
                        noise, refs, ms, loop_i, align=self.align)
                    samples = self.sample(
                        latent=noise,
                        prompts=batch_prompts_loop,
                        mask=masks,
                        additional_args=model_args,
                    )
                    samples = self.vae.decode(samples.to(
                        str_dtype_to_torch(self.dtype)),
                                              num_frames=self.num_frames)
                    video_clips.append(samples)

                self.save_video(video_clips, save_paths, len(batch_prompts),
                                loop)
            start_idx += len(batch_prompts)

    def sample(
        self,
        latent,
        prompts,
        mask=None,
        additional_args=None,
        guidance_scale=None,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        # text encoding
        model_args = self.text_encoder.encode_with_null(prompts)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps
                     for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [
            torch.tensor([t] * latent.shape[0], device=self.device)
            for t in timesteps
        ]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(t,
                                   additional_args,
                                   num_timesteps=self.num_timesteps)
                for t in timesteps
            ]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        for i, t in enumerate(timesteps):
            print_progress_bar(i, self.num_sampling_steps)
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = latent.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                latent = torch.where(mask_add_noise[:, None, :, None, None],
                                     x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            latent_in = torch.cat([latent, latent], 0)
            t = torch.cat([t, t], 0)
            for k in list(model_args.keys()):
                if k not in ['y', 'mask', 'x_mask', 'fps', 'height', 'width']:
                    model_args.pop(k)
                else:
                    if model_args[k].dtype == torch.float32:
                        model_args[k] = model_args[k].to(torch.float16)
            latent_in = latent_in.to(torch.float16)
            t = t.to(torch.float16)
            model_args['mask'] = model_args['mask'].to(torch.int32)
            model_args['y_lens'] = self._get_y_lens(model_args['y'],
                                                    model_args['mask'])

            pred = self.stdit.forward(x=latent_in, timestep=t,
                                      **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update latent
            dt = timesteps[i] - timesteps[i + 1] if i < len(
                timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            latent = latent + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                latent = torch.where(mask_t_upper[:, None, :, None, None],
                                     latent, x0)
        print()
        return latent

    def _get_y_lens(self, y, mask=None):
        assert (len(y.shape) == 4)
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
        return y_lens

    def save_video(
        self,
        video_clips,
        save_paths,
        batch_size,
        loop,
    ):
        os.makedirs(self.video_save_dir, exist_ok=True)
        if tensorrt_llm.mpi_rank() == 0:
            for idx in range(batch_size):
                save_path = save_paths[idx]
                video = [video_clips[i][idx] for i in range(loop)]
                for i in range(1, loop):
                    video[i] = video[i][:,
                                        self.data_processor.dframe_to_frame(
                                            self.condition_frame_length):]
                video = torch.cat(video, dim=1)
                save_path = self.data_processor.save_sample(
                    video,
                    fps=self.save_fps,
                    save_path=save_path,
                )
                if save_path.endswith(".mp4") and self.watermark:
                    time.sleep(1)  # prevent loading previous generated video
                    self.data_processor.add_watermark(save_path)
