import math
import os
import shutil
import sys
import tarfile
from time import time

import yaml

# isort: off
import torch
import tensorrt as trt
from pathlib import Path
from tensorrt_llm._utils import torch_dtype_to_str, to_json_file
from tensorrt_llm.builder import Builder
from tensorrt_llm.logger import logger
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForVision2Seq, AutoProcessor,
                          Blip2ForConditionalGeneration, Blip2Processor,
                          FuyuForCausalLM, FuyuProcessor,
                          LlavaForConditionalGeneration, NougatProcessor,
                          Pix2StructForConditionalGeneration,
                          VisionEncoderDecoderModel, CLIPVisionModel)
# isort: on
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_model, save_file
from transformers import CLIPImageProcessor

from ..runtime.session import Session


def add_multimodal_arguments(parser):
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        choices=[
            'blip2', 'llava', 'llava_next', 'llava_onevision',
            'llava_onevision_lmms', 'vila', 'nougat', 'cogvlm', 'fuyu',
            'pix2struct', 'neva', 'kosmos-2', 'video-neva', 'phi-3-vision',
            'phi-4-multimodal', 'mllama', 'internvl', 'qwen2_vl',
            'internlm-xcomposer2', 'qwen2_audio', 'pixtral', 'eclair'
        ],
        help="Model type")
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help=
        "Huggingface repo, local directory with weights or path to checkpoint file"
    )
    parser.add_argument('--vila_path',
                        type=str,
                        default=None,
                        help="Path to VILA source code directory")
    parser.add_argument('--output_dir',
                        type=str,
                        default=None,
                        help="Directory where visual TRT engines are saved")
    parser.add_argument('--max_batch_size',
                        type=int,
                        default=4,
                        help="Maximum batch size for input images")
    parser.add_argument(
        '--max_hw_dims',
        type=int,
        default=5184,
        help=
        "Maximum multiply of h and w after patching for input images for qwen2_vl"
    )
    parser.add_argument(
        '--min_hw_dims',
        type=int,
        default=128,
        help=
        "Minimum multiply of h and w after patching for input images for qwen2_vl"
    )
    parser.add_argument(
        '--num_mul_bins',
        type=int,
        default=128,
        help="Number of Mel frequency bins of input audios for qwen2_audio")
    parser.add_argument(
        '--max_mel_seq_len',
        type=int,
        default=3000,
        help=
        "Maximum Mel frequency feature lengths of input audios for qwen2_audio")
    return parser


class MultimodalEngineBuilder:

    def __init__(self, args):
        args.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        if args.output_dir is None:
            # default path to save the engines
            model_name = args.model_path.split('/')[-1]
            args.output_dir = f'tmp/trt_engines/{model_name}/multimodal_encoder'

        os.makedirs(args.output_dir, exist_ok=True)

        self.args = args

    def build(self):
        args = self.args
        if args.model_type == 'blip2':
            build_blip2_engine(args)
        elif args.model_type == 'internlm-xcomposer2':
            build_interlm_xcomposer2_engine(args)
        elif args.model_type == 'pix2struct':
            build_pix2struct_engine(args)
        elif 'llava' in args.model_type:
            build_llava_engine(args)
        elif args.model_type == 'vila':
            assert args.vila_path is not None, "Please clone and provide VILA source code path"
            build_vila_engine(args)
        elif args.model_type == 'nougat':
            build_nougat_engine(args)
        elif args.model_type == 'cogvlm':
            build_cogvlm_engine(args)
        elif args.model_type == 'fuyu':
            build_fuyu_engine(args)
        elif args.model_type == 'neva':
            build_neva_engine(args)
        elif args.model_type == 'video-neva':
            build_video_neva_engine(args)
        elif args.model_type == 'kosmos-2':
            build_kosmos_engine(args)
        elif args.model_type == 'phi-3-vision':
            build_phi_engine(args)
        elif args.model_type == 'phi-4-multimodal':
            build_phi4mm_engine(args)
        elif args.model_type == 'mllama':
            build_mllama_engine(args)
        elif args.model_type == 'internvl':
            build_internvl_engine(args)
        elif args.model_type == 'qwen2_vl':
            build_qwen2_vl_engine(args)
        elif args.model_type == 'qwen2_audio':
            build_qwen2_audio_engine(args)
        elif args.model_type == "pixtral":
            build_pixtral_engine(args)
        elif args.model_type == "eclair":
            build_eclair_engine(args)
        else:
            raise RuntimeError(f"Invalid model type {args.model_type}")


def export_onnx(model,
                input,
                onnx_dir,
                onnx_name='model.onnx',
                input_names=['input'],
                output_names=['encoder_output'],
                dynamic_axes={'input': {
                    0: 'batch'
                }},
                logger=trt.Logger(trt.Logger.INFO)):
    logger.log(trt.Logger.INFO, f"Exporting onnx to {onnx_dir}/{onnx_name}")
    os.makedirs(onnx_dir, exist_ok=True)

    torch.onnx.export(model,
                      input,
                      f'{onnx_dir}/{onnx_name}',
                      opset_version=17,
                      input_names=input_names,
                      output_names=output_names,
                      dynamic_axes=dynamic_axes)


def build_trt_engine(model_type,
                     input_sizes,
                     onnx_dir,
                     engine_dir,
                     max_batch_size,
                     dtype=torch.float16,
                     model_params=None,
                     onnx_name='model.onnx',
                     engine_name='model.engine',
                     delete_onnx=True,
                     logger=trt.Logger(trt.Logger.INFO)):
    """Build TensorRT engine from ONNX model.

    Args:
        model_params (dict): Optional model specific parameters, e.g.:
            - qwen2_vl_dim (int): Dimension for Qwen2-VL model
            - min_hw_dims (int): Minimum HW dimensions
            - max_hw_dims (int): Maximum HW dimensions
            - num_frames (int): Number of frames for video models
    """
    model_params = model_params or {}
    onnx_file = f'{onnx_dir}/{onnx_name}'
    engine_file = f'{engine_dir}/{engine_name}'
    config_file = f'{engine_dir}/config.json'
    logger.log(trt.Logger.INFO, f"Building TRT engine to {engine_file}")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config_args = {
        "precision": torch_dtype_to_str(dtype),
        "model_type": model_type,
        "strongly_typed": False,
        "max_batch_size": max_batch_size,
        "model_name": "multiModal"
    }

    if "num_frames" in model_params:
        config_args["num_frames"] = model_params["num_frames"]

    config_wrapper = Builder().create_builder_config(**config_args)
    config = config_wrapper.trt_builder_config

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    nBS = -1
    nMinBS = 1
    nOptBS = max(nMinBS, int(max_batch_size / 2))
    nMaxBS = max_batch_size

    # input sizes can be:
    # - integer list, when inputs are constant size images. e.g. [3, H, W]
    # - list of integer lists, when inputs are dynamic size images. e.g. [[1, 1, 2700], [1, 500, 2700], [1, 4096, 2700]]
    # - list of list of integer lists, when there are many inputs and each input have dynamic size. e.g.
    #   [[[1, 1, 2700], [1, 500, 2700], [1, 4096, 2700]], [[1, 1], [1, 1], [1,1]]]
    assert isinstance(input_sizes, list), "input_sizes must be a list"
    if model_type == "qwen2_vl":
        input_images = network.get_input(0)
        inputT = network.get_input(1)
        attenstion_mask = network.get_input(2)

        qwen2_vl_dim = model_params.get('qwen2_vl_dim', 0)
        min_hw_dims = model_params.get('min_hw_dims', 0)
        max_hw_dims = model_params.get('max_hw_dims', 0)

        assert min_hw_dims > 0, "min_hw_dims must be positive for qwen2_vl"
        assert max_hw_dims > 0, "max_hw_dims must be positive for qwen2_vl"

        multi_size_min = min_hw_dims
        multi_size_max = max_hw_dims * max_batch_size
        multi_size_opt = max(multi_size_min, int(multi_size_max / 2))

        inputT.shape = [-1, *input_sizes]
        profile.set_shape(inputT.name, [multi_size_min, *input_sizes],
                          [multi_size_opt, *input_sizes],
                          [multi_size_max, *input_sizes])

        input_images.shape = [-1, qwen2_vl_dim]
        profile.set_shape(input_images.name, [multi_size_min, qwen2_vl_dim],
                          [multi_size_opt, qwen2_vl_dim],
                          [multi_size_max, qwen2_vl_dim])

        attenstion_mask.shape = [1, -1, -1]
        profile.set_shape(attenstion_mask.name,
                          [1, multi_size_min, multi_size_min],
                          [1, multi_size_opt, multi_size_opt],
                          [1, multi_size_max, multi_size_max])
    elif model_type == "qwen2_audio":
        inputT = network.get_input(0)
        mask = network.get_input(1)

        num_mul_bins = model_params.get('num_mul_bins', 0)
        max_mel_seq_len = model_params.get('max_mel_seq_len', 0)

        assert num_mul_bins > 0, "num_mul_bins must be positive for qwen2_audio"
        assert max_mel_seq_len > 0, "max_mel_seq_len must be positive for qwen2_audio"

        inputT.shape = [nBS, num_mul_bins, max_mel_seq_len]
        max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        mask.shape = [nBS, 1, max_seq_len, max_seq_len]

        profile.set_shape(
            inputT.name,
            [nMinBS, num_mul_bins, max_mel_seq_len],
            [nOptBS, num_mul_bins, max_mel_seq_len],
            [nMaxBS, num_mul_bins, max_mel_seq_len],
        )
        profile.set_shape(
            mask.name,
            [nMinBS, 1, max_seq_len, max_seq_len],
            [nOptBS, 1, max_seq_len, max_seq_len],
            [nMaxBS, 1, max_seq_len, max_seq_len],
        )
    else:
        if isinstance(input_sizes[0], int):
            logger.log(trt.Logger.INFO, f"Processed input sizes {input_sizes}")
            inputT = network.get_input(0)
            inputT.shape = [nBS, *input_sizes]
            min_size = opt_size = max_size = input_sizes
            profile.set_shape(inputT.name, [nMinBS, *min_size],
                              [nOptBS, *opt_size], [nMaxBS, *max_size])
        elif isinstance(input_sizes[0], list) and isinstance(
                input_sizes[0][0], list):
            for idx, input_size in enumerate(input_sizes):
                assert len(input_size) == 3
                inputT = network.get_input(idx)
                min_size, opt_size, max_size = input_size
                profile.set_shape(inputT.name, [nMinBS, *min_size],
                                  [nOptBS, *opt_size], [nMaxBS, *max_size])
        elif len(input_sizes) == 3 and isinstance(input_sizes[0], list):
            inputT = network.get_input(0)
            min_size, opt_size, max_size = input_sizes
            profile.set_shape(inputT.name, [nMinBS, *min_size],
                              [nOptBS, *opt_size], [nMaxBS, *max_size])
            logger.log(
                trt.Logger.INFO,
                f"Processed min/opt/max input sizes {min_size}/{opt_size}/{max_size}"
            )
        else:
            raise ValueError(f"invalid input sizes: {input_sizes}")

    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO,
                   "Succeeded building %s in %d s" % (engine_file, t1 - t0))

        logger.log(trt.Logger.INFO, 'Recording engine output shape in config')
        engine_session = Session.from_serialized_engine(engine_string)
        output_tensor_name = network.get_output(0).name
        output_shape = engine_session.engine.get_tensor_shape(
            output_tensor_name)
        output_shape = list(output_shape)
        config_wrapper.output_shape = output_shape

        os.makedirs(engine_dir, exist_ok=True)
        with open(engine_file, 'wb') as f:
            f.write(engine_string)

        # Clear onnx files since we no longer need them after a successful engine build
        if delete_onnx:
            shutil.rmtree(onnx_dir)

    Builder.save_config(config_wrapper, config_file)


def build_blip2_engine(args):
    processor = Blip2Processor.from_pretrained(args.model_path)

    raw_image = Image.new('RGB', [10, 10])  # dummy image
    prompt = "Question: what is this? Answer:"
    inputs = processor(raw_image, prompt,
                       return_tensors="pt").to(args.device, torch.float16)
    image = inputs['pixel_values']

    class Blip2VisionWrapper(torch.nn.Module):

        def __init__(self, vision_model, qformer, projector, query_tokens):
            super().__init__()
            self.vision_model = vision_model
            self.qformer = qformer
            self.projector = projector
            self.query_tokens = query_tokens

        def forward(self, image):
            features = self.vision_model(image)[0]
            qformer_output = self.qformer(query_embeds=self.query_tokens,
                                          encoder_hidden_states=features,
                                          return_dict=True)
            return self.projector(qformer_output.last_hidden_state)

    model = Blip2ForConditionalGeneration.from_pretrained(args.model_path,
                                                          dtype=torch.float16)

    blip2_llm = ""
    if model.language_model.config.architectures[
            0] == 'T5ForConditionalGeneration':
        blip2_llm = "t5"
    elif model.language_model.config.architectures[0] == 'OPTForCausalLM':
        blip2_llm = "opt"

    wrapper = Blip2VisionWrapper(model.vision_model, model.qformer,
                                 model.language_projection, model.query_tokens)
    wrapper.to(args.device)

    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type + "-" + blip2_llm,  # blip2-t5 or blip2-opt
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)


def build_interlm_xcomposer2_engine(args):
    model = AutoModel.from_pretrained(args.model_path,
                                      trust_remote_code=True).to(torch.float16)
    raw_image = Image.new('RGB', [10, 10])
    image = model.vis_processor(raw_image).unsqueeze(0).to(
        args.device, torch.float16)

    class InternLMXComposer2VisionWrapper(torch.nn.Module):

        def __init__(self, vision_model, vision_proj):
            super().__init__()
            self.vision_model = vision_model
            self.vision_proj = vision_proj

        def forward(self, image):
            return self.vision_proj(self.vision_model(image))

    wrapper = InternLMXComposer2VisionWrapper(model.vit, model.vision_proj)
    wrapper.to(args.device)
    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)


def build_pix2struct_engine(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    raw_image = Image.new('RGB', [10, 10])  # dummy image
    dtype = torch.float16
    inputs = processor(text="dummy", images=raw_image, return_tensors="pt")
    image = inputs['flattened_patches'].to(args.device, dtype)

    class pix2structVisionWrapper(torch.nn.Module):

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, image):
            attention_mask = (image.abs().sum(dim=-1) != 0)
            vision_x = self.encoder.embeddings(image)
            img_features = self.encoder.encoder(vision_x,
                                                attention_mask=attention_mask)
            img_features = self.encoder.layernorm(img_features[0])
            return img_features

    model = Pix2StructForConditionalGeneration.from_pretrained(args.model_path,
                                                               dtype=dtype)

    wrapper = pix2structVisionWrapper(model.encoder.to(args.device))
    # input shape: batch size, number of patches, hidden dimension
    # attention mask shape: batch size, number of patches
    # The number of image patches can vary depending on the image size, but it typically
    # falls within a relatively narrow range. To improve performance, we can avoid using
    # dynamic axis for the input patches and instead use a fixed number of patches along
    # with an attention mask.
    export_onnx(wrapper, (image, ),
                f'{args.output_dir}/onnx',
                input_names=['input'],
                dynamic_axes={'input': {
                    0: 'batch'
                }})
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2]],  # Number of Patches, Hidden Dimension
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size,
        dtype=torch.bfloat16)


def build_llava_engine(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    if args.model_type == "llava":
        raw_image = Image.new('RGB', [10, 10])  # dummy image
        image = processor(text="dummy", images=raw_image,
                          return_tensors="pt")['pixel_values'].to(
                              args.device, torch.float16)

        class LlavaVisionWrapper(torch.nn.Module):

            def __init__(self, tower, projector, feature_layer):
                super().__init__()
                self.tower = tower
                self.projector = projector
                self.feature_layer = feature_layer

            def forward(self, image):
                all_hidden_states = self.tower(
                    image, output_hidden_states=True).hidden_states
                features = all_hidden_states[self.feature_layer][:, 1:]
                return self.projector(features)

        hf_config = AutoConfig.from_pretrained(args.model_path)
        hf_config.vision_config._attn_implementation = "eager"
        # Need to setup at hf_config._attn_implementation after transformers >= 4.46
        hf_config._attn_implementation = "eager"
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, dtype=torch.float16, config=hf_config)
        wrapper = LlavaVisionWrapper(
            model.vision_tower.to(args.device),
            model.multi_modal_projector.to(args.device),
            model.config.vision_feature_layer)
    elif args.model_type == "llava_next":
        from transformers import LlavaNextForConditionalGeneration
        raw_image = Image.new('RGB', [512, 512])
        image = processor(text="dummy", images=raw_image,
                          return_tensors="pt")['pixel_values'].to(
                              args.device, torch.float16)[0]

        class LlavaNextVisionWrapper(torch.nn.Module):

            def __init__(self, vision_tower, projector):
                super().__init__()
                self.vision_tower = vision_tower
                self.projector = projector

            def forward(self, pixel_values):
                image_features = self.vision_tower(pixel_values,
                                                   output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[-2][:, 1:]
                image_features = self.projector(selected_image_feature)
                return image_features  # (bs, 576, c)

        hf_config = AutoConfig.from_pretrained(args.model_path)
        hf_config.vision_config._attn_implementation = "eager"
        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, dtype=torch.float16, config=hf_config)
        wrapper = LlavaNextVisionWrapper(
            model.vision_tower.vision_model.to(args.device),
            model.multi_modal_projector.to(args.device),
        )
    elif args.model_type == "llava_onevision_lmms":
        from llava.mm_utils import process_images
        from llava.model.builder import load_pretrained_model
        _, model, processor, _ = load_pretrained_model(args.model_path,
                                                       None,
                                                       args.model_type,
                                                       torch_dtype="float16")
        raw_image = Image.new('RGB', [512, 512])
        image = process_images([raw_image], processor,
                               model.config).squeeze(0).to(
                                   args.device, torch.float16)

        class LlavaQwenVisionWrapper(torch.nn.Module):

            def __init__(self, vision_tower, projector):
                super().__init__()
                self.vision_tower = vision_tower
                self.projector = projector

            def forward(self, pixel_values):
                image_features = self.vision_tower(pixel_values)
                image_features = self.projector(image_features)
                return image_features  # (sigma(bs, patches_i), 729, c)

        wrapper = LlavaQwenVisionWrapper(model.get_model().get_vision_tower(),
                                         model.get_model().mm_projector)
    elif args.model_type == "llava_onevision":
        from transformers import LlavaOnevisionForConditionalGeneration
        raw_image = Image.new('RGB', [512, 512])
        image = processor(text="dummy", images=raw_image,
                          return_tensors="pt")['pixel_values'].to(
                              args.device, torch.float16)[0]

        class LlavaOnevisionVisionWrapper(torch.nn.Module):

            def __init__(self, vision_tower, projector, config):
                super().__init__()
                self.vision_tower = vision_tower
                self.projector = projector
                self.config = config

            def forward(self, pixel_values):
                image_features = self.vision_tower(pixel_values,
                                                   output_hidden_states=True)
                selected_image_feature = image_features.hidden_states[
                    self.config.vision_feature_layer]
                image_features = self.projector(selected_image_feature)
                return image_features  # (sigma(bs, patches_i), 729, c)

        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            args.model_path, dtype=torch.float16)
        wrapper = LlavaOnevisionVisionWrapper(
            model.vision_tower.vision_model.to(args.device),
            model.multi_modal_projector.to(args.device), model.config)

    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)
    if args.model_type == "llava_next":
        image_newline = model.model.image_newline.data
        tensor_img_newline = {"image_newline": image_newline}
        save_file(tensor_img_newline,
                  os.path.join(args.output_dir, "image_newlines.safetensors"))
    if args.model_type == "llava_onevision":
        image_newline = model.model.image_newline.data
        tensor_img_newline = {"image_newline": image_newline}
        save_file(tensor_img_newline,
                  os.path.join(args.output_dir, "image_newlines.safetensors"))
    if args.model_type == "llava_onevision_lmms":
        image_newline = model.model.image_newline.data
        tensor_img_newline = {"image_newline": image_newline}
        save_file(tensor_img_newline,
                  os.path.join(args.output_dir, "image_newlines.safetensors"))


def build_vila_engine(args):
    # Note: VILA model is not in public HF model zoo yet. We need to explicitly import from the git repo
    sys.path.append(args.vila_path)
    from llava.model import LlavaLlamaConfig, LlavaLlamaModel  # noqa
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        args.model_path,
        device_map='auto',
    )

    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    raw_image = Image.new('RGB', [10, 10])  # dummy image
    image = image_processor(images=raw_image,
                            return_tensors="pt")['pixel_values']
    if isinstance(image, list):
        image = image[0].unsqueeze(0)
    image = image.to(args.device, torch.float16)

    class VilaVisionWrapper(torch.nn.Module):

        def __init__(self, tower, projector):
            super().__init__()
            self.tower = tower
            self.projector = projector

        def forward(self, image):
            features = self.tower(image)
            return self.projector(features)

    model = AutoModel.from_pretrained(
        args.model_path,
        device_map='auto',
    )
    wrapper = VilaVisionWrapper(model.get_vision_tower().to(args.device),
                                model.mm_projector.to(args.device))
    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)


def build_nougat_engine(args):
    processor = NougatProcessor.from_pretrained(args.model_path)
    raw_image = Image.new('RGB', [10, 10])  # dummy image
    image = processor(raw_image, return_tensors="pt")['pixel_values'].to(
        args.device, torch.float16)

    class SwinEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, image):
            return self.encoder(image).last_hidden_state

    model = VisionEncoderDecoderModel.from_pretrained(args.model_path,
                                                      dtype=torch.float16)
    swin_encoder = model.get_encoder().to(args.device)
    wrapper = SwinEncoderWrapper(swin_encoder)

    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)


def build_cogvlm_engine(args):
    hf_config = AutoConfig.from_pretrained(args.model_path,
                                           trust_remote_code=True)
    image_size = hf_config.vision_config['image_size']
    dtype = hf_config.torch_dtype
    image = torch.empty(1,
                        3,
                        image_size,
                        image_size,
                        dtype=dtype,
                        device=args.device)  # dummy image

    class CogVlmVisionWrapper(torch.nn.Module):

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, image):
            return self.encoder(image)

    cogvlm = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                  dtype=dtype,
                                                  trust_remote_code=True)
    vit_encoder = cogvlm.model.vision.to(args.device).eval()

    wrapper = CogVlmVisionWrapper(vit_encoder)
    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size,
        dtype=dtype)


def build_fuyu_engine(args):
    processor = FuyuProcessor.from_pretrained(args.model_path)
    raw_image = Image.new('RGB', [10, 10])
    image = processor(text="dummy", images=raw_image,
                      return_tensors="pt")['image_patches'][0].to(
                          args.device, torch.float16).unsqueeze(0)

    class FuyuEncoderWrapper(torch.nn.Module):

        def __init__(self, linear):
            super().__init__()
            self.linear = linear.to(torch.float16)

        def forward(self, patches):
            return self.linear(patches).flatten(0, 1)

    model = FuyuForCausalLM.from_pretrained(args.model_path,
                                            dtype=torch.float16)

    vision_encoder = model.vision_embed_tokens
    wrapper = FuyuEncoderWrapper(vision_encoder).to(args.device)

    export_onnx(wrapper,
                image,
                f'{args.output_dir}/onnx',
                dynamic_axes={'input': {
                    0: 'batch',
                    2: 'patch'
                }})
    build_trt_engine(
        args.model_type,
        # [nImgs, nImgPatches, nDims]
        # nImgs is always one since each query has exactly one image
        # nImgPatches depends on image size (patch size: 30x30)
        # nDims is 30x30x3=2700 (patch size x color channels)
        [[1, 1, 2700], [1, 500, 2700], [1, 4096, 2700]],
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)


def build_neva_engine(args):
    # extract NeMo checkpoint
    with tarfile.open(args.model_path) as tar:
        nemo_config = yaml.safe_load(tar.extractfile("./model_config.yaml"))
        try:
            # trained without TP
            mp0_weights = torch.load(tar.extractfile("./model_weights.ckpt"),
                                     map_location=args.device)
        except KeyError:
            # trained with TP
            mp0_weights = torch.load(
                tar.extractfile("./mp_rank_00/model_weights.ckpt"),
                map_location=args.device)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            vision_x = self.encoder(pixel_values=images,
                                    output_hidden_states=True)
            vision_x = vision_x.hidden_states[-2]
            vision_x = vision_x[:, 1:]
            vision_x = self.connector(vision_x)
            return vision_x

    vision_path = vision_config["from_pretrained"]
    joined_path = os.path.join(os.path.dirname(args.model_path),
                               os.path.basename(vision_path))
    if os.path.isdir(joined_path):
        vision_path = joined_path
    encoder = AutoModel.from_pretrained(vision_path,
                                        dtype=torch.bfloat16,
                                        trust_remote_code=True)
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    assert nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "mlp2x_gelu"
    vision_connector = torch.nn.Sequential(
        torch.nn.Linear(vision_config["hidden_size"],
                        nemo_config["hidden_size"],
                        bias=True), torch.nn.GELU(),
        torch.nn.Linear(nemo_config["hidden_size"],
                        nemo_config["hidden_size"],
                        bias=True)).to(dtype=dtype)

    key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
    for layer in range(0, 3, 2):
        vision_connector[layer].load_state_dict({
            'weight':
            mp0_weights[f"{key_prefix}.{layer}.weight"].to(dtype),
            'bias':
            mp0_weights[f"{key_prefix}.{layer}.bias"].to(dtype),
        })

    # export the whole wrapper
    wrapper = VisionEncoderWrapper(vision_encoder,
                                   vision_connector).to(args.device, dtype)
    image_size = hf_config.vision_config.image_size
    dummy_image = torch.empty(
        1, 3, image_size, image_size, dtype=dtype,
        device=args.device)  # dummy image shape [B, C, H, W]
    export_onnx(wrapper, dummy_image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [3, image_size, image_size],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size,
        dtype=dtype)


def build_video_neva_engine(args):
    # extract NeMo checkpoint
    with tarfile.open(args.model_path) as tar:
        nemo_config = yaml.safe_load(tar.extractfile("./model_config.yaml"))
        try:
            # trained without TP
            mp0_weights = torch.load(tar.extractfile("./model_weights.ckpt"),
                                     map_location=args.device)
        except KeyError:
            # trained with TP
            mp0_weights = torch.load(
                tar.extractfile("./mp_rank_00/model_weights.ckpt"),
                map_location=args.device)

    vision_config = nemo_config["mm_cfg"]["vision_encoder"]

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            b, num_frames, c, h, w = images.shape
            images = images.view(b * num_frames, c, h, w)
            vision_x = self.encoder(
                pixel_values=images,  #[(B num_frames), C, H, W]
                output_hidden_states=True)
            vision_x = vision_x.hidden_states[-2]
            vision_x = vision_x[:, 1:]

            # reshape back to [B, num_frames, img_size, hidden_size]
            vision_x = vision_x.view(b, num_frames, -1, vision_x.shape[-1])

            vision_x = self.connector(vision_x)
            return vision_x

    encoder = AutoModel.from_pretrained(vision_config["from_pretrained"],
                                        dtype=torch.bfloat16,
                                        trust_remote_code=True,
                                        attn_implementation="eager")
    vision_encoder = encoder.vision_model
    hf_config = encoder.config
    dtype = hf_config.torch_dtype

    # connector
    assert nemo_config["mm_cfg"]["mm_mlp_adapter_type"] == "linear"
    vision_connector = torch.nn.Linear(vision_config["hidden_size"],
                                       nemo_config["hidden_size"],
                                       bias=True)

    key_prefix = "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector"
    vision_connector.load_state_dict({
        'weight':
        mp0_weights[f"{key_prefix}.weight"].to(dtype),
        'bias':
        mp0_weights[f"{key_prefix}.bias"].to(dtype),
    })

    # export the whole wrapper
    wrapper = VisionEncoderWrapper(vision_encoder,
                                   vision_connector).to(args.device, dtype)
    image_size = hf_config.vision_config.image_size
    num_frames = nemo_config['data']['num_frames']
    dummy_video = torch.empty(1,
                              num_frames,
                              3,
                              image_size,
                              image_size,
                              dtype=dtype,
                              device=args.device)  # dummy image
    export_onnx(wrapper, dummy_video, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [num_frames, 3, image_size, image_size],  # [num_frames, 3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size,
        dtype=dtype,
        model_params={'num_frames': num_frames})


def build_kosmos_engine(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    raw_image = Image.new('RGB', [10, 10])  # dummy image
    image = processor(text="dummy", images=raw_image,
                      return_tensors="pt")['pixel_values'].to(
                          args.device, torch.float16)

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, encoder, connector):
            super().__init__()
            self.encoder = encoder
            self.connector = connector

        def forward(self, images):
            vision_x = self.encoder(images, output_hidden_states=True)
            img_features = self.encoder.model.post_layernorm(
                vision_x.last_hidden_state)
            img_features = F.normalize(img_features, dim=-1)
            img_features, _ = self.connector(img_features)
            return img_features

    model = AutoModelForVision2Seq.from_pretrained(args.model_path,
                                                   dtype=torch.float16)
    wrapper = VisionEncoderWrapper(
        model.vision_model.to(args.device),
        model.image_to_text_projection.to(args.device))

    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)


def build_phi_engine(args):
    logger.warning(
        "Skipping TRT engine build for Phi-3 vision encoder.  MultimodalModelRunner will use PyTorch vision encoder. Flash/SDPA attention in CLIP encoder is not compatible with torch.onnx.export and eager attention is unstable in PyTorch."
    )

    # Dump config.json needed by model runner
    config_args = {
        "builder_config": {
            "precision": torch_dtype_to_str(torch.float16),
            "model_type": "phi-3-vision",
        }
    }
    to_json_file(config_args, args.output_dir + "/config.json")
    return

    processor = AutoProcessor.from_pretrained(args.model_path,
                                              trust_remote_code=True,
                                              num_crops=16)
    raw_image = Image.new('RGB', [10, 10])  # dummy image
    image = processor(text="<|image_1|>\ndummy",
                      images=raw_image,
                      return_tensors="pt")['pixel_values'].to(
                          args.device, torch.float16)
    image = image.flatten(0, 1)

    class Phi3VisionWrapper(torch.nn.Module):

        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model

        def forward(self, pixel_values):
            return self.vision_model.get_img_features(pixel_values).reshape(
                1, pixel_values.shape[0], -1, self.vision_model.image_dim_out)

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 dtype=torch.float16,
                                                 trust_remote_code=True)
    vision_model = model.model.vision_embed_tokens

    # Replace img_processor that uses flash attention with eager attention
    clip_config = vision_model.img_processor.config
    clip_config._attn_implementation = 'eager'
    del vision_model.img_processor
    vision_model.img_processor = CLIPVisionModel(clip_config).to(torch.float16)

    vision_model = vision_model.to(args.device)
    wrapper = Phi3VisionWrapper(vision_model)

    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    num_crops = processor.image_processor.num_crops
    build_trt_engine(args.model_type,
                     [image.shape[1], image.shape[2], image.shape[3]],
                     f'{args.output_dir}/onnx', args.output_dir,
                     args.max_batch_size * (num_crops + 1))


def build_phi4mm_engine(args):
    logger.warning(
        "Skipping TRT engine build for Phi-4-multimodal encoder.  MultimodalModelRunner will use PyTorch vision & audio encoder. Flash/SDPA attention in CLIP encoder is not compatible with torch.onnx.export and eager attention is unstable in PyTorch."
    )

    # Dump config.json needed by model runner
    config_args = {
        "builder_config": {
            "precision": torch_dtype_to_str(torch.float16),
            "model_type": "phi-4-multimodal",
        }
    }
    os.makedirs(os.path.join(args.output_dir, "vision"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "audio"), exist_ok=True)
    to_json_file(config_args,
                 os.path.join(args.output_dir, "vision", "config.json"))
    to_json_file(config_args,
                 os.path.join(args.output_dir, "audio", "config.json"))
    return

    # Following code works ok with eager mode attention. Leaving it here so that it could
    # be used once issues in torch / onnx mentioned above resolved.
    processor = AutoProcessor.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    raw_image = Image.new('RGB', [10, 10])  # dummy image

    import numpy as np
    audio_feature_size = 500
    audio_compression_rate = 8
    audio_sampling_rate = 16000
    audio_len = int((audio_feature_size * audio_compression_rate + 2) *
                    audio_sampling_rate / 100)
    raw_audio = (np.zeros(audio_len), audio_sampling_rate)  # dummy audio

    inputs = processor(text="<|image_1|><|audio_1|>\ndummy",
                       images=[raw_image],
                       audios=[raw_audio],
                       return_tensors="pt")

    img_embeds = inputs['input_image_embeds'].to(args.device, torch.float16)
    img_attention_mask = inputs['image_attention_mask'].to(
        args.device, torch.bool)
    img_embeds = img_embeds.flatten(0, 1)  # (2, 3, 448, 448)
    img_attention_mask = img_attention_mask.flatten(0, 1)  # (2, 32, 32)

    aud_embeds = inputs['input_audio_embeds'].to(args.device,
                                                 torch.float16)  # (1, 4000, 80)
    aud_len, aud_dim = aud_embeds.shape[1:]
    aud_embeds = torch.cat(
        [aud_embeds,
         aud_embeds.new_zeros(1, 4000 - aud_len, aud_dim)], dim=1)
    aud_attention_mask = torch.ones(1, aud_embeds.shape[1]).to(
        args.device, torch.bool)
    aud_attention_mask[0, aud_len:] = 0

    class Phi4VisionWrapper(torch.nn.Module):

        def __init__(self, vision_model):
            super().__init__()
            self.vision_model = vision_model

        @torch.no_grad
        def forward(self, img_embeds, attention_mask):
            features = self.vision_model.get_img_features(
                img_embeds, attention_mask)
            return self.vision_model.img_projection(features)

    class Phi4AudioWrapper(torch.nn.Module):

        def __init__(self, audio_model):
            super().__init__()
            self.audio_model = audio_model

        @torch.no_grad
        def forward(self, aud_embeds, attention_mask):
            features, _ = self.audio_model.encoder(aud_embeds, attention_mask)
            speech_out = self.audio_model.audio_projection['speech'](features)
            vision_out = self.audio_model.audio_projection['vision'](features)
            return torch.cat((speech_out, vision_out), dim=-1)

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 dtype='auto',
                                                 trust_remote_code=True)

    vision_model = model.model.embed_tokens_extend.image_embed
    vision_model = vision_model.to(args.device, torch.float16)
    vision_model.eval()
    vision_wrapper = Phi4VisionWrapper(vision_model)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    part_name = 'vision'
    onnx_dir = f"{args.output_dir}/{part_name}/onnx"

    export_onnx(vision_wrapper,
                input=(img_embeds, img_attention_mask),
                onnx_dir=onnx_dir,
                input_names=['input', 'attention_mask'],
                dynamic_axes={
                    'input': {
                        0: "batch"
                    },
                    'attention_mask': {
                        0: "batch"
                    }
                })
    build_trt_engine(
        args.model_type,
        input_sizes=[[list(img_embeds.shape[1:]) for _ in range(3)],
                     [list(img_attention_mask.shape[1:]) for _ in range(3)]],
        onnx_dir=onnx_dir,
        engine_dir=f"{args.output_dir}/{part_name}",
        max_batch_size=args.max_batch_size,
        engine_name=f"visual_encoder.engine",
        dtype=torch.float16)

    audio_model = model.model.embed_tokens_extend.audio_embed
    audio_model = audio_model.to(args.device, torch.float16)
    audio_model.eval()
    audio_wrapper = Phi4AudioWrapper(audio_model)

    part_name = 'audio'
    onnx_dir = f"{args.output_dir}/{part_name}/onnx"

    export_onnx(audio_wrapper,
                input=(aud_embeds, aud_attention_mask),
                onnx_dir=onnx_dir,
                input_names=['input', 'attention_mask'],
                dynamic_axes={
                    'input': {
                        0: "batch"
                    },
                    'attention_mask': {
                        0: 'batch'
                    }
                })
    build_trt_engine(
        args.model_type,
        input_sizes=[[list(aud_embeds.shape[1:]) for _ in range(3)],
                     [list(aud_attention_mask.shape[1:]) for _ in range(3)]],
        onnx_dir=onnx_dir,
        engine_dir=f"{args.output_dir}/{part_name}",
        max_batch_size=args.max_batch_size,
        engine_name=f"audio_encoder.engine",
        dtype=torch.float16)


def build_mllama_engine(args):

    class MLLaMAVisionWrapper(torch.nn.Module):

        def __init__(self, vision_model, output_proj):
            super().__init__()
            self.vision_model = vision_model
            self.output_proj = output_proj

        def forward(self, pixel_values, aspect_ratio_ids, aspect_ratio_mask):
            out = self.vision_model(pixel_values, aspect_ratio_ids,
                                    aspect_ratio_mask).last_hidden_state
            out = self.output_proj(out)
            return out

    processor = AutoProcessor.from_pretrained(args.model_path)
    # MllamaForConditionalGeneration requires transformers >= 4.45, which is
    # conflict with limitation of other multimodal models.
    from transformers import MllamaForConditionalGeneration
    model = MllamaForConditionalGeneration.from_pretrained(args.model_path,
                                                           dtype='auto',
                                                           device_map='auto')

    # Check if the model structure is updated to transformers >= 4.52.0
    if hasattr(model, 'model') and hasattr(model.model, 'vision_model'):
        vision_model = model.model.vision_model
        multi_modal_projector = model.model.multi_modal_projector
    else:
        # transformers < 4.52.0
        vision_model = model.vision_model
        multi_modal_projector = model.multi_modal_projector

    wrapper = MLLaMAVisionWrapper(vision_model, multi_modal_projector)

    model_dtype = model.dtype
    image = Image.new('RGB', [2048, 2688])  # dummy image
    inputs = processor(images=image,
                       return_tensors="pt").to(model_dtype).to(model.device)

    # inputs["pixel_values"]: torch.Size([1, 1, 4, 3, 448, 448])
    # inputs["aspect_ratio_ids"]: torch.Size([1, 1])
    # inputs["aspect_ratio_mask"]: torch.Size([1, 1, 4])
    export_onnx(
        wrapper,
        input=tuple([value for key, value in inputs.items()]),
        onnx_dir=f'{args.output_dir}/onnx',
        input_names=[key for key in inputs],
        output_names=['encoder_output'],
        dynamic_axes={key: {
            0: "batch"
        }
                      for key in inputs},
    )

    build_trt_engine(
        args.model_type,
        [[list(inputs[key].shape[1:]) for _ in range(3)] for key in inputs],
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size,
        model_dtype,
    )


def build_internvl_engine(args):
    raw_image = Image.new('RGB', [10, 10])  # Dummy image
    if 'InternVL2-26B' in args.model_path:
        image_processor = AutoProcessor.from_pretrained(
            'OpenGVLab/InternViT-6B-448px-V1-5')
    else:
        image_processor = CLIPImageProcessor.from_pretrained(
            'OpenGVLab/InternViT-300M-448px')
    image = image_processor(images=raw_image, return_tensors='pt').pixel_values
    image = image.to(args.device, torch.float16)

    class InternvlVisionWrapper(torch.nn.Module):

        def __init__(self, model, downsample_ratio=0.5, layer_idx=-1):
            super().__init__()
            self.vision_model = model.vision_model
            self.mlp1 = model.mlp1
            self.downsample_ratio = downsample_ratio
            self.layer_idx = layer_idx

        def pixel_shuffle(self, x, scale_factor=0.5):
            n, w, h, c = x.size()
            # N, W, H, C --> N, W, H * scale, C // scale
            x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
            # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
            x = x.permute(0, 2, 1, 3).contiguous()
            # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
            x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                       int(c / (scale_factor * scale_factor)))

            x = x.permute(0, 2, 1, 3).contiguous()
            return x

        def forward(self, image):
            immde_res = self.vision_model(image, output_hidden_states=True)
            vit_embeds = immde_res.hidden_states[self.layer_idx]
            vit_embeds = vit_embeds[:, 1:, :]
            h = w = int(vit_embeds.shape[1]**0.5)
            vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
            vit_embeds_px = self.pixel_shuffle(
                vit_embeds, scale_factor=self.downsample_ratio)
            vit_embeds_px = vit_embeds_px.reshape(vit_embeds_px.shape[0], -1,
                                                  vit_embeds_px.shape[-1])
            vit_embeds_mlp = self.mlp1(vit_embeds_px)
            return vit_embeds_mlp

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 dtype=torch.float16,
                                                 trust_remote_code=True,
                                                 use_flash_attn=False).to(
                                                     args.device)
    max_num_crops = model.config.max_dynamic_patch
    wrapper = InternvlVisionWrapper(model, model.config.downsample_ratio,
                                    model.config.select_layer)

    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(args.model_type,
                     [image.shape[1], image.shape[2], image.shape[3]],
                     f'{args.output_dir}/onnx', args.output_dir,
                     args.max_batch_size * max_num_crops)


def compute_rotary_pos_emb(grid_thw, hf_config, VisionRotaryEmbedding):
    head_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads
    rotary_pos_emb_func = VisionRotaryEmbedding(head_dim // 2)
    hf_config.vision_config.spatial_merge_size

    def rot_pos_emb(grid_thw, rotary_pos_emb_func):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
                w // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
                w // hf_config.vision_config.spatial_merge_size,
                hf_config.vision_config.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(
                torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = rotary_pos_emb_func(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    rotary_pos_emb = rot_pos_emb(grid_thw, rotary_pos_emb_func)
    return rotary_pos_emb


def build_qwen2_vl_engine(args):
    import transformers
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from transformers.models.qwen2_vl.configuration_qwen2_vl import \
        Qwen2VLVisionConfig
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        Qwen2VisionTransformerPretrainedModel, Qwen2VLVisionBlock,
        VisionAttention, VisionRotaryEmbedding)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager")
    hf_config = AutoConfig.from_pretrained(args.model_path)
    qwen2_vl_dim = hf_config.vision_config.in_chans * hf_config.vision_config.patch_size * hf_config.vision_config.patch_size * hf_config.vision_config.temporal_patch_size
    processor = AutoProcessor.from_pretrained(args.model_path)
    messages = [{
        "role":
        "user",
        "content": [
            {
                "type":
                "image",
                "image":
                "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type": "text",
                "text": "Describe this picture?"
            },
        ],
    }]
    text = processor.apply_chat_template(messages,
                                         tokenize=False,
                                         add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    for i in range(len(image_inputs)):
        image_inputs[i] = image_inputs[i].resize(
            (image_inputs[i].size[0] // 2, image_inputs[i].size[1] // 2))
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs
    image = inputs['pixel_values'].to(torch.float16)
    image_grid_thw = inputs['image_grid_thw']
    cu_seqlens = torch.repeat_interleave(
        image_grid_thw[:, 1] * image_grid_thw[:, 2],
        image_grid_thw[:, 0]).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    seq_length = image.shape[0]
    attention_mask = torch.full([1, seq_length, seq_length],
                                torch.finfo(image.dtype).min,
                                device=image.device,
                                dtype=image.dtype)
    for i in range(1, len(cu_seqlens)):
        attention_mask[..., cu_seqlens[i - 1]:cu_seqlens[i],
                       cu_seqlens[i - 1]:cu_seqlens[i]] = 0
    rotary_pos_emb = compute_rotary_pos_emb(image_grid_thw, hf_config,
                                            VisionRotaryEmbedding)

    class VisionAttentionOpt(VisionAttention):

        def __init__(self, config: Qwen2VLVisionConfig):
            # Fallback for compatibility with older transformers versions (for certain nvbugs/tests)
            if transformers.__version__ >= '4.53.0':
                super().__init__(config)
                self.head_dim = config.embed_dim // config.num_heads
            else:
                num_heads = config.num_heads
                dim = config.embed_dim
                super().__init__(dim, num_heads)
                self.head_dim = dim // num_heads

        def forward(self,
                    hidden_states: torch.Tensor,
                    attention_mask: torch.Tensor,
                    rotary_pos_emb: torch.Tensor = None) -> torch.Tensor:
            seq_length = hidden_states.shape[0]
            q, k, v = self.qkv(hidden_states).reshape(seq_length, 3,
                                                      self.num_heads,
                                                      -1).permute(1, 0, 2,
                                                                  3).unbind(0)

            # Copied from transformers.models.llama.modeling_qwen2_vl in v4.48
            def rotate_half(x):
                x1 = x[..., :x.shape[-1] // 2]
                x2 = x[..., x.shape[-1] // 2:]
                return torch.cat((-x2, x1), dim=-1)

            def apply_rotary_pos_emb_vision(
                    tensor: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
                orig_dtype = tensor.dtype
                tensor = tensor.float()
                cos = freqs.cos()
                sin = freqs.sin()
                cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
                sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
                output = (tensor * cos) + (rotate_half(tensor) * sin)
                output = output.to(orig_dtype)
                return output

            q = apply_rotary_pos_emb_vision(q.unsqueeze(0),
                                            rotary_pos_emb).squeeze(0)
            k = apply_rotary_pos_emb_vision(k.unsqueeze(0),
                                            rotary_pos_emb).squeeze(0)
            q = q.transpose(0, 1)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            attn_weights = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(
                self.head_dim)
            attn_weights = attn_weights + attention_mask
            attn_weights = nn.functional.softmax(attn_weights,
                                                 dim=-1,
                                                 dtype=torch.float32).to(
                                                     q.dtype)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(0, 1)
            attn_output = attn_output.reshape(seq_length, -1)
            attn_output = self.proj(attn_output)
            return attn_output

    class Qwen2VLVisionBlockOpt(Qwen2VLVisionBlock):

        def __init__(self, config, attn_implementation: str = "eager") -> None:
            super().__init__(config)
            self.attn = VisionAttentionOpt(config)

        def forward(self, hidden_states, attention_mask,
                    rotary_pos_emb) -> torch.Tensor:
            hidden_states = hidden_states + self.attn(
                self.norm1(hidden_states),
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb)
            hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
            return hidden_states

    class Qwen2VisionTransformerPretrainedModelOpt(
            Qwen2VisionTransformerPretrainedModel):

        def __init__(self, config) -> None:
            super().__init__(config)
            self.blocks = nn.ModuleList([
                Qwen2VLVisionBlockOpt(config, config._attn_implementation)
                for _ in range(config.depth)
            ])

        def forward(self, hidden_states: torch.Tensor,
                    rotary_pos_emb: torch.Tensor,
                    attention_mask: torch.Tensor) -> torch.Tensor:
            hidden_states = self.patch_embed(hidden_states)
            for blk in self.blocks:
                hidden_states = blk(hidden_states,
                                    attention_mask=attention_mask,
                                    rotary_pos_emb=rotary_pos_emb)
            res = self.merger(hidden_states)
            return res

    class VisionEncoderWrapper(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.visual = Qwen2VisionTransformerPretrainedModelOpt._from_config(
                model.config.vision_config,
                dtype=torch.float32,
            )
            self.visual.load_state_dict(model.visual.state_dict())

        def forward(self, images, rotary_pos_emb, attention_mask):
            img_features = self.visual(images, rotary_pos_emb, attention_mask)
            return img_features

    wrapper = VisionEncoderWrapper(model)
    dynamic_axes = {
        'input': {
            0: 'hw'
        },
        'rotary_pos_emb': {
            0: 'hw'
        },
        'attention_mask': {
            1: 'hw',
            2: 'hw'
        }
    }
    export_onnx(wrapper, (image, rotary_pos_emb, attention_mask),
                f'{args.output_dir}/onnx',
                input_names=['input', 'rotary_pos_emb', 'attention_mask'],
                output_names=['encoder_output'],
                dynamic_axes=dynamic_axes)
    rotary_pos_emb_dim = hf_config.vision_config.embed_dim // hf_config.vision_config.num_heads // 2
    build_trt_engine(args.model_type, [rotary_pos_emb_dim],
                     f'{args.output_dir}/onnx',
                     args.output_dir,
                     args.max_batch_size,
                     model_params={
                         'qwen2_vl_dim': qwen2_vl_dim,
                         'min_hw_dims': args.min_hw_dims,
                         'max_hw_dims': args.max_hw_dims
                     })


def build_qwen2_audio_engine(args):
    from transformers import Qwen2AudioForConditionalGeneration

    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_path, dtype=torch.float16)

    # dummy audio features, dtype is float32
    audio = torch.randn(1,
                        args.num_mul_bins,
                        args.max_mel_seq_len,
                        device=args.device)

    max_seq_len = (args.max_mel_seq_len - 2) // 2 + 1
    mask = torch.zeros((audio.size(0), 1, max_seq_len, max_seq_len),
                       device=args.device,
                       dtype=torch.float16)

    class AudioEncoderWrapper(torch.nn.Module):

        def __init__(self, audio_tower, multi_modal_projector):
            super(AudioEncoderWrapper, self).__init__()
            self.audio_tower = audio_tower
            self.multi_modal_projector = multi_modal_projector

        def forward(self, x, mask):
            audio_outputs = self.audio_tower(x, attention_mask=mask)
            selected_audio_feature = audio_outputs.last_hidden_state
            audio_features = self.multi_modal_projector(selected_audio_feature)
            return audio_features

    wrapper = AudioEncoderWrapper(model.audio_tower,
                                  model.multi_modal_projector)
    wrapper.eval().to(args.device)
    del model  # To save memory

    dynamic_axes = {
        "input": {
            0: "batch"
        },
        "mask": {
            0: "batch"
        },
        "output": {
            0: "batch"
        },
    }
    export_onnx(wrapper, (audio, mask),
                f'{args.output_dir}/onnx',
                input_names=["input", "mask"],
                output_names=["output"],
                dynamic_axes=dynamic_axes)

    build_trt_engine(args.model_type, [],
                     f'{args.output_dir}/onnx',
                     args.output_dir,
                     args.max_batch_size,
                     model_params={
                         'num_mul_bins': args.num_mul_bins,
                         'max_mel_seq_len': args.max_mel_seq_len
                     })


def build_pixtral_engine(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
    hf_config = AutoConfig.from_pretrained(args.model_path)
    vision_config = hf_config.vision_config
    raw_image = Image.new(
        'RGB',
        [vision_config.image_size, vision_config.image_size])  # dummy image

    inputs = processor(text="dummy", images=[raw_image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(args.device, torch.bfloat16)
    attention_mask = torch.zeros(
        1, vision_config.image_size // vision_config.patch_size,
        vision_config.image_size // vision_config.patch_size).to(
            args.device, torch.bfloat16)

    # isort: off
    from transformers.models.pixtral.modeling_pixtral import \
        apply_rotary_pos_emb
    from transformers import Mistral3ForConditionalGeneration
    from transformers.models.pixtral.modeling_pixtral import (PixtralAttention,
                                                              PixtralVisionModel
                                                              )
    from transformers.models.mistral3.modeling_mistral3 import (
        Mistral3MultiModalProjector, Mistral3PatchMerger)
    # isort: on
    @torch.no_grad
    def attn_forward(self,
                     hidden_states,
                     attention_mask,
                     position_embeddings,
                     output_attentions=False):
        batch, patches, _ = hidden_states.size()

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(batch, patches, self.num_heads,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch, patches, self.num_heads,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch, patches, self.num_heads,
                   self.head_dim).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=0)

        # attention_mask is of shape [batch, patches].
        mask = attention_mask[:, None, None, :]

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask).transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(batch, patches, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    @torch.no_grad
    def vision_tower_forward(self, pixel_values, attention_mask):
        patch_embeds = self.patch_conv(pixel_values)  # (bs, c, h, w)

        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # (bs, h*w, c)
        attention_mask = attention_mask.flatten(1)  # (bs, h*w)

        patch_embeds = self.ln_pre(patch_embeds)
        position_ids = self.position_ids.flatten()  # (h*w, )
        position_embeddings = self.patch_positional_embedding(
            patch_embeds, position_ids)

        out = self.transformer(patch_embeds,
                               attention_mask=attention_mask,
                               position_embeddings=position_embeddings,
                               output_hidden_states=False,
                               output_attentions=False,
                               return_dict=False)[0]
        return out

    @torch.no_grad
    def patch_merger_forward(self, image_features, attention_mask):
        h, w = attention_mask.shape[-2:]
        bs, n, d = image_features.shape
        image_grid = image_features.view(bs, h, w, d).permute(0, 3, 1, 2)
        image_features = torch.nn.functional.unfold(image_grid, 2,
                                                    stride=2).transpose(1, 2)
        image_features = self.merging_layer(image_features)
        return image_features

    @torch.no_grad
    def mm_projector_forward(self, image_features, attention_mask):
        image_features = self.norm(image_features)
        image_features = self.patch_merger(image_features, attention_mask)
        hidden_states = self.linear_2(self.act(self.linear_1(image_features)))
        return hidden_states

    class PixtralVisionWrapper(torch.nn.Module):

        def __init__(self, vision_tower, mm_projector):
            super().__init__()
            self.vision_tower = vision_tower
            self.mm_projector = mm_projector

        @torch.no_grad
        def forward(self, pixel_values, attention_mask):
            features = self.vision_tower(pixel_values, attention_mask)
            out = self.mm_projector(features, attention_mask)
            return out

    model = Mistral3ForConditionalGeneration.from_pretrained(args.model_path,
                                                             dtype="auto")
    vision_tower = model.vision_tower
    mm_projector = model.multi_modal_projector

    height = width = vision_config.image_size // vision_config.patch_size
    mesh = torch.meshgrid(torch.arange(height),
                          torch.arange(width),
                          indexing="ij")
    h_grid, v_grid = torch.stack(mesh, dim=-1).chunk(2, -1)
    ids = h_grid[..., 0] * width + v_grid[..., 0]
    vision_tower.register_buffer("position_ids", ids)

    PixtralAttention.forward = attn_forward
    PixtralVisionModel.forward = vision_tower_forward

    Mistral3PatchMerger.forward = patch_merger_forward
    Mistral3MultiModalProjector.forward = mm_projector_forward

    vision_tower = vision_tower.to(args.device, torch.bfloat16)
    mm_projector = mm_projector.to(args.device, torch.bfloat16)
    vision_tower.eval()
    mm_projector.eval()
    wrapper = PixtralVisionWrapper(vision_tower, mm_projector)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    part_name = 'vision'
    onnx_dir = f"{args.output_dir}/{part_name}/onnx"

    export_onnx(wrapper,
                input=(pixel_values, attention_mask),
                onnx_dir=onnx_dir,
                input_names=['input', 'attention_mask'],
                dynamic_axes={
                    'input': {
                        0: "batch"
                    },
                    'attention_mask': {
                        0: "batch"
                    }
                })
    build_trt_engine(
        args.model_type,
        input_sizes=[[list(pixel_values.shape[1:]) for _ in range(3)],
                     [list(attention_mask.shape[1:]) for _ in range(3)]],
        onnx_dir=onnx_dir,
        engine_dir=args.output_dir,
        max_batch_size=args.max_batch_size,
        engine_name=f"model.engine",
        dtype=torch.bfloat16)


def build_eclair_engine(args):

    class RadioWithNeck(torch.nn.Module):

        def __init__(self):
            super().__init__()

            try:
                self.model_encoder = torch.hub.load("NVlabs/RADIO",
                                                    "radio_model",
                                                    version="radio_v2.5-h")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load RADIO model from torch.hub: {e}")
            self.model_encoder.summary_idxs = torch.tensor(4)

            self.conv1 = torch.nn.Conv1d(1280, 1024, 1)
            self.layer_norm1 = torch.nn.LayerNorm(1024,
                                                  eps=1e-6,
                                                  elementwise_affine=True)
            self.conv2 = torch.nn.Conv2d(1024,
                                         1024,
                                         kernel_size=(1, 4),
                                         stride=(1, 4),
                                         padding=0,
                                         bias=False)
            self.layer_norm2 = torch.nn.LayerNorm(1024,
                                                  eps=1e-6,
                                                  elementwise_affine=True)

        @torch.no_grad
        def forward(self, pixel_values):
            _, feature = self.model_encoder(pixel_values)
            output = self.conv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
            output = self.layer_norm1(output).permute(0, 2, 1)

            b, d, _ = output.shape
            h = pixel_values.shape[-2] // 16
            w = pixel_values.shape[-1] // 16
            output = self.conv2(output.reshape(b, d, h, w))
            output = output.flatten(-2, -1).permute(0, 2, 1)
            output = self.layer_norm2(output)
            return output

    processor = NougatProcessor.from_pretrained(args.model_path)
    model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-base")
    model.encoder = RadioWithNeck()
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    model.config.decoder_start_token_id = processor.tokenizer.eos_token_id  # 2
    model.config.pad_token_id = processor.tokenizer.pad_token_id  # 1
    checkpoint_path = os.path.join(args.model_path, "model.safetensors")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}")
    load_model(model, checkpoint_path)

    wrapper = model.encoder.to(args.device)
    # temporary fix due to TRT onnx export bug
    for block in wrapper.model_encoder.model.blocks:
        block.attn.fused_attn = False

    image = torch.randn((1, 3, 2048, 1648),
                        device=args.device,
                        dtype=torch.bfloat16)
    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size,
        dtype=torch.bfloat16,
        engine_name='visual_encoder.engine')
