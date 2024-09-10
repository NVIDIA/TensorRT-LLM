import os
import shutil
import sys
import tarfile
from time import time

import yaml

# isort: off
import torch
import tensorrt as trt
from tensorrt_llm.builder import Builder
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForVision2Seq, AutoProcessor,
                          Blip2ForConditionalGeneration, Blip2Processor,
                          FuyuForCausalLM, FuyuProcessor,
                          LlavaForConditionalGeneration, NougatProcessor,
                          Pix2StructForConditionalGeneration,
                          VisionEncoderDecoderModel)
# isort: on
import json
import math

import torch.nn.functional as F
from PIL import Image
from safetensors.torch import save_file


def add_multimodal_arguments(parser):
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        choices=[
                            'blip2', 'llava', 'llava_next', 'vila', 'nougat',
                            'cogvlm', 'fuyu', 'pix2struct', 'neva', 'kosmos-2',
                            'video-neva', 'phi-3-vision'
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
    return parser


class VisionEngineBuilder:

    def __init__(self, args):
        args.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        if args.output_dir is None:
            # default path to save the engines
            model_name = args.model_path.split('/')[-1]
            args.output_dir = f'tmp/trt_engines/{model_name}/vision_encoder'

        os.makedirs(args.output_dir, exist_ok=True)

        self.args = args

    def build(self):
        args = self.args
        if args.model_type == 'blip2':
            build_blip2_engine(args)
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
        else:
            raise RuntimeError(f"Invalid model type {args.model_type}")


def export_onnx(model,
                input,
                onnx_dir,
                onnx_name='model.onnx',
                input_names=['input'],
                output_names=['output'],
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
                     num_frames=None,
                     onnx_name='model.onnx',
                     engine_name='model.engine',
                     delete_onnx=True,
                     logger=trt.Logger(trt.Logger.INFO)):
    onnx_file = f'{onnx_dir}/{onnx_name}'
    engine_file = f'{engine_dir}/{engine_name}'
    config_file = f'{engine_dir}/config.json'
    logger.log(trt.Logger.INFO, f"Building TRT engine to {engine_file}")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()

    config_args = {
        "precision": str(dtype).split('.')[-1],
        "model_type": model_type
    }
    if num_frames is not None:
        config_args["num_frames"] = num_frames

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

    inputT = network.get_input(0)

    # input sizes can be:
    # - integer list, when inputs are constant size images. e.g. [3, H, W]
    # - list of integer lists, when inputs are dynamic size images. e.g. [[1, 1, 2700], [1, 500, 2700], [1, 4096, 2700]]
    assert isinstance(input_sizes, list), "input_sizes must be a list"
    if isinstance(input_sizes[0], int):
        logger.log(trt.Logger.INFO, f"Processed input sizes {input_sizes}")
        inputT.shape = [nBS, *input_sizes]
        min_size = opt_size = max_size = input_sizes
    elif len(input_sizes) == 3 and isinstance(input_sizes[0], list):
        min_size, opt_size, max_size = input_sizes
        logger.log(
            trt.Logger.INFO,
            f"Processed min/opt/max input sizes {min_size}/{opt_size}/{max_size}"
        )
    else:
        raise ValueError(f"invalid input sizes: {input_sizes}")

    profile.set_shape(inputT.name, [nMinBS, *min_size], [nOptBS, *opt_size],
                      [nMaxBS, *max_size])
    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO,
                   "Succeeded building %s in %d s" % (engine_file, t1 - t0))
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

    model = Blip2ForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16)

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

    model = Pix2StructForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=dtype)

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

        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16)
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

        model = LlavaNextForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float16)
        wrapper = LlavaNextVisionWrapper(
            model.vision_tower.vision_model.to(args.device),
            model.multi_modal_projector.to(args.device),
        )

    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type,
        [image.shape[1], image.shape[2], image.shape[3]],  # [3, H, W]
        f'{args.output_dir}/onnx',
        args.output_dir,
        args.max_batch_size)
    if args.model_type == "llava_next":
        image_newline = model.image_newline.data
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
                                                      torch_dtype=torch.float16)
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
                                                  torch_dtype=dtype,
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
                                            torch_dtype=torch.float16)

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
                                        torch_dtype=torch.bfloat16,
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
                                        torch_dtype=torch.bfloat16,
                                        trust_remote_code=True)
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
        num_frames=num_frames)


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
                                                   torch_dtype=torch.float16)
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
    processor = AutoProcessor.from_pretrained(args.model_path,
                                              trust_remote_code=True)
    raw_image = Image.new('RGB', [10, 10])  # dummy image
    image = processor(text="<|image_1|>\ndummy",
                      images=raw_image,
                      return_tensors="pt")['pixel_values'].to(
                          args.device, torch.float16)
    try:
        with open(f"{args.model_path}/preprocessor_config.json", "r") as file:
            config = file.read()
            config_dict = json.loads(config)
            num_crops = config_dict.get("num_crops")
    except:
        num_crops = 16

    class Phi3VisionWrapper(torch.nn.Module):

        def __init__(self, img_processor, img_projection, layer_idx,
                     image_dim_out):
            super().__init__()
            self.img_processor = img_processor
            self.img_projection = img_projection
            self.layer_idx = layer_idx
            self.image_dim_out = image_dim_out

        def get_img_features(
                self, img_embeds: torch.FloatTensor) -> torch.FloatTensor:
            LAYER_IDX = self.layer_idx

            img_processor_output = self.img_processor(img_embeds,
                                                      output_hidden_states=True)
            img_feature = img_processor_output.hidden_states[LAYER_IDX]

            patch_feature = img_feature[:, 1:]
            return patch_feature

        def forward(self, image):
            img_features = self.get_img_features(image)
            base_feat_height = int(math.sqrt(img_features.shape[1]))
            C = self.image_dim_out
            H = base_feat_height
            img_features = img_features.reshape(-1, H, H, C).reshape(
                -1, H // 2, 2, H // 2, 2,
                C).contiguous().permute(0, 1, 3, 2, 4,
                                        5).reshape(-1, H // 2, H // 2,
                                                   4 * C).contiguous()
            return self.apply_img_projection(img_features)

        def apply_img_projection(self, input):
            return self.img_projection(input)

    model = AutoModelForCausalLM.from_pretrained(args.model_path,
                                                 torch_dtype=torch.float16,
                                                 trust_remote_code=True).to(
                                                     args.device)

    wrapper = Phi3VisionWrapper(model.model.vision_embed_tokens.img_processor,
                                model.model.vision_embed_tokens.img_projection,
                                model.model.vision_embed_tokens.layer_idx,
                                model.model.vision_embed_tokens.image_dim_out)
    image = image.flatten(0, 1)
    glb_GN = wrapper.apply_img_projection(
        model.model.vision_embed_tokens.glb_GN)
    sub_GN = wrapper.apply_img_projection(
        model.model.vision_embed_tokens.sub_GN)
    tensors = {"glb_GN": glb_GN, "sub_GN": sub_GN}
    save_file(tensors, args.output_dir + "/image_newlines.safetensors")
    export_onnx(wrapper, image, f'{args.output_dir}/onnx')
    build_trt_engine(
        args.model_type, [image.shape[1], image.shape[2], image.shape[3]],
        f'{args.output_dir}/onnx', args.output_dir,
        args.max_batch_size * (num_crops + 1))  #TODO: Take input from config
