import argparse
import os
import shutil
import sys
from time import time

# isort: off
import torch
import tensorrt as trt
from tensorrt_llm.builder import Builder
# isort: on

from PIL import Image
from transformers import (AutoProcessor, Blip2ForConditionalGeneration,
                          Blip2Processor, LlavaForConditionalGeneration,
                          NougatProcessor, VisionEncoderDecoderModel)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        type=str,
                        default=None,
                        choices=[
                            'opt-2.7b', 'opt-6.7b', 'flan-t5-xl', 'flan-t5-xxl',
                            'llava', 'vila', 'nougat'
                        ],
                        help="Model type")
    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help="Huggingface repo or local directory with weights")
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
    return parser.parse_args()


class VisionEngineBuilder:

    def __init__(self, args):
        args.device = torch.device(
            "cuda") if torch.cuda.is_available() else "cpu"
        if args.output_dir is None:
            args.output_dir = 'visual_engines/%s' % (
                args.model_path.split('/')[-1])
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        self.args = args

    def build(self):
        args = self.args
        if 'opt' in args.model_type or 't5' in args.model_type:
            build_blip2_engine(args)
        elif args.model_type == 'llava':
            build_llava_engine(args)
        elif args.model_type == 'vila':
            assert args.vila_path is not None, "Please clone and provide VILA source code path"
            build_vila_engine(args)
        elif args.model_type == 'nougat':
            build_nougat_engine(args)
        else:
            raise RuntimeError(f"Invalid model type {args.model_type}")


def export_visual_wrapper_onnx(visual_wrapper, image, output_dir):
    logger.log(trt.Logger.INFO, "Exporting onnx")
    os.makedirs(f'{output_dir}/onnx', exist_ok=True)
    torch.onnx.export(visual_wrapper,
                      image,
                      f'{output_dir}/onnx/visual_encoder.onnx',
                      opset_version=17,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {
                          0: 'batch'
                      }})


def build_trt_engine(model_type, img_height, img_width, output_dir,
                     max_batch_size):
    part_name = 'visual_encoder'
    onnx_file = '%s/onnx/%s.onnx' % (output_dir, part_name)
    engine_file = '%s/%s.engine' % (output_dir, part_name)
    config_file = '%s/%s' % (output_dir, "config.json")
    logger.log(trt.Logger.INFO, "Building TRT engine for %s" % part_name)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config_wrapper = Builder().create_builder_config(precision="float16",
                                                     model_type=model_type)
    config = config_wrapper.trt_builder_config

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), os.path.abspath(onnx_file)):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    # Delete onnx files since we don't need them now
    shutil.rmtree(f'{output_dir}/onnx')

    nBS = -1
    nMinBS = 1
    nOptBS = max(nMinBS, int(max_batch_size / 2))
    nMaxBS = max_batch_size

    logger.log(trt.Logger.INFO,
               f"Processed image dims {img_height}x{img_width}")
    H, W = img_height, img_width
    inputT = network.get_input(0)
    inputT.shape = [nBS, 3, H, W]
    profile.set_shape(inputT.name, [nMinBS, 3, H, W], [nOptBS, 3, H, W],
                      [nMaxBS, 3, H, W])
    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO,
                   "Succeeded building %s in %d s" % (engine_file, t1 - t0))
        with open(engine_file, 'wb') as f:
            f.write(engine_string)

    Builder.save_config(config_wrapper, config_file)


def build_blip2_engine(args):
    model_type = 'Salesforce/blip2-' + args.model_type
    processor = Blip2Processor.from_pretrained(model_type)

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
        model_type, torch_dtype=torch.float16)
    wrapper = Blip2VisionWrapper(model.vision_model, model.qformer,
                                 model.language_projection, model.query_tokens)
    wrapper.to(args.device)

    export_visual_wrapper_onnx(wrapper, image, args.output_dir)
    build_trt_engine(model_type, image.shape[2], image.shape[3],
                     args.output_dir, args.max_batch_size)


def build_llava_engine(args):
    processor = AutoProcessor.from_pretrained(args.model_path)
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
    wrapper = LlavaVisionWrapper(model.vision_tower.to(args.device),
                                 model.multi_modal_projector.to(args.device),
                                 model.config.vision_feature_layer)

    export_visual_wrapper_onnx(wrapper, image, args.output_dir)
    build_trt_engine(args.model_type, image.shape[2], image.shape[3],
                     args.output_dir, args.max_batch_size)


def build_vila_engine(args):
    # Note: VILA model is not in public HF model zoo yet. We need to explicitly import from the git repo
    sys.path.append(args.vila_path)
    from llava.model import LlavaLlamaForCausalLM

    model = LlavaLlamaForCausalLM.from_pretrained(args.model_path,
                                                  torch_dtype=torch.float16)
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    raw_image = Image.new('RGB', [10, 10])  # dummy image
    image = image_processor(images=raw_image,
                            return_tensors="pt")['pixel_values'].to(
                                args.device, torch.float16)

    class VilaVisionWrapper(torch.nn.Module):

        def __init__(self, tower, projector):
            super().__init__()
            self.tower = tower
            self.projector = projector

        def forward(self, image):
            features = self.tower(image)
            return self.projector(features)

    model = LlavaLlamaForCausalLM.from_pretrained(args.model_path,
                                                  torch_dtype=torch.float16)
    wrapper = VilaVisionWrapper(
        model.get_model().get_vision_tower().to(args.device),
        model.get_model().mm_projector.to(args.device))

    export_visual_wrapper_onnx(wrapper, image, args.output_dir)
    build_trt_engine(args.model_type, image.shape[2], image.shape[3],
                     args.output_dir, args.max_batch_size)


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

    export_visual_wrapper_onnx(wrapper, image, args.output_dir)
    build_trt_engine(args.model_type, image.shape[2], image.shape[3],
                     args.output_dir, args.max_batch_size)


if __name__ == '__main__':
    logger = trt.Logger(trt.Logger.INFO)
    args = parse_arguments()
    builder = VisionEngineBuilder(args)
    builder.build()
