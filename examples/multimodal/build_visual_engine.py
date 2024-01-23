import argparse
import os
import shutil
from time import time

import tensorrt as trt
import torch
from PIL import Image
from transformers import (AutoProcessor, Blip2ForConditionalGeneration,
                          Blip2Processor, LlavaForConditionalGeneration,
                          NougatProcessor, VisionEncoderDecoderModel)


def export_visual_wrapper_onnx(visual_wrapper, image, output_dir):
    logger.log(trt.Logger.INFO, "Exporting onnx")
    os.mkdir(f'{output_dir}/onnx')
    torch.onnx.export(visual_wrapper,
                      image,
                      f'{output_dir}/onnx/visual_encoder.onnx',
                      opset_version=17,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {
                          0: 'batch'
                      }})


def build_trt_engine(img_height,
                     img_width,
                     output_dir,
                     minBS=1,
                     optBS=2,
                     maxBS=4):
    part_name = 'visual_encoder'
    onnx_file = '%s/onnx/%s.onnx' % (output_dir, part_name)
    engine_file = '%s/%s_fp16.engine' % (output_dir, part_name)
    logger.log(trt.Logger.INFO, "Building TRT engine for %s" % part_name)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    # Delete onnx files since we don't need them now
    shutil.rmtree(f'{output_dir}/onnx')

    nBS = -1
    nMinBS = minBS
    nOptBS = optBS
    nMaxBS = maxBS
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


def build_blip_engine(args):
    model_type = 'Salesforce/blip2-' + args.model_name
    processor = Blip2Processor.from_pretrained(model_type)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_type, torch_dtype=torch.float16)
    model.to(args.device)

    raw_image = Image.new('RGB', [10, 10])  # dummy image
    prompt = "Question: what is this? Answer:"
    inputs = processor(raw_image, prompt,
                       return_tensors="pt").to(args.device, torch.float16)
    image = inputs['pixel_values']

    class BlipVisionWrapper(torch.nn.Module):

        def __init__(self, model):
            super().__init__()
            self.vision_model = model.vision_model
            self.qformer = model.qformer
            self.projector = model.language_projection
            self.query_tokens = model.query_tokens

        def forward(self, image):
            features = self.vision_model(image)[0]
            qformer_output = self.qformer(query_embeds=self.query_tokens,
                                          encoder_hidden_states=features,
                                          return_dict=True)
            return self.projector(qformer_output.last_hidden_state)

    wrapper = BlipVisionWrapper(model)

    export_visual_wrapper_onnx(wrapper, image, args.output_dir)
    build_trt_engine(image.shape[2], image.shape[3], args.output_dir)


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
    model.to(args.device)
    wrapper = LlavaVisionWrapper(model.vision_tower,
                                 model.multi_modal_projector,
                                 model.config.vision_feature_layer)

    export_visual_wrapper_onnx(wrapper, image, args.output_dir)
    build_trt_engine(image.shape[2], image.shape[3], args.output_dir)


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
    build_trt_engine(image.shape[2], image.shape[3], args.output_dir)


if __name__ == '__main__':
    logger = trt.Logger(trt.Logger.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default=None,
                        help="Model name")
    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help="Huggingface repo or local directory with weights")
    parser.add_argument('--output_dir',
                        type=str,
                        default='visual_engines',
                        help="Directory where visual TRT engines are saved")
    args = parser.parse_args()

    args.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    args.output_dir = args.output_dir + "/" + args.model_name
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_name in ['opt-2.7b', 'flan-t5-xl']:
        build_blip_engine(args)
    elif 'llava' in args.model_name:
        build_llava_engine(args)
    elif 'nougat' in args.model_name:
        build_nougat_engine(args)
    else:
        raise RuntimeError(f"Invalid model name {args.model_name}")
