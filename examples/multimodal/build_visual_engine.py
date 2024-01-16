import argparse
import os
from time import time

import tensorrt as trt
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


def export_visual_wrapper_onnx(visual_wrapper, image, output_dir):
    logger.log(trt.Logger.INFO, "Exporting onnx")
    torch.onnx.export(visual_wrapper,
                      image,
                      f'{output_dir}/visual_encoder.onnx',
                      opset_version=17,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {
                          0: 'batch'
                      }})


def build_trt_engine(part_id,
                     img_height,
                     img_width,
                     output_dir,
                     minBS=1,
                     optBS=2,
                     maxBS=4):
    part_name = 'visual_encoder' if part_id == 0 else 'Qformer'
    onnx_file = '%s/%s.onnx' % (output_dir, part_name)
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

    nBS = -1
    nMinBS = minBS
    nOptBS = optBS
    nMaxBS = maxBS
    logger.log(trt.Logger.INFO,
               f"Processed image dims {img_height}x{img_width}")

    if part_id == 0:  # Feature extractor
        H, W = img_height, img_width
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, H, W]
        profile.set_shape(inputT.name, [nMinBS, 3, H, W], [nOptBS, 3, H, W],
                          [nMaxBS, 3, H, W])
    elif part_id == 1:  # BLIP Qformer
        inputT = network.get_input(0)
        dims = [32, 768]
        inputT.shape = [nBS] + dims
        profile.set_shape(inputT.name, [nMinBS] + dims, [nOptBS] + dims,
                          [nMaxBS] + dims)

        inputT = network.get_input(1)
        dims = [257, 1408]
        inputT.shape = [nBS] + dims
        profile.set_shape(inputT.name, [nMinBS] + dims, [nOptBS] + dims,
                          [nMaxBS] + dims)

        inputT = network.get_input(2)
        inputT.shape = [nBS, 257]
        profile.set_shape(inputT.name, [nMinBS, inputT.shape[1]],
                          [nOptBS, inputT.shape[1]], [nMaxBS, inputT.shape[1]])
    else:
        raise RuntimeError("Invalid part id")

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
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(model_type)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_type, torch_dtype=torch.float16)
    model.to(device)

    raw_image = Image.new('RGB', [10, 10])  # dummy image
    # image = vis_processors["eval"](image).unsqueeze(0).to(device)
    prompt = "Question: what is this? Answer:"
    inputs = processor(raw_image, prompt,
                       return_tensors="pt").to(device, torch.float16)
    image = inputs['pixel_values']

    visual_wrapper = model.vision_model
    image_embeds = visual_wrapper(image)[0]
    export_visual_wrapper_onnx(visual_wrapper, image, args.output_dir)
    build_trt_engine(0, image.shape[2], image.shape[3], args.output_dir)

    class QformerWrapper(torch.nn.Module):

        def __init__(self, Qformer, projector):
            super().__init__()
            self.model = Qformer
            self.projector = projector

        def forward(self, query_tokens, image_embeds, image_atts):
            query_output = self.model(query_embeds=query_tokens,
                                      encoder_hidden_states=image_embeds,
                                      encoder_attention_mask=image_atts,
                                      return_dict=True)
            return self.projector(query_output.last_hidden_state)

    projector = model.language_projection
    q_wrapper = QformerWrapper(model.qformer, projector)

    image_atts = torch.ones(image_embeds.size()[:-1],
                            dtype=torch.long).to(image.device)
    query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)

    torch.onnx.export(
        q_wrapper, (query_tokens, image_embeds, image_atts),
        f'{args.output_dir}/Qformer.onnx',
        opset_version=17,
        input_names=['query_tokens', 'image_embeds', 'image_atts'],
        output_names=['query_output'],
        dynamic_axes={
            'query_tokens': {
                0: 'batch'
            },
            'image_embeds': {
                0: 'batch'
            },
            'image_atts': {
                0: 'batch'
            }
        })

    build_trt_engine(1, image.shape[2], image.shape[3], args.output_dir)


def build_llava_engine(args):
    # Import these here to avoid installing llava when running blip models only
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    model_name = get_model_name_from_path(args.model_path)
    _, model, image_processor, _ = load_pretrained_model(
        args.model_path, None, model_name)

    image = Image.new('RGB', [10, 10])  # dummy image
    image = image_processor(image, return_tensors='pt')['pixel_values']
    image = image.half().to(device)

    visual_wrapper = torch.nn.Sequential(model.get_vision_tower(),
                                         model.get_model().mm_projector)
    export_visual_wrapper_onnx(visual_wrapper, image, args.output_dir)
    build_trt_engine(0, image.shape[2], image.shape[3], args.output_dir)


if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
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

    args.output_dir = args.output_dir + "/" + args.model_name
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.model_name in ['opt-2.7b', 'flan-t5-xl']:
        build_blip_engine(args)
    elif 'llava' in args.model_name:
        build_llava_engine(args)
    else:
        raise RuntimeError(f"Invalid model name {args.model_name}")
