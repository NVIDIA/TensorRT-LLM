#!/usr/bin/python

import argparse
import base64
import io
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from functools import partial

import numpy as np
import requests
import tritonclient.grpc as grpcclient
from PIL import Image
from transformers import AutoProcessor, Blip2Processor
from utils import utils


def pixtral_pad_images(
        image_list: List[Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
    if not image_list:
        return np.empty((0, 0, 0, 0), dtype=np.uint8), np.empty((0, 2),
                                                                dtype=np.int64)
    image_list_np = [np.array(img) for img in image_list]
    shapes = [img.shape for img in image_list_np]
    assert all(len(s) == 3
               for s in shapes), "All input images must have three dimensions"
    assert all(s[-1] == shapes[0][-1] for s in
               shapes), "All input images must have the same number of channels"
    max_h, max_w = max(s[0] for s in shapes), max(s[1] for s in shapes)
    for i in range(len(image_list_np)):
        image_list_np[i] = np.pad(image_list_np[i],
                                  ((0, max_h - image_list_np[i].shape[0]),
                                   (0, max_w - image_list_np[i].shape[1]),
                                   (0, 0)),
                                  mode='constant')
    raw_image = np.stack(image_list_np, axis=0)
    image_sizes = np.array([s[:2] for s in shapes], dtype=np.int64)
    return raw_image, image_sizes


def prepare_inputs(text_data,
                   image_data,
                   image_sizes,
                   request_output_len_data,
                   beam_width_data,
                   temperature_data,
                   repetition_penalty_data,
                   presence_penalty_data,
                   end_id,
                   pad_id,
                   top_k_data,
                   top_p_data,
                   streaming_data,
                   prompt_table_extra_id_data,
                   image_input_name="image_input"):
    inputs = [
        utils.prepare_tensor("text_input", text_data, grpcclient),
        utils.prepare_tensor("max_tokens", request_output_len_data, grpcclient),
        utils.prepare_tensor("beam_width", beam_width_data, grpcclient),
        utils.prepare_tensor("temperature", temperature_data, grpcclient),
        utils.prepare_tensor("end_id", end_id, grpcclient),
        utils.prepare_tensor("pad_id", pad_id, grpcclient),
        utils.prepare_tensor("top_k", top_k_data, grpcclient),
        utils.prepare_tensor("top_p", top_p_data, grpcclient),
        utils.prepare_tensor("stream", streaming_data, grpcclient),
    ]
    if image_data is not None:
        inputs += [
            utils.prepare_tensor(image_input_name, image_data, grpcclient),
        ]
    if image_sizes is not None:
        inputs += [
            utils.prepare_tensor("image_sizes_input", image_sizes, grpcclient),
        ]
    if repetition_penalty_data is not None:
        inputs += [
            utils.prepare_tensor("repetition_penalty", repetition_penalty_data,
                                 grpcclient),
        ]
    if presence_penalty_data is not None:
        inputs += [
            utils.prepare_tensor("presence_penalty", presence_penalty_data,
                                 grpcclient),
        ]
    if prompt_table_extra_id_data is not None:
        inputs += [
            utils.prepare_tensor("prompt_table_extra_id",
                                 prompt_table_extra_id_data, grpcclient),
        ]
    return inputs


def load_image(image_path) -> Image.Image:
    if image_path.startswith("http") or image_path.startswith("https"):
        image_bytes = requests.get(image_path, stream=True).content
    elif image_path.startswith("data:image/jpeg;base64,"):
        image_base64 = image_path.split(",")[1]
        image_bytes = base64.b64decode(image_base64)
    else:
        image_bytes = Path(image_path).read_bytes()

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_video(video_path, num_of_frames):
    import av
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames,
                        total_frames / num_of_frames).astype(int)

    def read_video_pyav(container, indices):

        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])

    video = read_video_pyav(container, indices)
    return video


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL. Default is localhost:8001.",
    )
    parser.add_argument('--text',
                        type=str,
                        required=False,
                        default='Question: which city is this? Answer:',
                        help='Input text')

    parser.add_argument(
        '--image',
        type=str,
        required=False,
        default=
        'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png',
        help='Input image')

    parser.add_argument('--video',
                        type=str,
                        required=False,
                        default=None,
                        help='Input video')

    parser.add_argument(
        '--video_num_frames',
        type=int,
        required=False,
        default=None,
        help=
        'The number of frames sampled from the video in the Llava-OneVision model.'
    )

    parser.add_argument('--end-id',
                        type=int,
                        required=False,
                        default=-1,
                        help='The token id for end token.')

    parser.add_argument('--pad-id',
                        type=int,
                        required=False,
                        default=1,
                        help='The token id for pad token.')

    parser.add_argument(
        "-b",
        "--beam-width",
        required=False,
        type=int,
        default=1,
        help="Beam width value",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=1.0,
        help="temperature value",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        required=False,
        default=1.0,
        help="The repetition penalty value",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        required=False,
        default=None,
        help="The presence penalty value",
    )

    parser.add_argument(
        "--request-output-len",
        type=int,
        required=False,
        default=16,
        help="Request output length",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        required=False,
        default=1,
        help="top k value",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        required=False,
        default=0.,
        help="top p value",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode. Default is False.",
    )
    parser.add_argument(
        "--use_bls",
        action="store_true",
        required=False,
        default=False,
        help="Use BLS model instead of ensemble.",
    )
    parser.add_argument(
        "--prompt_table_extra_id",
        type=int,
        required=False,
        default=None,
        help=
        "When enable kv cache reuse, we need a unique id to determine whether the images are the same. The type of extra id is uint64, and its range is from 1 to the maximum value of uint64.",
    )
    parser.add_argument("--model_type",
                        required=True,
                        choices=[
                            'blip2', 'llava', 'vila', 'mllama',
                            'llava_onevision', 'qwen2_vl', 'pixtral'
                        ],
                        help="Model type")
    parser.add_argument("--hf_model_dir",
                        required=False,
                        type=str,
                        default=None,
                        help="path to the model directory")
    FLAGS = parser.parse_args()
    # load and process images or video
    image_sizes = np.empty((0, 2), dtype=np.int64)
    if 'vila' in FLAGS.model_type:
        image_paths = FLAGS.image.split(",")
        raw_image = []
        for image_path in image_paths:
            raw_image.append(load_image(image_path))
    elif 'pixtral' in FLAGS.model_type:
        image_paths = FLAGS.image.split(",") if FLAGS.image else []
        raw_image = []
        for image_path in image_paths:
            raw_image.append(load_image(image_path))
        raw_image, image_sizes = pixtral_pad_images(raw_image)
    elif FLAGS.video is not None:
        assert FLAGS.video_num_frames is not None, "Number of frames should be provided for video input."
        raw_video = load_video(FLAGS.video, FLAGS.video_num_frames)
    else:
        raw_image = load_image(FLAGS.image)

    if 'blip2' in FLAGS.model_type:
        if FLAGS.hf_model_dir is not None and os.path.exists(
                FLAGS.hf_model_dir):
            processor = Blip2Processor.from_pretrained(FLAGS.hf_model_dir)
        else:
            processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b")
        image = processor(raw_image, FLAGS.text,
                          return_tensors="pt")['pixel_values']
    elif FLAGS.model_type == 'llava':
        if FLAGS.hf_model_dir is not None and os.path.exists(
                FLAGS.hf_model_dir):
            processor = AutoProcessor.from_pretrained(FLAGS.hf_model_dir)
        else:
            processor = AutoProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf")

        image = processor(text=FLAGS.text,
                          images=raw_image,
                          return_tensors="pt")['pixel_values']
    elif 'vila' in FLAGS.model_type:
        # vila support multiple images input
        sys.path.append(FLAGS.hf_model_dir + "/../VILA")
        from llava.model import LlavaLlamaConfig  # noqa
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            FLAGS.hf_model_dir,
            device_map='auto',
            trust_remote_code=True,
        )
        vision_tower = model.get_vision_tower()
        image_processor = vision_tower.image_processor
        from llava.mm_utils import process_images
        if not isinstance(raw_image, list):
            raw_image = [raw_image]
        image = process_images(raw_image, image_processor, model.config)

    if 'mllama' in FLAGS.model_type:
        image_tag = '<|image|>'
        if image_tag not in FLAGS.text:
            FLAGS.text = image_tag + FLAGS.text
        image_data = np.array([[raw_image]])
        image_input_name = "image_bytes_input"
    elif 'pixtral' in FLAGS.model_type:
        image_data = np.array([raw_image])
        image_input_name = "image_bytes_input"
    elif 'llava_onevision' in FLAGS.model_type:
        if FLAGS.video is not None:
            image_data = np.array([raw_video])
            image_input_name = "video_bytes_input"
        else:
            image_data = np.array([[raw_image]])
            image_input_name = "image_bytes_input"
    elif FLAGS.model_type == 'qwen2_vl':
        raw_image = raw_image.resize((504, 504))
        image_data = np.array([[raw_image]])
        image_input_name = "image_bytes_input"
    else:
        image = image.unsqueeze(0)
        image_data = image.numpy().astype(np.float16)
        image_input_name = "image_input"

    text_data = np.array([[FLAGS.text.encode("utf8")]], dtype=np.object_)
    end_id_data = np.array([[FLAGS.end_id]], dtype=np.int32)
    pad_id_data = np.array([[FLAGS.pad_id]], dtype=np.int32)
    request_output_len = [[FLAGS.request_output_len]]
    request_output_len_data = np.array(request_output_len, dtype=np.int32)
    beam_width = [[FLAGS.beam_width]]
    beam_width_data = np.array(beam_width, dtype=np.int32)
    top_k = [[FLAGS.top_k]]
    top_k_data = np.array(top_k, dtype=np.int32)
    top_p = [[FLAGS.top_p]]
    top_p_data = np.array(top_p, dtype=np.float32)
    temperature = [[FLAGS.temperature]]
    temperature_data = np.array(temperature, dtype=np.float32)
    streaming = [[FLAGS.streaming]]
    streaming_data = np.array(streaming, dtype=bool)
    image_data = None if image_data.size == 0 else image_data
    image_sizes_data = None if image_sizes.size == 0 else np.array(
        [image_sizes], dtype=np.int64)

    model_name = "ensemble"
    if FLAGS.use_bls:
        model_name = "tensorrt_llm_bls"

    repetition_penalty_data = None
    if FLAGS.repetition_penalty is not None:
        repetition_penalty = [[FLAGS.repetition_penalty]]
        repetition_penalty_data = np.array(repetition_penalty, dtype=np.float32)
    presence_penalty_data = None
    if FLAGS.presence_penalty is not None:
        presence_penalty = [[FLAGS.presence_penalty]]
        presence_penalty_data = np.array(presence_penalty, dtype=np.float32)

    prompt_table_extra_id_data = None
    if FLAGS.prompt_table_extra_id is not None:
        prompt_table_extra_id = [[FLAGS.prompt_table_extra_id]]
        prompt_table_extra_id_data = np.array(prompt_table_extra_id,
                                              dtype=np.uint64)

    inputs = prepare_inputs(text_data,
                            image_data,
                            image_sizes_data,
                            request_output_len_data,
                            beam_width_data,
                            temperature_data,
                            repetition_penalty_data,
                            presence_penalty_data,
                            end_id_data,
                            pad_id_data,
                            top_k_data,
                            top_p_data,
                            streaming_data,
                            prompt_table_extra_id_data,
                            image_input_name=image_input_name)

    start_time = datetime.now()

    #Only include needed outputs
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("text_output"))

    with utils.create_inference_server_client('grpc',
                                              FLAGS.url,
                                              concurrency=None,
                                              verbose=FLAGS.verbose) as client:
        user_data = utils.UserData()

        if FLAGS.streaming:
            client.start_stream(
                callback=partial(utils.completion_callback, user_data),
                stream_timeout=None,
            )
            client.async_stream_infer(model_name, inputs, outputs=outputs)
            client.stop_stream(cancel_requests=False)

            results = []
            while True:
                try:
                    (result,
                     error) = user_data._completed_requests.get(block=False)
                    output = result.as_numpy("text_output")
                    for i in range(FLAGS.beam_width):
                        print("[beam", i, "]: ", output[i].decode())
                except Exception:
                    break

        else:
            client.async_infer(model_name,
                               inputs,
                               partial(utils.completion_callback, user_data),
                               outputs=outputs)
            results = utils.get_grpc_results(user_data, request_parallelism=1)

    stop_time = datetime.now()

    if not FLAGS.streaming:
        output = results[0].as_numpy("text_output")
        for i in range(FLAGS.beam_width):
            print("[beam", i, "]:")
            print(output[i].decode())

    latency = (stop_time - start_time).total_seconds() * 1000.0
    latency = round(latency, 3)
    print(f"[INFO] Latency: {latency} ms")
