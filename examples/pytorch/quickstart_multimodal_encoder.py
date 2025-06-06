import argparse
import json
import os
from typing import Any, Dict, List

from quickstart_advanced import add_llm_args

from tensorrt_llm.inputs import ALL_SUPPORTED_MULTIMODAL_MODELS
import asyncio

example_images = [
    "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/inpaint.png",
    "https://huggingface.co/datasets/Sayali9141/traffic_signal_images/resolve/main/61.jpg",
]
example_image_prompts = [
    "Describe the natural environment in the image.",
    "Describe the object and the weather condition in the image.",
    "Describe the traffic condition on the road in the image.",
]
example_videos = [
    "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/OAI-sora-tokyo-walk.mp4",
    "https://huggingface.co/datasets/Efficient-Large-Model/VILA-inference-demos/resolve/main/world.mp4",
]
example_video_prompts = [
    "Tell me what you see in the video briefly.",
    "Describe the scene in the video briefly.",
]


def add_multimodal_args(parser):
    parser.add_argument("--model_type",
                        type=str,
                        choices=ALL_SUPPORTED_MULTIMODAL_MODELS,
                        help="Model type.")
    parser.add_argument("--modality",
                        type=str,
                        choices=["image", "video"],
                        default="image",
                        help="Media type.")
    parser.add_argument("--media",
                        type=str,
                        nargs="+",
                        help="A single or a list of media filepaths / urls.")
    parser.add_argument("--num_frames",
                        type=int,
                        default=8,
                        help="The number of video frames to be sampled.")
    return parser


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Multimodal models with the PyTorch workflow.")
    parser = add_llm_args(parser)
    parser = add_multimodal_args(parser)
    args = parser.parse_args()
    return args

def setup_encoder(args):
    from tensorrt_llm._torch.multimodal.mm_encoder import MultimodalEncoder
    return MultimodalEncoder(model=args.model_dir, max_batch_size=args.max_batch_size)

def main():
    args = parse_arguments()

    encoder = setup_encoder(args)
    from tensorrt_llm.executor.multimodal.request import MultimodalRequest, MultimodalItem
    from tensorrt_llm._torch.multimodal.mm_utils import SharedTensorContainer
    items = [
        MultimodalItem(
            req_id=1,
            id=0,
            modality_type="image",
            url=example_images[0]
        ),
        MultimodalItem(
            req_id=1,
            id=1,
            modality_type="image",
            url=example_images[1]
        ),
        MultimodalItem(
            req_id=1,
            id=2,
            modality_type="image",
            url=example_images[2]
        )
    ]
    mm_request = MultimodalRequest(
        items=items
    )

    mm_requests = [mm_request] * 20
    outputs = encoder.generate_from_mm_request(mm_requests)
    for output in outputs:
        #print(f"output: {output.multimodal_params.embeddings.device}")
        for i in range(output.multimodal_params.num_items):
            sta = output.multimodal_params.item_offsets[i]
            end = sta + output.multimodal_params.item_token_length[i]
            mm_embedding_dict = output.multimodal_params.embeddings[0]
            mm_embedding = SharedTensorContainer.from_dict(mm_embedding_dict).get_local_view()

            print(f"item {i} embedding: {mm_embedding[sta:end].reshape(1, -1)[:5]}")
        #del output.multimodal_params.embeddings
        #gc.collect()
        print(f"deleting output embeddings")
        #torch.cuda.ipc_collect()
        #torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
