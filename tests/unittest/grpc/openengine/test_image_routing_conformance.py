# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-weight image cache reuse and streaming routing-hint conformance."""

import asyncio
import base64
from io import BytesIO

import grpc
import msgspec
import pytest
import torch
import zmq
from PIL import Image
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.grpc.openengine.bindings import generation_pb2 as pb
from tensorrt_llm.grpc.openengine.bindings import model_pb2, openengine_pb2_grpc
from tensorrt_llm.grpc.openengine.server import OpenEngineServer
from tensorrt_llm.llmapi import KvCacheConfig, KVEventsConfig

pytestmark = pytest.mark.threadleak(enabled=False)


def test_real_image_routing_keeps_cache_identity_content_sensitive(unused_tcp_port, tmp_path):
    assert torch.cuda.is_available(), "GPU conformance requires a CUDA device"
    endpoint = f"tcp://127.0.0.1:{unused_tcp_port}"
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.setsockopt(zmq.SUBSCRIBE, b"")
    subscriber.connect(endpoint)
    llm = None
    try:
        llm = LLM(
            model=str(llm_models_root(check=True) / "Qwen3/Qwen3-VL-2B-Instruct"),
            max_batch_size=1,
            max_seq_len=2048,
            max_num_tokens=2048,
            kv_cache_config=KvCacheConfig(
                max_tokens=4096,
                free_gpu_memory_fraction=0.3,
                enable_block_reuse=True,
                use_kv_cache_manager_v2=True,
                kv_events_config=KVEventsConfig(
                    enable_kv_cache_events=True, publisher="zmq", endpoint=endpoint
                ),
            ),
        )

        async def run():
            server = OpenEngineServer(host="127.0.0.1", port=0, llm=llm, model="image-routing")
            try:
                await server.start()
                async with grpc.aio.insecure_channel(f"127.0.0.1:{server.port}") as channel:
                    control = openengine_pb2_grpc.ControlStub(channel)
                    info = await control.GetModelInfo(
                        model_pb2.GetModelInfoRequest(model="image-routing")
                    )
                    assert info.extra["trtllm_image_routing_version"] == 1
                    inference = openengine_pb2_grpc.InferenceStub(channel)

                    async def generate(color, request_id):
                        image = BytesIO()
                        Image.new("RGB", (224, 224), color).save(image, format="PNG")
                        request = pb.GenerateRequest(
                            model="image-routing",
                            request_id=request_id,
                            prompt=(
                                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                                "Name the solid background color. Answer with one color word."
                                "<|im_end|>\n<|im_start|>assistant\n"
                            ),
                            sampling=pb.SamplingParams(temperature=0),
                            stopping=pb.StoppingOptions(max_tokens=16),
                            media=[
                                pb.MediaItem(
                                    modality=pb.MODALITY_IMAGE,
                                    uuid="dynamo-image-v1:0123456789abcdef",
                                    data_uri="data:image/png;base64,"
                                    + base64.b64encode(image.getvalue()).decode(),
                                )
                            ],
                        )
                        ids, usages, finished = [], [], 0
                        async for response in inference.Generate(request, timeout=180):
                            assert not response.HasField("error"), response
                            if response.HasField("token"):
                                ids.extend(token.token_id for token in response.token.tokens)
                            if response.HasField("usage"):
                                usages.append(response.usage)
                            finished += int(response.HasField("finished"))
                        assert finished == 1 and usages
                        answer = llm.tokenizer.decode(ids).lower()
                        assert color in answer, answer
                        return usages[-1].cached_prompt_tokens

                    first = await generate("red", "red-first")
                    repeated = await generate("red", "red-repeat")
                    changed = await generate("blue", "blue-same-hint")
                    assert repeated > first
                    assert changed < repeated, "changed content reused the first image's KV prefix"
            finally:
                await server.stop()

        asyncio.run(run())
        captured, hashes = [], set()
        while subscriber.poll(1000):
            frames = subscriber.recv_multipart()
            captured.append(frames[-1])
            batch = msgspec.msgpack.decode(frames[-1])
            for event in batch[1]:
                metadata = event.get("trtllm_image_routing")
                if not metadata:
                    continue
                for keys in event["mm_keys"]:
                    for key in keys:
                        assert key["uuid"] == "dynamo-image-v1:0123456789abcdef"
                        hashes.add(key["hash"])
        assert len(hashes) == 2, (
            "both content digests must retain the routing hint on the event wire"
        )
        (tmp_path / "image-routing-events.msgpack").write_bytes(msgspec.msgpack.encode(captured))
    finally:
        if llm is not None:
            llm.shutdown()
        subscriber.close(linger=0)
        context.term()
