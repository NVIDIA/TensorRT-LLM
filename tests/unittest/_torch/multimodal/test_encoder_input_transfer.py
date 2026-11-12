# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Raw multimodal encoder inputs go to the GPU once, in the encoder call.

`encode_multimodal_by_groups` makes the owner's device keys on the GPU for each call,
and `request_device_paths` keeps the executor's per-request transfer away from raw media,
so request payloads stay on the host and only the encoded items are copied.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch import host_staging
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalModelMixin,
    _get_mm_aux_stream,
    encode_multimodal_by_groups,
    maybe_prefetch_mm_encoder_for_next_iter,
    request_device_paths,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalInput,
    MultimodalItemMetadata,
    MultimodalItemPartition,
    MultimodalParams,
    MultimodalRuntimeData,
    _batch_item_values,
)
from tensorrt_llm.llmapi.llm_args import MultimodalConfig

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

HIDDEN = 8
ROWS = 4  # Patch rows and embedding rows per item.
PATHS = ["image.pixels", "multimodal_embedding", "mrope_config.mrope_position_ids"]


def _image_data(markers, hashes=None):
    """One request with an item per marker: ``ROWS`` rows of ``marker``, packed."""
    pixels = torch.cat([torch.full((ROWS, HIDDEN), float(m)) for m in markers])
    metadata = MultimodalItemMetadata(
        [("image", i) for i in range(len(markers))],
        output_embedding_lengths=[ROWS] * len(markers),
        input_partitions={
            "image": {
                "pixels": MultimodalItemPartition(item_lengths=(ROWS,) * len(markers)),
                "grid": MultimodalItemPartition(),
            }
        },
    )
    return {
        "image": {"pixels": pixels, "grid": torch.tensor([[ROWS]] * len(markers))},
        "mrope_config": {"mrope_position_ids": torch.zeros(3, 1, ROWS * len(markers))},
        "multimodal_embed_mask_cumsum": torch.arange(1, ROWS * len(markers) + 1),
        "mm_processor_kwargs_hash": "kwargs-a",
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
    }


class _GroupModel(MultimodalModelMixin):
    """A model on the default encoder builder with an encoder cache."""

    supports_encoder_cache = True
    multimodal_data_device_paths = PATHS

    def __init__(self):
        self.model_config = ModelConfig(
            multimodal_config=MultimodalConfig(
                encoder_cache_max_bytes=1 << 16, encoder_side_stream_max_ahead=2
            )
        )
        self.calls = []
        self.mm_encoder_groups = (EncoderGroup(("image",), self._encode),)
        self._initialize_multimodal_encoder_cache(1 << 16)

    @property
    def embedding_dim(self):
        return HIDDEN

    @property
    def embedding_dtype(self):
        return torch.float32

    def _encode(self, pixels, grid):
        self.calls.append(
            dict(
                device=pixels.device.type,
                grid_device=grid.device.type,
                stream=torch.cuda.current_stream().cuda_stream,
                markers=pixels.cpu()[::ROWS, 0].tolist(),
            )
        )
        return (pixels * 2).float()

    def encode_multimodal_inputs(self, multimodal_params):
        return encode_multimodal_by_groups(
            self.mm_encoder_groups, list(multimodal_params), self.multimodal_data_device_paths
        )


class _DeviceOnlyModel(_GroupModel):
    """The encoder only queues device work: nothing in it reads the GPU from the host."""

    def _encode(self, pixels, grid):
        self.calls.append(
            dict(device=pixels.device.type, stream=torch.cuda.current_stream().cuda_stream)
        )
        return (pixels * 2).float()


def _host_leaves(data):
    """Every tensor under the raw media keys must still be the request's host tensor."""
    return [
        value
        for modality in ("image", "video", "audio")
        for value in (data.get(modality) or {}).values()
        if isinstance(value, torch.Tensor)
    ]


def _native_allocator():
    """The native caching allocator: 512-byte blocks, a pool per stream (not cudaMallocAsync)."""
    return torch.cuda.get_allocator_backend() == "native"


def _allocated_after_pending_frees():
    """Allocated bytes once the allocator has freed the blocks that waited for a stream (a batch
    made on the copy stream is freed after its consumer stream is done with it)."""
    torch.cuda.synchronize()
    probe = torch.empty(128, device="cuda")  # An allocation processes the finished frees.
    return torch.cuda.memory_allocated() - probe.nbytes


def test_executor_leaves_only_non_media_paths_per_request():
    assert request_device_paths(_GroupModel()) == PATHS[1:]


def test_listed_keys_reach_the_encoder_on_the_gpu_and_payloads_stay_on_the_host():
    model = _GroupModel()
    params = [MultimodalParams(multimodal_data=_image_data(m)) for m in ([1, 2], [3])]
    payloads = [dict(p.multimodal_data["image"]) for p in params]
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = model.encode_multimodal_inputs(params)
    torch.cuda.synchronize()
    (call,) = model.calls
    assert (call["device"], call["grid_device"], call["markers"]) == ("cuda", "cpu", [1, 2, 3])
    # The device holds the call's batch once and the output (384 bytes each, one 512-byte block
    # with the native allocator): no copy per request, no device temporary. The batch is freed
    # once the encoder stream is done with it.
    block = 512 if _native_allocator() else 3 * ROWS * HIDDEN * 4
    assert torch.cuda.max_memory_allocated() - before == 2 * block
    assert _allocated_after_pending_frees() - before == block
    assert out[::ROWS, 0].tolist() == [2, 4, 6]
    for param, payload in zip(params, payloads):
        image = param.multimodal_data["image"]
        assert image.keys() == payload.keys()
        assert all(image[key] is payload[key] for key in payload)
        assert all(leaf.device.type == "cpu" for leaf in _host_leaves(param.multimodal_data))


def test_calls_split_by_dtype_each_copy_their_own_batch():
    model = _GroupModel()
    params = [MultimodalParams(multimodal_data=_image_data(m)) for m in ([1], [2], [3])]
    image = params[1].multimodal_data["image"]
    image["pixels"] = image["pixels"].to(torch.bfloat16)
    model.encode_multimodal_inputs(params)
    assert [(c["device"], c["markers"]) for c in model.calls] == [
        ("cuda", [1, 3]),
        ("cuda", [2]),
    ]


def test_own_builder_reads_payloads_as_the_executor_moved_them():
    group = EncoderGroup(
        ("image",),
        lambda pixels: pixels,
        lambda params: {"pixels": params[0].multimodal_data["image"]["pixels"]},
    )
    param = MultimodalParams(multimodal_data=_image_data([1]))
    model = type("OwnBuilder", (), {"mm_encoder_groups": (group,)})
    model.multimodal_data_device_paths = PATHS
    assert request_device_paths(model) == PATHS
    assert encode_multimodal_by_groups([group], [param], PATHS).device.type == "cpu"


def _cache_request(request_id, hashes):
    data = _image_data(list(range(1, len(hashes) + 1)))
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=[0] * (ROWS * len(hashes)),
        sampling_config=SamplingConfig(beam_width=1),
        is_streaming=False,
        py_multimodal_data=data,
        multimodal_hashes=hashes,
        multimodal_positions=[ROWS * i for i in range(len(hashes))],
        multimodal_lengths=[ROWS] * len(hashes),
        multimodal_uuids=[None] * len(hashes),
    )


def test_prefetch_copies_only_missed_items_on_the_aux_stream():
    model = _GroupModel()
    hit, miss = [1] * 8, [2] * 8
    first = _cache_request(0, [hit])
    assert maybe_prefetch_mm_encoder_for_next_iter(model, [first], max_prefetch_ahead=2) == 1
    first.py_mm_encoder_event.synchronize()
    partial = _cache_request(1, [hit, miss])
    assert maybe_prefetch_mm_encoder_for_next_iter(model, [partial], max_prefetch_ahead=2) == 1
    partial.py_mm_encoder_event.synchronize()
    full = _cache_request(2, [hit, miss])
    assert maybe_prefetch_mm_encoder_for_next_iter(model, [full], max_prefetch_ahead=2) == 1
    full.py_mm_encoder_event.synchronize()

    main = torch.cuda.current_stream().cuda_stream
    # The partial hit encodes item 1 only (marker 2); the full hit runs no encoder.
    assert [(c["device"], c["markers"]) for c in model.calls] == [("cuda", [1]), ("cuda", [2])]
    assert all(c["stream"] != main for c in model.calls)
    for request in (first, partial, full):
        data = request.py_multimodal_data
        # The request holds its whole output now, so its raw media is dropped.
        assert "image" not in data
        assert data["multimodal_embedding"].is_cuda
        # MRoPE still moves per request on the aux stream when the encoder runs.
        assert data["mrope_config"]["mrope_position_ids"].is_cuda == (request is not full)
    torch.testing.assert_close(
        partial.py_multimodal_data["multimodal_embedding"][:, 0].cpu(),
        torch.tensor([2.0] * ROWS + [4.0] * ROWS),
    )


def test_chunked_prefill_encodes_once_and_keeps_no_media_on_the_gpu():
    """The inline path over two chunks: one encoder call, raw media never on the GPU."""
    model = _GroupModel()
    data = _image_data([1, 2])
    data["multimodal_embed_mask_cumsum"] = torch.arange(1, 2 * ROWS + 1)
    paths = request_device_paths(model)
    for begin, end in ((0, ROWS), (ROWS, 2 * ROWS)):
        param = MultimodalParams(
            multimodal_input=MultimodalInput(
                multimodal_hashes=[[1] * 8, [2] * 8],
                multimodal_positions=[0, ROWS],
                multimodal_lengths=[ROWS, ROWS],
            ),
            multimodal_data=data,
            multimodal_runtime=MultimodalRuntimeData(
                embed_mask_cumsum=data["multimodal_embed_mask_cumsum"],
                past_seen_token_num=begin,
                chunk_end_pos=end,
            ),
        )
        param.to_device("multimodal_data", "cuda", pin_memory=True, target_keywords=paths)
        data = param.multimodal_data  # The executor writes the dict back to the request.
        rows = model._get_or_encode_multimodal_embeddings([param])
        assert rows.shape[1] == HIDDEN and rows.is_cuda
        assert all(leaf.device.type == "cpu" for leaf in _host_leaves(data))
    assert [(c["device"], c["markers"]) for c in model.calls] == [("cuda", [1, 2])]


def test_item_path_batches_selected_items_into_one_device_tensor():
    model = _GroupModel()
    requests = [MultimodalParams(multimodal_data=_image_data(m)) for m in ([1, 2, 3], [4, 5])]
    inputs = model.prepare_multimodal_encoder_inputs(
        [(requests[0], 2), (requests[0], 0), (requests[1], 1)]
    )
    for encoder_input, _, _ in inputs:
        encoder_input.to_device(
            "multimodal_data", "cuda", pin_memory=True, target_keywords=request_device_paths(model)
        )
        assert encoder_input.multimodal_data["image"]["pixels"].device.type == "cpu"
    outputs = model.forward_multimodal_encoder_items(inputs)
    assert [(c["device"], c["markers"]) for c in model.calls] == [("cuda", [3, 1, 5])]
    assert [int(o[0, 0]) for o in outputs] == [6, 2, 10]
    assert all(
        leaf.device.type == "cpu" for r in requests for leaf in _host_leaves(r.multimodal_data)
    )


def test_encoder_only_runner_moves_mrope_per_request_and_media_in_the_call(monkeypatch):
    """The encoder-only worker: MRoPE goes to the GPU per request (CUDA IPC handoff), media in the call."""
    from unittest.mock import Mock

    from tensorrt_llm._torch.pyexecutor.engine.runners import mm_encoder as mm_encoder_module
    from tensorrt_llm._torch.pyexecutor.engine.runners.interface import ScheduledInputs
    from tensorrt_llm._torch.pyexecutor.engine.runners.mm_encoder import MultimodalEncoderRunner

    model = _GroupModel()
    vision = SimpleNamespace(
        extra_attrs={},
        mm_encoder_groups=model.mm_encoder_groups,
        multimodal_data_device_paths=PATHS,
        forward=lambda params: [model.encode_multimodal_inputs(params)],
    )
    data = _image_data([1, 2])
    data["mrope_config"]["mrope_position_deltas"] = torch.zeros(1, 1, dtype=torch.int32)
    request = SimpleNamespace(py_multimodal_data=data, multimodal_lengths=[ROWS, ROWS])
    request.get_num_tokens = Mock(return_value=2 * ROWS)
    request.py_multimodal_params = MultimodalParams(multimodal_data=data)
    monkeypatch.setattr(
        mm_encoder_module, "_build_request_multimodal_input", Mock(return_value=None)
    )
    runner = MultimodalEncoderRunner(vision, encoder_cache_enabled=False, moe_load_balancer=None)
    result = runner.forward(
        ScheduledInputs(batch=SimpleNamespace(context_requests=[request])),
        resource_manager=SimpleNamespace(get_resource_manager=Mock(return_value=None)),
    )
    assert [(c["device"], c["markers"]) for c in model.calls] == [("cuda", [1, 2])]
    assert all(leaf.device.type == "cpu" for leaf in _host_leaves(data))
    assert result["mrope_position_ids"][0].is_cuda
    assert result["mm_embeddings"][0].cpu()[::ROWS, 0].tolist() == [2.0, 4.0]


def _mixed_data(image_markers, video_markers):
    """Image and video items that one group batches under the same `pixels` key."""

    def rows(markers):
        return torch.cat([torch.full((ROWS, HIDDEN), float(m)) for m in markers])

    refs = [("image", i) for i in range(len(image_markers))]
    refs += [("video", i) for i in range(len(video_markers))]
    metadata = MultimodalItemMetadata(
        refs,
        output_embedding_lengths=[ROWS] * len(refs),
        input_partitions={
            modality: {"pixels": MultimodalItemPartition(item_lengths=(ROWS,) * len(markers))}
            for modality, markers in (("image", image_markers), ("video", video_markers))
        },
    )
    return {
        "image": {"pixels": rows(image_markers)},
        "video": {"pixels": rows(video_markers)},
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
    }


@pytest.mark.parametrize(
    "paths, placed, calls",
    [
        # A whole modality still moves per request; its values share the GPU batch.
        (["image.pixels", "video"], ("cpu", "cuda"), [("cuda", [1, 2, 3])]),
        # A nested path names no tensor here; the listed key takes the video rows to the GPU.
        (["image.pixels", "video.pixels.scale"], ("cpu", "cpu"), [("cuda", [1, 2, 3])]),
        # Video pixels are kept on the host: they are encoded in their own call.
        (["image.pixels"], ("cpu", "cpu"), [("cuda", [1, 2]), ("cpu", [3])]),
    ],
    ids=["flat_and_whole", "flat_and_nested", "flat_and_unlisted"],
)
def test_mixed_device_paths_keep_each_payload_policy(paths, placed, calls):
    recorded = []

    def encode(pixels):
        recorded.append((pixels.device.type, pixels.cpu()[::ROWS, 0].tolist()))
        return (pixels * 2).cuda()

    group = EncoderGroup(("image", "video"), encode)
    owner = SimpleNamespace(mm_encoder_groups=(group,), multimodal_data_device_paths=paths)
    param = MultimodalParams(multimodal_data=_mixed_data([1, 2], [3]))
    # As the executor does: move the per-request paths, then encode with the owner's paths.
    param.to_device(
        "multimodal_data", "cuda", pin_memory=True, target_keywords=request_device_paths(owner)
    )
    out = encode_multimodal_by_groups([group], [param], paths)
    assert recorded == calls
    assert tuple(param.multimodal_data[m]["pixels"].device.type for m in ("image", "video")) == (
        placed
    )
    expected = torch.cat([torch.full((ROWS, HIDDEN), 2.0 * m) for m in (1, 2, 3)])
    assert out.is_cuda and torch.equal(out.cpu(), expected)


def test_values_on_another_device_are_moved_to_the_batch_device():
    """Media already on a device other than the batch's (loaded on another GPU) is moved to the
    batch's device, alone or batched with host values. With one GPU, the batch is made on the
    host here, and only placement is checked: copies to the host are asynchronous."""
    host = torch.ones(ROWS, HIDDEN)
    device = torch.ones(ROWS, HIDDEN, device="cuda")
    alone = _batch_item_values([device], 0, 0, "cpu")
    mixed = _batch_item_values([host, device], 0, 0, "cpu")
    torch.cuda.synchronize()
    assert alone.device.type == "cpu" and alone.shape == (ROWS, HIDDEN)
    assert mixed.device.type == "cpu" and mixed.shape == (2 * ROWS, HIDDEN)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_values_on_another_gpu_are_copied_to_the_batch_gpu():
    """Media loaded on another GPU (TP > 1) reaches the batch's GPU with its exact contents,
    alone or batched with host values."""
    host = torch.full((ROWS, HIDDEN), 1.0)
    other = torch.full((ROWS, HIDDEN), 2.0, device="cuda:1")
    with torch.cuda.device(0):
        alone = _batch_item_values([other], 0, 0, "cuda:0")
        mixed = _batch_item_values([host, other], 0, 0, "cuda:0")
    assert alone.device == mixed.device == torch.device("cuda", 0)
    assert torch.equal(alone.cpu(), torch.full((ROWS, HIDDEN), 2.0))
    assert torch.equal(mixed.cpu(), torch.cat([host, torch.full((ROWS, HIDDEN), 2.0)]))


def test_prefetch_on_a_delayed_aux_stream_hands_over_by_the_request_event():
    """No host wait: the host returns with the aux work queued, and may then reuse the payload
    memory; the main stream reads the outputs after waiting on the request event only."""
    model = _DeviceOnlyModel()
    request = _cache_request(0, [[1] * 8, [2] * 8])
    data = request.py_multimodal_data
    pixels = data["image"]["pixels"]
    expected = (pixels * 2).float()
    with torch.cuda.stream(_get_mm_aux_stream(2)):
        torch.cuda._sleep(500_000_000)
    assert maybe_prefetch_mm_encoder_for_next_iter(model, [request], max_prefetch_ahead=2) == 1
    event = request.py_mm_encoder_event
    assert not event.query()
    # The call staged its own copy of the payload, so the request drops it and its memory may
    # change from here on.
    assert "image" not in request.py_multimodal_data
    pixels.fill_(-1)
    main = torch.cuda.current_stream()
    main.wait_event(event)
    data = request.py_multimodal_data
    embedding = data["multimodal_embedding"].clone()
    mrope = data["mrope_config"]["mrope_position_ids"].clone()
    torch.cuda.synchronize()  # Only to read the results here.
    assert [(c["device"], c["stream"] != main.cuda_stream) for c in model.calls] == [("cuda", True)]
    assert torch.equal(embedding.cpu(), expected)
    assert torch.equal(mrope.cpu(), torch.zeros(3, 1, 2 * ROWS))


def test_a_call_that_fails_after_its_copy_leaves_the_staging_buffers_usable(monkeypatch):
    """An encoder error after the copies were queued: the next calls get their own bytes and the
    staging buffers serve again, with no buffer beyond the pool."""
    pool = host_staging.PinnedStagingPool(buffer_bytes=ROWS * HIDDEN * 4, buffer_count=2)
    monkeypatch.setattr(
        host_staging, "_STAGED_UPLOADERS", {0: host_staging.StagedUploader(0, pool)}
    )
    model = _DeviceOnlyModel()
    failures = [RuntimeError("encoder failed")]

    def encode(pixels, grid):
        out = _DeviceOnlyModel._encode(model, pixels, grid)
        if failures:
            raise failures.pop()
        return out

    model.mm_encoder_groups = (EncoderGroup(("image",), encode),)
    torch.cuda._sleep(200_000_000)  # The encoder work stays queued while the error propagates.
    with pytest.raises(RuntimeError, match="encoder failed"):
        model.encode_multimodal_inputs([MultimodalParams(multimodal_data=_image_data([1, 2]))])
    out = model.encode_multimodal_inputs([MultimodalParams(multimodal_data=_image_data([3, 4]))])
    torch.cuda.synchronize()
    assert out[::ROWS, 0].tolist() == [6, 8]
    out = model.encode_multimodal_inputs([MultimodalParams(multimodal_data=_image_data([5, 6]))])
    torch.cuda.synchronize()
    assert out[::ROWS, 0].tolist() == [10, 12]
    assert 1 <= len(pool.buffers) <= 2


def test_startup_profile_makes_the_batch_in_the_copy_stream_pool(monkeypatch):
    """The startup encoder profile goes through the same uploader as runtime calls, so the copy
    stream's allocator pool that the KV-cache estimate counts holds the profiled batch (native
    allocator; cudaMallocAsync keeps no pool apart and reports 0)."""
    from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

    uploader = host_staging.StagedUploader(0)
    monkeypatch.setattr(host_staging, "_STAGED_UPLOADERS", {0: uploader})
    model = _DeviceOnlyModel()
    param = MultimodalParams(multimodal_data=_image_data([1, 2, 3]))
    creator = object.__new__(KvCacheCreator)
    creator._dummy_encoder_inputs = [param]
    creator._model_engine = SimpleNamespace(model=model)
    assert uploader.reserved_bytes() == 0
    out = creator._encode_dummy_inputs()
    torch.cuda.synchronize()
    assert [c["device"] for c in model.calls] == ["cuda"]
    assert out[::ROWS, 0].tolist() == [2, 4, 6]
    assert (uploader.reserved_bytes() > 0) == _native_allocator()
    assert all(leaf.device.type == "cpu" for leaf in _host_leaves(param.multimodal_data))


def test_item_path_encodes_released_items_again_from_the_host_payload():
    model = _GroupModel()
    request = MultimodalParams(multimodal_data=_image_data([1, 2, 3]))
    pixels = request.multimodal_data["image"]["pixels"]
    outputs = []
    # Item 1 is encoded, released (pause), then scheduled again with item 2.
    for selection in ([(request, 1)], [(request, 1), (request, 2)]):
        inputs = model.prepare_multimodal_encoder_inputs(selection)
        for encoder_input, _, _ in inputs:
            encoder_input.to_device(
                "multimodal_data",
                "cuda",
                pin_memory=True,
                target_keywords=request_device_paths(model),
            )
        outputs.append([int(o[0, 0]) for o in model.forward_multimodal_encoder_items(inputs)])
    assert [(c["device"], c["markers"]) for c in model.calls] == [
        ("cuda", [2]),
        ("cuda", [2, 3]),
    ]
    assert outputs == [[4], [4, 6]]
    assert request.multimodal_data["image"]["pixels"] is pixels and not pixels.is_cuda


def test_preparer_chunks_then_replays_the_prefill_from_the_host_payload():
    """The decoder input preparer over two chunks, a recompute reset that replays the prefill,
    then a replay after the encoder outputs were released: raw media stays on the host and is
    encoded from there whenever no output is kept."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer

    model = _DeviceOnlyModel()
    preparer = object.__new__(DecoderInputPreparer)
    preparer.model = model
    preparer._config = SimpleNamespace(mm_encoder_cache_enabled=True)
    preparer._mm_item_scheduler = None
    request = _cache_request(0, [[1] * 8, [2] * 8])
    pixels = request.py_multimodal_data["image"]["pixels"]

    def prefill():
        rows = []
        for begin, end in ((0, ROWS), (ROWS, 2 * ROWS)):
            lists = SimpleNamespace(
                position_ids=list(range(begin, end)),
                mrope_position_ids=[],
                mrope_delta_write_seq_slots=[],
                multimodal_params_list=[],
            )
            host = SimpleNamespace(use_mrope=False, lists=lists)
            preparer._pack_context_multimodal(
                host, request, begin, end, begin, [0] * (end - begin), 0
            )
            (params,) = lists.multimodal_params_list
            rows.append(model._get_or_encode_multimodal_embeddings([params]))
            assert all(
                leaf.device.type == "cpu" for leaf in _host_leaves(request.py_multimodal_data)
            )
        return rows

    torch.cuda._sleep(200_000_000)  # No host sync between the chunks.
    rows = prefill()
    request.reset_for_recompute(1 << 20)
    rows += prefill()  # The replay reuses the output the request still holds.
    assert len(model.calls) == 1
    # Outputs released (cache entries evicted, no embedding kept): encode again from the host.
    request.py_multimodal_data.pop("multimodal_embedding")
    model._multimodal_encoder_cache.clear()
    rows += prefill()
    torch.cuda.synchronize()
    assert [c["device"] for c in model.calls] == ["cuda", "cuda"]
    # Without item scheduling every chunk gathers the request's whole output.
    assert [r.cpu()[::ROWS, 0].tolist() for r in rows] == [[2.0, 4.0]] * 6
    assert request.py_multimodal_data["image"]["pixels"] is pixels


def test_executor_drops_host_media_between_chunks_once_the_output_is_held():
    """Chunked prefill through the decoder input preparer and the executor's request update: the
    first chunk encodes, then the request drops its raw media (it holds the whole output) and
    keeps MRoPE data, item metadata and the embedding; the second chunk reuses the output, and
    prefill completion clears the rest as before."""
    from tensorrt_llm._torch.pyexecutor.engine.runners.decoder.inputs import DecoderInputPreparer
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

    model = _DeviceOnlyModel()
    preparer = object.__new__(DecoderInputPreparer)
    preparer.model = model
    preparer._config = SimpleNamespace(mm_encoder_cache_enabled=True)
    preparer._mm_item_scheduler = None
    executor = SimpleNamespace(
        _terminate_attention_dp_dummy_requests=lambda: None,
        _is_kv_manager_v2=False,
        disable_overlap_scheduler=True,
        model_engine=SimpleNamespace(mm_encoder_cache=model._multimodal_encoder_cache),
    )
    executor._release_multimodal_resources = functools.partial(
        PyExecutor._release_multimodal_resources, executor
    )
    request = _cache_request(0, [[1] * 8, [2] * 8])
    pixels = request.py_multimodal_data["image"]["pixels"]
    rows = []
    for begin, end in ((0, ROWS), (ROWS, 2 * ROWS)):
        lists = SimpleNamespace(
            position_ids=list(range(begin, end)),
            mrope_position_ids=[],
            mrope_delta_write_seq_slots=[],
            multimodal_params_list=[],
        )
        host = SimpleNamespace(use_mrope=False, lists=lists)
        preparer._pack_context_multimodal(host, request, begin, end, begin, [0] * (end - begin), 0)
        (params,) = lists.multimodal_params_list
        rows.append(model._get_or_encode_multimodal_embeddings([params]))
        request.context_chunk_size = end - begin
        PyExecutor._update_request_states_tp(executor, SimpleNamespace(context_requests=[request]))
        if end < 2 * ROWS:
            data = request.py_multimodal_data
            assert "image" not in data
            assert {
                "multimodal_embedding",
                "mrope_config",
                "multimodal_embed_mask_cumsum",
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
            } <= data.keys()
    torch.cuda.synchronize()
    assert len(model.calls) == 1
    assert [r.cpu()[::ROWS, 0].tolist() for r in rows] == [[2.0, 4.0]] * 2
    assert not request.py_multimodal_data  # No MRoPE deltas to keep for generation here.
    assert not pixels.is_cuda
