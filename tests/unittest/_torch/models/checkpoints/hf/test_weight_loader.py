# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mmap
import os
import threading
from unittest import mock

import pytest

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only


class MyError(Exception):
    pass


@pytest.fixture(autouse=True)
def clean_weight_cache():
    HfWeightLoader._clear_weight_cache()
    yield
    HfWeightLoader._clear_weight_cache()


@pytest.mark.parametrize(
    "dir_name, safetensor_filenames, expected_safetensor_filenames, use_consolidated",
    [
        (
            "foo",
            [
                "model-00001-of-00002.safetensors",
                "model-000002-of-00002.safetensors",
                "consolidated.safetensors",
            ],
            ["model-00001-of-00002.safetensors", "model-000002-of-00002.safetensors"],
            False,
        ),
        # If use_consolidated specified explicitly.
        (
            "foo",
            [
                "model-00001-of-00002.safetensors",
                "model-000002-of-00002.safetensors",
                "consolidated.safetensors",
            ],
            ["consolidated.safetensors"],
            True,
        ),
        (
            "foo",
            [
                *(f"model-0000{i}-of-00010.safetensors" for i in range(1, 11)),
                "foo-consolidated.safetensors",
            ],
            [f"model-0000{i}-of-00010.safetensors" for i in range(1, 11)],
            False,
        ),
        # If there is only a consolidated safetensor, that one should still be used.
        (
            "foo",
            ["consolidated.safetensors"],
            ["consolidated.safetensors"],
            False,
        ),
        # If the directory contains "consolidated" in its name, but its contents are sharded tensors.
        (
            "consolidated-model",
            [
                "model-00001-of-00002.safetensors",
                "model-000002-of-00002.safetensors",
                "consolidated.safetensors",
            ],
            ["model-00001-of-00002.safetensors", "model-000002-of-00002.safetensors"],
            False,
        ),
    ],
)
def test_load_weights_ignores_consolidated_ckpt_when_sharded_ckpt_exists(
    tmp_path,
    dir_name: str,
    safetensor_filenames: list[str],
    expected_safetensor_filenames: list[str],
    use_consolidated: bool,
):
    checkpoint_dir = tmp_path / dir_name
    checkpoint_dir.mkdir()
    for filename in safetensor_filenames:
        (checkpoint_dir / filename).touch()
    expected_safetensor_filenames = set(
        str(checkpoint_dir / filename) for filename in expected_safetensor_filenames
    )

    loader = HfWeightLoader()
    with (
        mock.patch.object(
            loader, "_load_weights_in_parallel", side_effect=MyError
        ) as load_weights_in_parallel,
        mock.patch.object(loader, "prefetch_files") as prefetch_files,
        pytest.raises(MyError),
    ):
        loader.load_weights(
            checkpoint_dir=str(checkpoint_dir), mapping=Mapping(), use_consolidated=use_consolidated
        )

    prefetch_files.assert_called_once()
    prefetched_files = prefetch_files.call_args[0][0]
    assert set(prefetched_files) == expected_safetensor_filenames

    load_weights_in_parallel.assert_called_once()
    loaded_weight_files = load_weights_in_parallel.call_args[0][0]
    assert set(loaded_weight_files) == expected_safetensor_filenames


def test_bin_loader_propagates_fallback_error():
    loader = HfWeightLoader()
    with (
        mock.patch(
            "tensorrt_llm._torch.models.checkpoints.hf.weight_loader.torch.load",
            side_effect=[OSError("mmap failed"), MyError("fallback failed")],
        ) as torch_load,
        pytest.raises(MyError, match="fallback failed"),
    ):
        loader._load_bin_or_path_file("model.bin")

    assert torch_load.call_args_list == [
        mock.call("model.bin", weights_only=True, map_location="cpu", mmap=True),
        mock.call("model.bin", weights_only=True, map_location="cpu", mmap=False),
    ]


def test_weight_cache_reuses_raw_weights_with_fresh_consumable_wrapper(tmp_path, monkeypatch):
    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "1")

    checkpoint_dir = tmp_path / "foo"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()

    raw_weight = object()
    loader = HfWeightLoader()

    with (
        mock.patch.object(
            loader,
            "_load_weights_in_parallel",
            return_value=ConsumableWeightsDict({"foo.weight": raw_weight}),
        ) as load_weights_in_parallel,
        mock.patch.object(loader, "prefetch_files"),
    ):
        first = loader.load_weights(str(checkpoint_dir), mapping=Mapping())
        assert first["foo.weight"] is raw_weight
        assert first.mark_consumed("foo") == 1
        assert len(first) == 0

        second = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    load_weights_in_parallel.assert_called_once()
    assert second["foo.weight"] is raw_weight


def test_weight_cache_evicts_before_load_on_miss(tmp_path, monkeypatch):
    # On a cross-model miss with a full cache (max_entries=1), the old entry
    # must be evicted BEFORE the new weights load, so CPU never holds both the
    # old (cached) and new (loading) weights at once (no transient 2x peak).
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "1")
    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE_MAX_ENTRIES", "1")

    dir_a = tmp_path / "a"
    dir_a.mkdir()
    (dir_a / "model.safetensors").touch()
    dir_b = tmp_path / "b"
    dir_b.mkdir()
    (dir_b / "model.safetensors").touch()

    loader = HfWeightLoader()
    with mock.patch.object(loader, "prefetch_files"):
        with mock.patch.object(
            loader,
            "_load_weights_in_parallel",
            return_value=ConsumableWeightsDict({"foo.weight": object()}),
        ):
            loader.load_weights(str(dir_a), mapping=Mapping())
        assert len(wl._WEIGHT_CACHE) == 1

        def assert_room_freed_before_load(*args, **kwargs):
            # The old (A) entry must already be gone by the time B loads.
            assert len(wl._WEIGHT_CACHE) == 0
            return ConsumableWeightsDict({"foo.weight": object()})

        with mock.patch.object(
            loader,
            "_load_weights_in_parallel",
            side_effect=assert_room_freed_before_load,
        ):
            loader.load_weights(str(dir_b), mapping=Mapping())

    assert len(wl._WEIGHT_CACHE) == 1


def test_weight_cache_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_CACHE", raising=False)

    checkpoint_dir = tmp_path / "foo"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()

    loader = HfWeightLoader()

    with (
        mock.patch.object(
            loader,
            "_load_weights_in_parallel",
            side_effect=[
                ConsumableWeightsDict({"foo.weight": object()}),
                ConsumableWeightsDict({"foo.weight": object()}),
            ],
        ) as load_weights_in_parallel,
        mock.patch.object(loader, "prefetch_files"),
    ):
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert load_weights_in_parallel.call_count == 2


def test_weight_cache_detects_inplace_mutation_and_reloads(tmp_path, monkeypatch):
    # The cache shares raw tensors across loads (read-only by contract). A
    # consumer mutating one in place (e.g. an in-place transform in a weight
    # mapper) must be detected on the next hit: the poisoned entry is dropped
    # and the weights are reloaded from disk instead of silently corrupted.
    import torch

    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "1")

    checkpoint_dir = tmp_path / "foo"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()

    loader = HfWeightLoader()

    def fresh_weights(*args, **kwargs):
        return ConsumableWeightsDict({"a.weight": torch.ones(64)})

    with (
        mock.patch.object(
            loader, "_load_weights_in_parallel", side_effect=fresh_weights
        ) as load_weights_in_parallel,
        mock.patch.object(loader, "prefetch_files"),
    ):
        first = loader.load_weights(str(checkpoint_dir), mapping=Mapping())
        first["a.weight"].neg_()  # in-place mutation through the shared tensor

        second = loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert load_weights_in_parallel.call_count == 2  # poisoned hit -> reload
    assert torch.equal(second["a.weight"], torch.ones(64))  # clean weights


def test_cache_hit_and_miss_issue_identical_collectives(tmp_path, monkeypatch):
    # Rank-local caches can diverge, so a hit on one rank and a miss on
    # another must enqueue the SAME collectives in the same order (Allreduce
    # from _get_local_available_host_memory, then Barrier) or the job
    # deadlocks. Record the sequence each path produces and compare.
    monkeypatch.setenv("TRTLLM_HF_WEIGHT_CACHE", "1")

    checkpoint_dir = tmp_path / "foo"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()

    loader = HfWeightLoader()
    sequences = {"miss": [], "hit": []}
    current: list = []

    monkeypatch.setattr(
        "tensorrt_llm._torch.models.checkpoints.hf.weight_loader.local_mpi_barrier",
        lambda: current.append("barrier"),
    )

    def record_allreduce():
        current.append("allreduce")
        return 1 << 60  # plenty of host memory

    with (
        mock.patch.object(
            loader,
            "_load_weights_in_parallel",
            return_value=ConsumableWeightsDict({"foo.weight": object()}),
        ),
        mock.patch.object(loader, "prefetch_files"),
        mock.patch.object(
            HfWeightLoader,
            "_get_local_available_host_memory",
            side_effect=record_allreduce,
        ),
    ):
        current = sequences["miss"]
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())
        current = sequences["hit"]
        loader.load_weights(str(checkpoint_dir), mapping=Mapping())

    assert sequences["miss"] == ["allreduce", "barrier"]
    assert sequences["hit"] == sequences["miss"]  # divergence-safe ordering


def test_prefetch_one_file_reports_full_file_in_bounded_windows(tmp_path, monkeypatch):
    # Prefetch must never hold more than one window per thread in memory:
    # whole-file reads pinned up to hundreds of GB across ranks on slow
    # storage and OOMed the host. The madvise-populate path and the chunked
    # read fallback share the same window size, so this holds on whichever
    # path the host kernel supports; every byte must be reported exactly
    # once, including a trailing partial window.
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", mmap.PAGESIZE)
    file = tmp_path / "model.safetensors"
    payload = os.urandom(3 * mmap.PAGESIZE + 137)  # not a multiple of the window size
    file.write_bytes(payload)

    reported: list[int] = []
    HfWeightLoader()._prefetch_one_file(str(file), report_progress=reported.append)

    assert sum(reported) == len(payload)
    assert max(reported) <= mmap.PAGESIZE


def test_prefetch_one_file_chunked_fallback_when_populate_unsupported(tmp_path, monkeypatch):
    # Force the MADV_POPULATE_READ path to report "unsupported" so the chunked
    # readinto fallback is exercised even on kernels that support population.
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", mmap.PAGESIZE)
    monkeypatch.setattr(wl, "populate_file_pages", lambda *args, **kwargs: 0)
    file = tmp_path / "model.safetensors"
    payload = os.urandom(2 * mmap.PAGESIZE + 57)
    file.write_bytes(payload)

    reported: list[int] = []
    HfWeightLoader()._prefetch_one_file(str(file), report_progress=reported.append)

    assert sum(reported) == len(payload)
    assert max(reported) <= mmap.PAGESIZE


def test_prefetch_one_file_resumes_reads_after_partial_populate(tmp_path, monkeypatch):
    # If population stops early (e.g. a transient madvise failure), the
    # chunked reads must resume from that offset: every byte prefetched and
    # reported exactly once, none twice.
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", mmap.PAGESIZE)

    def partial_populate(file_name, window_bytes, on_window=None):
        if on_window is not None:
            on_window(mmap.PAGESIZE)
        return mmap.PAGESIZE

    monkeypatch.setattr(wl, "populate_file_pages", partial_populate)
    file = tmp_path / "model.safetensors"
    payload = os.urandom(3 * mmap.PAGESIZE + 137)
    file.write_bytes(payload)

    reported: list[int] = []
    HfWeightLoader()._prefetch_one_file(str(file), report_progress=reported.append)

    assert sum(reported) == len(payload)


def test_prefetch_one_file_missing_file_is_noop():
    # Missing files must be silently skipped (no exception).
    HfWeightLoader()._prefetch_one_file("/nonexistent/model.safetensors")


def test_prefetch_one_file_logs_partial_populate_once(tmp_path, monkeypatch):
    # A populate that consistently stops partway must be visible in the log
    # (once per process, with the populated byte count) — otherwise every
    # file silently pays for both a partial populate and a chunked read.
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", mmap.PAGESIZE)
    monkeypatch.setattr(wl, "_PREFETCH_FALLBACK_LOGGED", threading.Event())

    def partial_populate(file_name, window_bytes, on_window=None):
        if on_window is not None:
            on_window(mmap.PAGESIZE)
        return mmap.PAGESIZE

    monkeypatch.setattr(wl, "populate_file_pages", partial_populate)
    file = tmp_path / "model.safetensors"
    file.write_bytes(os.urandom(2 * mmap.PAGESIZE))

    with mock.patch.object(wl.logger, "info") as info:
        HfWeightLoader()._prefetch_one_file(str(file))
        HfWeightLoader()._prefetch_one_file(str(file))  # must not log again

    partial_logs = [c for c in info.call_args_list if "stopped after" in str(c)]
    assert len(partial_logs) == 1
    assert str(mmap.PAGESIZE) in str(partial_logs[0])


def test_prefetch_one_file_fallback_logs_once_across_threads(tmp_path, monkeypatch):
    # The once-per-process fallback log must stay single-shot when many files
    # fall back concurrently: the check-and-set is atomic under a lock.
    from concurrent.futures import ThreadPoolExecutor

    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", mmap.PAGESIZE)
    monkeypatch.setattr(wl, "_PREFETCH_FALLBACK_LOGGED", threading.Event())

    num_threads = 8
    barrier = threading.Barrier(num_threads)

    def unsupported_populate(file_name, window_bytes, on_window=None):
        barrier.wait()  # line every thread up just before the guarded log
        return 0

    monkeypatch.setattr(wl, "populate_file_pages", unsupported_populate)
    files = []
    for i in range(num_threads):
        file = tmp_path / f"model-{i}.safetensors"
        file.write_bytes(os.urandom(mmap.PAGESIZE))
        files.append(str(file))

    loader = HfWeightLoader()
    with mock.patch.object(wl.logger, "info") as info:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            list(pool.map(loader._prefetch_one_file, files))

    fallback_logs = [c for c in info.call_args_list if "chunked reads" in str(c)]
    assert len(fallback_logs) == 1


def test_read_file_in_chunks_no_buffer_when_nothing_to_read(tmp_path, monkeypatch):
    # When population covered the whole file, the fallback must return before
    # allocating its buffer: with this chunk size an allocation attempt would
    # raise MemoryError, so returning 0 proves the allocation was skipped.
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", 1 << 62)
    payload = os.urandom(mmap.PAGESIZE)
    file = tmp_path / "model.safetensors"
    file.write_bytes(payload)

    assert HfWeightLoader._read_file_in_chunks(str(file), offset=len(payload)) == 0


def test_prefetch_one_file_no_fallback_log_when_fully_populated(tmp_path, monkeypatch):
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", mmap.PAGESIZE)
    monkeypatch.setattr(wl, "_PREFETCH_FALLBACK_LOGGED", threading.Event())
    payload = os.urandom(2 * mmap.PAGESIZE)

    def full_populate(file_name, window_bytes, on_window=None):
        if on_window is not None:
            on_window(len(payload))
        return len(payload)

    monkeypatch.setattr(wl, "populate_file_pages", full_populate)
    file = tmp_path / "model.safetensors"
    file.write_bytes(payload)

    with mock.patch.object(wl.logger, "info") as info:
        HfWeightLoader()._prefetch_one_file(str(file))

    assert not [c for c in info.call_args_list if "chunked reads" in str(c)]


def test_prefetch_files_emits_progress_heartbeat(tmp_path, monkeypatch):
    # The heartbeat is what keeps a slow prefetch observable (and alive under
    # output-stall watchdogs); with the log interval forced to zero it must
    # fire for every chunk.
    from tensorrt_llm._torch.models.checkpoints.hf import weight_loader as wl

    monkeypatch.setattr(wl, "_PREFETCH_CHUNK_SIZE_BYTES", mmap.PAGESIZE)
    monkeypatch.setattr(wl, "_PREFETCH_LOG_INTERVAL_SEC", 0.0)
    # prefetch_files shards its input across local MPI ranks; pin to a single
    # rank so the asserted file set does not depend on how tests are launched.
    monkeypatch.setattr(wl, "local_mpi_rank", lambda: 0)
    monkeypatch.setattr(wl, "local_mpi_size", lambda: 1)
    files = []
    for i in range(3):
        file = tmp_path / f"model-0000{i}-of-00003.safetensors"
        file.write_bytes(os.urandom(4 * mmap.PAGESIZE))
        files.append(str(file))

    with mock.patch.object(wl.logger, "info") as info:
        HfWeightLoader().prefetch_files(files)

    progress_logs = [call for call in info.call_args_list if "Prefetch progress" in str(call)]
    # Every chunk logs when the interval is zero: 3 files x 4 KB at a 1 KB
    # chunk size means at least 12 heartbeats (short reads only add more).
    assert len(progress_logs) >= 12


def test_kimi_k3_lazy_load_records_the_checkpoint_dir(tmp_path):
    """A model that re-opens shards itself needs the directory back.

    Kimi K3 streams its rank-local experts per shard file to avoid holding
    the whole mapping open. A lazy slice does not carry its file, and
    transformers 5.x no longer sets ``PretrainedConfig._name_or_path``, so
    without this the model silently fell back to the shared mapping and the
    step was OOM-killed.
    """
    import json

    import safetensors.torch
    import torch

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "kimi_k3"}))
    safetensors.torch.save_file(
        {"w": torch.zeros(2, 2)}, tmp_path / "model-00001-of-00001.safetensors"
    )

    loader = HfWeightLoader()
    try:
        weights = loader.load_weights(str(tmp_path), Mapping())
        assert isinstance(weights, ConsumableWeightsDict)
        assert weights.checkpoint_dir == str(tmp_path)
    finally:
        loader.cleanup()
