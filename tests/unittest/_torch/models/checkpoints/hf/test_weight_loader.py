# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest import mock

import pytest
import safetensors
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm.mapping import Mapping


class MyError(Exception):
    pass


def test_layerwise_safetensors_keeps_cross_shard_layer_atomic(tmp_path):
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    shard2 = tmp_path / "model-00002-of-00002.safetensors"
    save_file(
        {
            "embed.weight": torch.tensor([1.0]),
            "layers.0.attn.compressor.wkv.weight": torch.tensor([2.0]),
        },
        str(shard1),
    )
    save_file(
        {
            "layers.0.attn.compressor.wgate.weight": torch.tensor([3.0]),
            "layers.1.attn.wq_b.weight": torch.tensor([4.0]),
            "mtp.0.enorm.weight": torch.tensor([5.0]),
        },
        str(shard2),
    )
    weight_map = {
        "embed.weight": shard1.name,
        "layers.0.attn.compressor.wkv.weight": shard1.name,
        "layers.0.attn.compressor.wgate.weight": shard2.name,
        "layers.1.attn.wq_b.weight": shard2.name,
        "mtp.0.enorm.weight": shard2.name,
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}), encoding="utf-8"
    )

    loader = HfWeightLoader()
    buckets = []
    for bucket in loader.iter_layer_weight_buckets(str(tmp_path), mapping=Mapping()):
        buckets.append(set(bucket.keys()))

    assert buckets == [
        {"embed.weight"},
        {
            "layers.0.attn.compressor.wkv.weight",
            "layers.0.attn.compressor.wgate.weight",
        },
        {"layers.1.attn.wq_b.weight"},
        {"mtp.0.enorm.weight"},
    ]


def test_layerwise_safetensors_without_index_discovers_keys(tmp_path):
    save_file(
        {
            "model.layers.2.foo.weight": torch.tensor([2.0]),
            "lm_head.weight": torch.tensor([1.0]),
        },
        str(tmp_path / "model.safetensors"),
    )

    loader = HfWeightLoader()
    buckets = [
        set(bucket.keys())
        for bucket in loader.iter_layer_weight_buckets(str(tmp_path), mapping=Mapping())
    ]

    assert buckets == [{"lm_head.weight"}, {"model.layers.2.foo.weight"}]


def test_layerwise_safetensors_rejects_missing_index_shard(tmp_path):
    shard1 = tmp_path / "model-00001-of-00002.safetensors"
    save_file({"layers.0.foo.weight": torch.tensor([1.0])}, str(shard1))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.foo.weight": shard1.name,
                    "layers.1.foo.weight": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    loader = HfWeightLoader()
    with pytest.raises(RuntimeError, match="missing checkpoint files"):
        next(loader.iter_layer_weight_buckets(str(tmp_path), mapping=Mapping()))


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "Yes", "on", "ON"])
def test_layerwise_safetensors_environment_true_values(monkeypatch, value):
    monkeypatch.setenv("TRTLLM_HF_LAYERWISE_SAFETENSORS", value)

    assert HfWeightLoader.is_layerwise_safetensors_enabled()


@pytest.mark.parametrize("value", [None, "0", "false", "no", "off", "unexpected"])
def test_layerwise_safetensors_environment_false_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TRTLLM_HF_LAYERWISE_SAFETENSORS", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_HF_LAYERWISE_SAFETENSORS", value)

    assert not HfWeightLoader.is_layerwise_safetensors_enabled()


def test_layerwise_safetensors_uses_natural_layer_order_and_preserves_values(tmp_path):
    save_file(
        {
            "layers.10.weight": torch.tensor([10.0]),
            "layers.2.weight": torch.tensor([2.0]),
            "mtp.1.weight": torch.tensor([101.0]),
            "mtp.0.weight": torch.tensor([100.0]),
        },
        str(tmp_path / "model.safetensors"),
    )

    buckets = list(HfWeightLoader().iter_layer_weight_buckets(str(tmp_path), mapping=Mapping()))

    assert [next(iter(bucket)) for bucket in buckets] == [
        "layers.2.weight",
        "layers.10.weight",
        "mtp.0.weight",
        "mtp.1.weight",
    ]
    assert [next(iter(bucket.values())).item() for bucket in buckets] == [2.0, 10.0, 100.0, 101.0]


def test_layerwise_safetensors_rejects_empty_index(tmp_path):
    save_file({"layers.0.weight": torch.tensor([1.0])}, str(tmp_path / "model.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="empty weight_map"):
        next(HfWeightLoader().iter_layer_weight_buckets(str(tmp_path), mapping=Mapping()))


def test_layerwise_safetensors_rejects_duplicate_keys_without_index(tmp_path):
    for shard_idx in range(2):
        save_file(
            {"layers.0.weight": torch.tensor([float(shard_idx)])},
            str(tmp_path / f"model-{shard_idx}.safetensors"),
        )

    with pytest.raises(RuntimeError, match="Duplicate tensor key"):
        next(HfWeightLoader().iter_layer_weight_buckets(str(tmp_path), mapping=Mapping()))


def test_layerwise_safetensors_strictly_selects_consolidated_files(tmp_path):
    save_file({"layers.0.weight": torch.tensor([1.0])}, str(tmp_path / "model.safetensors"))
    save_file(
        {"layers.1.weight": torch.tensor([2.0])},
        str(tmp_path / "model-consolidated.safetensors"),
    )

    ordinary = list(HfWeightLoader().iter_layer_weight_buckets(str(tmp_path), mapping=Mapping()))
    consolidated = list(
        HfWeightLoader().iter_layer_weight_buckets(
            str(tmp_path), mapping=Mapping(), use_consolidated=True
        )
    )

    assert [set(bucket) for bucket in ordinary] == [{"layers.0.weight"}]
    assert [set(bucket) for bucket in consolidated] == [{"layers.1.weight"}]


@pytest.mark.parametrize("use_consolidated", [False, True])
def test_layerwise_safetensors_does_not_fallback_to_wrong_file_kind(tmp_path, use_consolidated):
    filename = "model.safetensors" if use_consolidated else "model-consolidated.safetensors"
    save_file({"layers.0.weight": torch.tensor([1.0])}, str(tmp_path / filename))

    with pytest.raises(RuntimeError, match="requires .*safetensors weights"):
        next(
            HfWeightLoader().iter_layer_weight_buckets(
                str(tmp_path), mapping=Mapping(), use_consolidated=use_consolidated
            )
        )


def test_layerwise_safetensors_handles_follow_bucket_lifetime(tmp_path, monkeypatch):
    shard = tmp_path / "model.safetensors"
    save_file(
        {
            "layers.0.weight": torch.tensor([1.0]),
            "layers.1.weight": torch.tensor([2.0]),
        },
        str(shard),
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layers.0.weight": shard.name,
                    "layers.1.weight": shard.name,
                }
            }
        ),
        encoding="utf-8",
    )
    opened = []
    closed = []
    real_safe_open = safetensors.safe_open

    class RecordingHandle:
        def __init__(self, path, **kwargs):
            self._context = real_safe_open(path, **kwargs)
            self._path = path

        def __enter__(self):
            self._handle = self._context.__enter__()
            opened.append(self._path)
            return self._handle

        def __exit__(self, *args):
            closed.append(self._path)
            return self._context.__exit__(*args)

    monkeypatch.setattr(safetensors, "safe_open", RecordingHandle)
    iterator = HfWeightLoader().iter_layer_weight_buckets(str(tmp_path), mapping=Mapping())

    first = next(iterator)
    assert first["layers.0.weight"].item() == 1.0
    assert len(opened) == 1
    assert not closed

    second = next(iterator)
    assert second["layers.1.weight"].item() == 2.0
    assert len(opened) == 2
    assert len(closed) == 1

    iterator.close()
    assert len(closed) == 2


def test_layerwise_safetensors_closes_handle_on_consumer_exception(tmp_path, monkeypatch):
    shard = tmp_path / "model.safetensors"
    save_file({"layers.0.weight": torch.tensor([1.0])}, str(shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layers.0.weight": shard.name}}), encoding="utf-8"
    )
    closed = []
    real_safe_open = safetensors.safe_open

    class RecordingHandle:
        def __init__(self, path, **kwargs):
            self._context = real_safe_open(path, **kwargs)

        def __enter__(self):
            return self._context.__enter__()

        def __exit__(self, *args):
            closed.append(True)
            return self._context.__exit__(*args)

    monkeypatch.setattr(safetensors, "safe_open", RecordingHandle)

    with pytest.raises(MyError):
        for _bucket in HfWeightLoader().iter_layer_weight_buckets(str(tmp_path), mapping=Mapping()):
            raise MyError

    assert closed == [True]


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
