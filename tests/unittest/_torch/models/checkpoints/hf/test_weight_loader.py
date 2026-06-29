from unittest import mock

import pytest

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import ConsumableWeightsDict
from tensorrt_llm.mapping import Mapping


class MyError(Exception):
    pass


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
    HfWeightLoader.clear_weight_cache()

    checkpoint_dir = tmp_path / "foo"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()

    raw_weight = object()
    loader = HfWeightLoader()

    try:
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
    finally:
        HfWeightLoader.clear_weight_cache()


def test_weight_cache_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("TRTLLM_HF_WEIGHT_CACHE", raising=False)
    HfWeightLoader.clear_weight_cache()

    checkpoint_dir = tmp_path / "foo"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model.safetensors").touch()

    loader = HfWeightLoader()

    try:
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
    finally:
        HfWeightLoader.clear_weight_cache()
