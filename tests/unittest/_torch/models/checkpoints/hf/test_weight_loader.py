from unittest import mock

import pytest

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.hf import weight_loader
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


def test_prefetch_one_file_reads_in_bounded_chunks(monkeypatch):
    class FakeFile:
        def __init__(self):
            self.remaining_bytes = 10
            self.read_sizes = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def readinto(self, buffer):
            self.read_sizes.append(len(buffer))
            if self.remaining_bytes == 0:
                return 0
            read_size = min(len(buffer), self.remaining_bytes)
            buffer[:read_size] = b"x" * read_size
            self.remaining_bytes -= read_size
            return read_size

    fake_file = FakeFile()

    monkeypatch.setattr(weight_loader.os.path, "exists", lambda _: True)
    monkeypatch.setattr(weight_loader, "_PREFETCH_CHUNK_SIZE", 4)
    monkeypatch.setattr("builtins.open", lambda *_, **__: fake_file)

    HfWeightLoader()._prefetch_one_file("checkpoint.safetensors")

    assert fake_file.read_sizes == [4, 4, 4, 4]


def test_prefetch_files_caps_per_rank_workers(monkeypatch):
    class FakeThreadPoolExecutor:
        def __init__(self, max_workers):
            self.max_workers = max_workers

        def __enter__(self):
            observed_max_workers.append(self.max_workers)
            return self

        def __exit__(self, *args):
            return False

        def map(self, _func, file_names):
            return [None for _ in file_names]

    observed_max_workers = []
    file_names = [f"model-{idx}.safetensors" for idx in range(4)]

    monkeypatch.setattr(weight_loader, "ThreadPoolExecutor", FakeThreadPoolExecutor)
    monkeypatch.setattr(weight_loader, "local_mpi_rank", lambda: 0)
    monkeypatch.setattr(weight_loader, "local_mpi_size", lambda: 1)

    HfWeightLoader().prefetch_files(file_names)

    assert observed_max_workers == [weight_loader._MAX_PREFETCH_WORKERS_PER_RANK]
