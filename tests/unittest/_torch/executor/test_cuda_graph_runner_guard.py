from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import (
    should_skip_cuda_graph_for_deepseek_v4_mtp,
)


class _SpecDecMode:

    def __init__(self, *, mtp: bool):
        self._mtp = mtp

    def is_mtp_one_model(self) -> bool:
        return self._mtp

    def is_mtp_vanilla(self) -> bool:
        return False

    def is_mtp_eagle(self) -> bool:
        return False

    def is_mtp_eagle_one_model(self) -> bool:
        return False


def _spec_config(*, mtp: bool):
    return SimpleNamespace(spec_dec_mode=_SpecDecMode(mtp=mtp))


def _sparse_config(algorithm: str = "deepseek_v4"):
    return SimpleNamespace(algorithm=algorithm)


def _should_skip(**overrides) -> bool:
    args = dict(
        batch_size=128,
        cuda_graph_padding_enabled=False,
        cuda_graph_batch_sizes=list(range(1, 129)),
        max_cuda_graph_batch_size=128,
        spec_config=_spec_config(mtp=True),
        sparse_attention_config=_sparse_config(),
        use_kv_cache_manager_v2=True,
        is_draft_model=False,
    )
    args.update(overrides)
    return should_skip_cuda_graph_for_deepseek_v4_mtp(**args)


def test_deepseek_v4_mtp_dense_exact_near_max_graph_falls_back():
    assert _should_skip(batch_size=124)
    assert _should_skip(batch_size=128)
    assert not _should_skip(batch_size=123)


def test_deepseek_v4_mtp_padded_graph_mode_remains_allowed():
    assert not _should_skip(cuda_graph_padding_enabled=True)


def test_non_mtp_cuda_graph_behavior_is_unchanged():
    assert not _should_skip(spec_config=_spec_config(mtp=False))


def test_sparse_bucket_graph_mode_remains_allowed():
    assert not _should_skip(
        cuda_graph_batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
        batch_size=128,
    )
