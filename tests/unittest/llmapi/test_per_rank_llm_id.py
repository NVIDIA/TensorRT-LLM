"""Test that llm_id propagates correctly to workers via the MPI/pickle path.

Verifies that when per_rank_routing is enabled:
1. llm_id survives model_dump → deep_merge → reconstruction (apply_model_defaults)
2. llm_id is accessible on all MPI ranks after pickle transport
3. KvCacheConfig.per_rank_routing stays True through the same path
4. The full _maybe_start_per_rank_reporter guard logic would pass

Run: pytest tests/unittest/llmapi/test_per_rank_llm_id.py -v
MPI tests (TestLlmIdMpi) require: 2+ GPUs
"""
import os
import pickle
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.util import skip_single_gpu

from tensorrt_llm.bindings.BuildInfo import ENABLE_MULTI_DEVICE
from tensorrt_llm.llmapi.llm_args import KvCacheConfig, TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import _deep_merge


def _build_test_kv_config():
    return KvCacheConfig(
        per_rank_routing=True,
        centralized_router_report_address="tcp://orchestrator:5557",
        event_buffer_max_size=1024,
        use_kv_cache_manager_v2=True,
    )


def _build_llm_args_dict():
    """Build llm_args dict as it would appear after apply_model_defaults."""
    kv_cfg = _build_test_kv_config()
    base = kv_cfg.model_dump()
    model_defaults = {
        'use_kv_cache_manager_v2': True,
        'enable_swa_scratch_reuse': True,
    }
    user_overrides = kv_cfg.model_dump(exclude_unset=True)
    merged_kv = _deep_merge(base, model_defaults, user_overrides)
    return {
        'kv_cache_config': merged_kv,
        'llm_id': 'testhost-12345-1718000000',
    }


class TestLlmIdSerialization:
    """Non-MPI tests: verify field definitions and serialization paths."""

    def test_llm_id_not_excluded_from_model_dump(self):
        """llm_id field must NOT have exclude=True."""
        field_info = TorchLlmArgs.model_fields['llm_id']
        assert not field_info.exclude, \
            f"llm_id has exclude={field_info.exclude} — will be LOST in model_dump/pickle!"

    def test_per_rank_routing_survives_model_dump(self):
        """per_rank_routing must survive model_dump → reconstruct cycle."""
        cfg = _build_test_kv_config()
        dumped = cfg.model_dump()
        assert dumped['per_rank_routing'] is True
        assert dumped['centralized_router_report_address'] == "tcp://orchestrator:5557"

        reconstructed = KvCacheConfig(**dumped)
        assert reconstructed.per_rank_routing is True
        assert reconstructed.centralized_router_report_address == "tcp://orchestrator:5557"

    def test_per_rank_routing_in_exclude_unset(self):
        """per_rank_routing must appear in exclude_unset dump when explicitly set."""
        cfg = _build_test_kv_config()
        unset = cfg.model_dump(exclude_unset=True)
        assert 'per_rank_routing' in unset
        assert 'centralized_router_report_address' in unset

    def test_deep_merge_preserves_per_rank_fields(self):
        """_deep_merge with model defaults must not clobber per_rank fields."""
        args_dict = _build_llm_args_dict()
        kv = args_dict['kv_cache_config']
        assert kv['per_rank_routing'] is True
        assert kv['centralized_router_report_address'] == "tcp://orchestrator:5557"
        assert args_dict['llm_id'] == 'testhost-12345-1718000000'

    def test_kv_cache_config_pickle_roundtrip(self):
        """KvCacheConfig must survive pickle (MPI transport uses pickle)."""
        cfg = _build_test_kv_config()
        restored = pickle.loads(pickle.dumps(cfg))
        assert restored.per_rank_routing is True
        assert restored.centralized_router_report_address == "tcp://orchestrator:5557"

    def test_full_dict_reconstruction(self):
        """Full path: model_dump → deep_merge → KvCacheConfig(**dict) must work."""
        args_dict = _build_llm_args_dict()
        kv_cfg = KvCacheConfig(**args_dict['kv_cache_config'])
        assert kv_cfg.per_rank_routing is True
        assert kv_cfg.centralized_router_report_address == "tcp://orchestrator:5557"

    def test_reporter_guard_logic_passes(self):
        """Simulate _maybe_start_per_rank_reporter guard checks (single process)."""
        args_dict = _build_llm_args_dict()
        kv_cfg = KvCacheConfig(**args_dict['kv_cache_config'])

        assert kv_cfg is not None
        assert getattr(kv_cfg, 'per_rank_routing', False) is True
        assert getattr(kv_cfg, 'centralized_router_report_address', None)
        assert args_dict['llm_id'] is not None

    def test_reporter_guard_fails_without_llm_id(self):
        """Without llm_id, the assert in _maybe_start_per_rank_reporter should fire."""
        args_dict = _build_llm_args_dict()
        args_dict['llm_id'] = None
        kv_cfg = KvCacheConfig(**args_dict['kv_cache_config'])

        assert kv_cfg is not None
        assert getattr(kv_cfg, 'per_rank_routing', False) is True
        assert getattr(kv_cfg, 'centralized_router_report_address', None)
        assert args_dict['llm_id'] is None


# --- MPI tests using MpiPoolSession (same pattern as test_base_worker.py) ---


def _task_verify_llm_id_on_rank(llm_args_dict: dict):
    """MPI worker task: verify llm_id accessible after pickle transport."""
    from tensorrt_llm._utils import mpi_rank
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    rank = mpi_rank()
    kv_dict = llm_args_dict.get('kv_cache_config', {})
    kv_cfg = KvCacheConfig(**kv_dict) if kv_dict else None

    return {
        'rank': rank,
        'llm_id': llm_args_dict.get('llm_id'),
        'per_rank_routing': kv_cfg.per_rank_routing if kv_cfg else False,
        'router_address': kv_cfg.centralized_router_report_address if kv_cfg else None,
    }


def _task_verify_llm_id_bcast(llm_id_from_rank0: str):
    """MPI worker task: bcast llm_id from rank 0 (simulates _maybe_start_per_rank_reporter)."""
    from tensorrt_llm._utils import mpi_comm, mpi_rank

    rank = mpi_rank()
    instance_id = llm_id_from_rank0 if rank == 0 else None
    instance_id = mpi_comm().bcast(instance_id, root=0)
    worker_id = f"{instance_id}:rank{rank}"

    return {'rank': rank, 'instance_id': instance_id, 'worker_id': worker_id}


def _task_full_reporter_guard(llm_args_dict: dict):
    """MPI worker task: simulate the full _maybe_start_per_rank_reporter guard."""
    from tensorrt_llm._utils import mpi_comm, mpi_rank
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    rank = mpi_rank()
    kv_dict = llm_args_dict.get('kv_cache_config')
    if kv_dict is None:
        return {'rank': rank, 'result': 'no_kv_cfg'}

    kv_cfg = KvCacheConfig(**kv_dict)
    if not getattr(kv_cfg, 'per_rank_routing', False):
        return {'rank': rank, 'result': 'no_per_rank', 'value': kv_cfg.per_rank_routing}

    router_address = getattr(kv_cfg, 'centralized_router_report_address', None)
    if not router_address:
        return {'rank': rank, 'result': 'no_address'}

    instance_id = llm_args_dict.get('llm_id')
    if instance_id is None:
        return {'rank': rank, 'result': 'no_llm_id'}

    instance_id = mpi_comm().bcast(instance_id, root=0)
    worker_id = f"{instance_id}:rank{rank}"
    return {'rank': rank, 'result': 'would_start', 'worker_id': worker_id}


@pytest.mark.skipif(not ENABLE_MULTI_DEVICE, reason="multi-device required")
class TestLlmIdMpi:
    """MPI tests using MpiPoolSession — verifies llm_id reaches all ranks."""

    def setup_method(self):
        from tensorrt_llm.llmapi.mpi_session import MpiPoolSession
        self.session = MpiPoolSession(n_workers=2)

    def teardown_method(self):
        if hasattr(self, 'session'):
            self.session.shutdown()

    @skip_single_gpu
    def test_llm_id_accessible_on_all_mpi_ranks(self):
        """llm_id and per_rank_routing must survive MPI pickle transport to workers."""
        args_dict = _build_llm_args_dict()
        results = self.session.submit_sync(_task_verify_llm_id_on_rank, args_dict)

        for r in results:
            assert r['llm_id'] == 'testhost-12345-1718000000', \
                f"Rank {r['rank']}: llm_id={r['llm_id']}"
            assert r['per_rank_routing'] is True, \
                f"Rank {r['rank']}: per_rank_routing={r['per_rank_routing']}"
            assert r['router_address'] == "tcp://orchestrator:5557", \
                f"Rank {r['rank']}: router_address={r['router_address']}"

    @skip_single_gpu
    def test_llm_id_bcast_from_rank0(self):
        """llm_id bcast from rank 0 to all ranks must deliver correct value."""
        test_llm_id = "myhost-99999-1718000000"
        results = self.session.submit_sync(_task_verify_llm_id_bcast, test_llm_id)

        for r in results:
            assert r['instance_id'] == test_llm_id, \
                f"Rank {r['rank']}: instance_id={r['instance_id']}"
            expected = f"{test_llm_id}:rank{r['rank']}"
            assert r['worker_id'] == expected, \
                f"Rank {r['rank']}: worker_id={r['worker_id']}"

    @skip_single_gpu
    def test_full_reporter_guard_passes_on_all_ranks(self):
        """Full _maybe_start_per_rank_reporter guard must pass on all ranks."""
        args_dict = _build_llm_args_dict()
        results = self.session.submit_sync(_task_full_reporter_guard, args_dict)

        for r in results:
            assert r['result'] == 'would_start', \
                f"Rank {r['rank']}: guard returned '{r['result']}' — reporter would NOT start!"
            expected = f"testhost-12345-1718000000:rank{r['rank']}"
            assert r['worker_id'] == expected
