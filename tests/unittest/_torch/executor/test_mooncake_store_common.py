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
"""Unit tests for the Mooncake store pieces every deployment shares.

Covers how a block of tokens becomes a store key, how the JSON config is read,
which deployments are refused at startup, and the pinned host slots pages pass
through where GPUDirect RDMA is unavailable.

Runs without a Mooncake installation: the store handle is replaced by an
in-process fake that records what it was handed.
"""

import importlib
import json
from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store import staging as staging_module
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.config import (
    MooncakeStoreConnectorConfig,
    StoreRole,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.keys import (
    BlockHashChain,
    KeyNamespace,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.metadata import (
    MooncakeStoreMetadata,
    PageTransfer,
    RequestTransfers,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.staging import (
    HostStagingPool,
    describe_batch_for_get,
    plan_slot_geometry,
    stage_batch_for_put,
    unstage_batch_after_get,
)
from tensorrt_llm._torch.pyexecutor.connectors.mooncake_store.validation import validate_llm_args
from tensorrt_llm._torch.pyexecutor.connectors.registry import uses_connector
from tensorrt_llm.llmapi.llm_args import KvCacheConnectorConfig

TOKENS_PER_BLOCK = 4

#: Device addresses the staging tests gather from and scatter back to. Never
#: dereferenced: the copies are recorded rather than issued.
PAGE_ADDRESSES = [0xA000, 0xB000]
PAGE_SIZES = [64, 128]
PAGE_BYTES = sum(PAGE_SIZES)


# ---- fixtures and fakes ----


class FakeStore:
    """Records buffer registrations, nothing more.

    Staging only ever asks the store to register its host buffer; the transfer
    calls belong to the connector.
    """

    def __init__(self, register_status=0, unregister_status=0):
        self.registered = []
        self.unregistered = []
        self._register_status = register_status
        self._unregister_status = unregister_status

    def register_buffer(self, address, size):
        self.registered.append((address, size))
        return self._register_status

    def unregister_buffer(self, address):
        self.unregistered.append(address)
        return self._unregister_status


@pytest.fixture
def store_config(tmp_path, monkeypatch):
    path = tmp_path / "mooncake.json"
    path.write_text(
        json.dumps(
            {
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "master_server_address": "127.0.0.1:50051",
                "protocol": "tcp",
                "device_name": "",
                "global_segment_size": "1GiB",
                "local_buffer_size": "256MiB",
                "model_key": "test-model",
            }
        )
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_ROLE", raising=False)
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_PREFIX", raising=False)
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_MODEL_KEY", raising=False)
    monkeypatch.delenv("TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST", raising=False)
    return path


def make_llm_args():
    return SimpleNamespace(
        model="/models/test-model",
        kv_cache_config=SimpleNamespace(tokens_per_block=TOKENS_PER_BLOCK),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        context_parallel_size=1,
        sparse_attention_config=None,
    )


@pytest.fixture
def staged_copies(monkeypatch):
    """Record the copies staging would issue, instead of running them."""
    copies = []
    monkeypatch.setattr(
        staging_module,
        "_memcpy_async",
        lambda dst, src, size, kind, stream: copies.append((int(dst), int(src), int(size))),
    )
    return copies


def make_pool(*, slot_bytes=PAGE_BYTES, num_slots=4, store=None):
    return HostStagingPool(
        slot_bytes=slot_bytes,
        num_slots=num_slots,
        store=store if store is not None else FakeStore(),
        label="save",
    )


# ---- keys ----


def test_hash_chain_is_deterministic_and_prefix_sensitive():
    tokens = list(range(3 * TOKENS_PER_BLOCK))
    first = list(BlockHashChain(TOKENS_PER_BLOCK).extend(tokens))
    second = list(BlockHashChain(TOKENS_PER_BLOCK).extend(tokens))
    assert first == second

    # Changing a token in block 0 must change every hash after it, which is what
    # makes a key safe to share: a hit implies the whole prefix matched.
    altered = list(tokens)
    altered[0] += 1
    changed = list(BlockHashChain(TOKENS_PER_BLOCK).extend(altered))
    assert all(a != b for a, b in zip(first, changed))


def test_hash_chain_ignores_partial_trailing_block():
    full = list(range(2 * TOKENS_PER_BLOCK))
    chain = BlockHashChain(TOKENS_PER_BLOCK)
    assert len(chain.extend(full)) == 2
    assert len(chain.extend(full + [99])) == 2


def test_hash_chain_extends_incrementally():
    tokens = list(range(4 * TOKENS_PER_BLOCK))
    incremental = BlockHashChain(TOKENS_PER_BLOCK)
    for end in range(0, len(tokens) + 1, TOKENS_PER_BLOCK):
        incremental.extend(tokens[:end])
    assert list(incremental.hashes) == list(BlockHashChain(TOKENS_PER_BLOCK).extend(tokens))


def test_hash_chain_separates_cache_salts():
    tokens = list(range(TOKENS_PER_BLOCK))
    unsalted = BlockHashChain(TOKENS_PER_BLOCK).extend(tokens)
    salted = BlockHashChain(TOKENS_PER_BLOCK, cache_salt="tenant-a").extend(tokens)
    other = BlockHashChain(TOKENS_PER_BLOCK, cache_salt="tenant-b").extend(tokens)
    assert unsalted[0] != salted[0] != other[0]
    assert salted[0] != other[0]


def test_hash_chain_rejects_shrinking_token_list():
    chain = BlockHashChain(TOKENS_PER_BLOCK)
    chain.extend(list(range(2 * TOKENS_PER_BLOCK)))
    with pytest.raises(ValueError, match="shrank"):
        chain.extend(list(range(TOKENS_PER_BLOCK)))


def test_key_namespace_separates_every_dimension():
    base = dict(
        cache_prefix="trtllm",
        model_key="m",
        rank=0,
        world_size=2,
        layer_group_id=0,
        tokens_per_block=32,
        bytes_per_page=1024,
    )
    block_hash = b"\x01" * 16
    reference = KeyNamespace(**base).key(block_hash)
    for field, value in [
        ("cache_prefix", "other"),
        ("model_key", "n"),
        ("rank", 1),
        ("world_size", 4),
        ("layer_group_id", 1),
        ("tokens_per_block", 64),
        ("bytes_per_page", 2048),
    ]:
        assert KeyNamespace(**{**base, field: value}).key(block_hash) != reference


# ---- config ----


def test_config_reads_sizes_and_staging_from_the_json(store_config):
    """Sizes arrive as unit strings, and staging is off until the JSON asks."""
    config = MooncakeStoreConnectorConfig.from_env()
    assert config.global_segment_size == 1024**3
    assert config.local_buffer_size == 256 * 1024**2
    assert config.role is StoreRole.BOTH
    assert config.resolve_model_key("/models/ignored") == "test-model"
    assert config.stage_through_host is False

    raw = json.loads(store_config.read_text())
    raw["stage_through_host"] = True
    raw["staging_buffer_bytes"] = "256MiB"
    store_config.write_text(json.dumps(raw))

    config = MooncakeStoreConnectorConfig.from_env()
    assert config.stage_through_host is True
    assert config.staging_buffer_bytes == 256 * 1024**2


def test_config_role_comes_from_environment(store_config, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "producer")
    config = MooncakeStoreConnectorConfig.from_env()
    assert config.role is StoreRole.PRODUCER
    assert config.role.saves and not config.role.loads

    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "consumer")
    config = MooncakeStoreConnectorConfig.from_env()
    assert config.role.loads and not config.role.saves

    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_ROLE", "nonsense")
    with pytest.raises(ValueError, match="TRTLLM_MOONCAKE_STORE_ROLE"):
        MooncakeStoreConnectorConfig.from_env()


def test_config_requires_the_env_var(monkeypatch):
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("TRTLLM_MOONCAKE_RUN_DIR", raising=False)
    with pytest.raises(ValueError, match="MOONCAKE_CONFIG_PATH"):
        MooncakeStoreConnectorConfig.from_env()


def test_config_falls_back_to_the_run_directory(tmp_path, monkeypatch):
    # A rank an external launcher started was already running when its leader
    # provisioned the pool, so it never inherited the exported path and reads
    # the rendered config out of the shared run directory instead.
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("TRTLLM_MOONCAKE_RUN_DIR", str(tmp_path))
    (tmp_path / "mooncake.json").write_text(
        json.dumps({"master_server_address": "10.0.0.1:50051", "global_segment_size": "8GiB"})
    )

    config = MooncakeStoreConnectorConfig.from_env()

    assert config.master_server_address == "10.0.0.1:50051"
    assert config.global_segment_size == 8 * 1024**3


def test_config_run_directory_without_a_rendered_config_still_asks(tmp_path, monkeypatch):
    # An empty run directory means no leader provisioned anything, which is a
    # missing pool rather than a default one.
    monkeypatch.delenv("MOONCAKE_CONFIG_PATH", raising=False)
    monkeypatch.setenv("TRTLLM_MOONCAKE_RUN_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="MOONCAKE_CONFIG_PATH"):
        MooncakeStoreConnectorConfig.from_env()


def test_config_env_var_wins_over_the_run_directory(tmp_path, monkeypatch):
    # An externally managed pool stays reachable, since the run directory is
    # only consulted when nothing was passed in.
    named = tmp_path / "external.json"
    named.write_text(json.dumps({"master_server_address": "external:50051"}))
    (tmp_path / "mooncake.json").write_text(
        json.dumps({"master_server_address": "provisioned:50051"})
    )
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(named))
    monkeypatch.setenv("TRTLLM_MOONCAKE_RUN_DIR", str(tmp_path))

    assert MooncakeStoreConnectorConfig.from_env().master_server_address == "external:50051"


@pytest.mark.parametrize("named", [{}, {"metadata_server": ""}], ids=["omitted", "empty"])
def test_config_metadata_server_falls_back_to_the_handshake(tmp_path, monkeypatch, named):
    # No metadata service means Mooncake's peer-to-peer handshake. An empty
    # connstring is not one of the forms setup accepts, so leaving the field
    # out of a hand-written config must not reach it.
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"master_server_address": "127.0.0.1:50051", **named}))
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    assert MooncakeStoreConnectorConfig.from_env().metadata_server == "P2PHANDSHAKE"


def test_config_requires_an_explicit_model_key(store_config, tmp_path, monkeypatch):
    # A default derived from the path would let org-a/model and org-b/model
    # agree on a namespace while disagreeing on what the pages mean.
    path = tmp_path / "no_model_key.json"
    path.write_text(json.dumps({"master_server_address": "127.0.0.1:50051"}))
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", str(path))
    config = MooncakeStoreConnectorConfig.from_env()
    with pytest.raises(ValueError, match="needs a model key"):
        config.resolve_model_key("/models/MiniMax-M3/")


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("true", True), ("on", True), ("0", False), ("off", False)]
)
def test_config_staging_env_override(store_config, monkeypatch, value, expected):
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST", value)
    assert MooncakeStoreConnectorConfig.from_env().stage_through_host is expected


def test_config_namespace_env_overrides(store_config, monkeypatch):
    # Both feed KeyNamespace, which decides whether two engines share cache.
    # Getting either wrong silently loses every hit, or lets one engine read
    # another's pages.
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_PREFIX", "tenant-a")
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_MODEL_KEY", "llama-3.1-8b@rev7")

    config = MooncakeStoreConnectorConfig.from_env()

    assert config.cache_prefix == "tenant-a"
    # Overrides the JSON's model_key rather than only supplying a missing one.
    assert config.resolve_model_key("/models/test-model") == "llama-3.1-8b@rev7"


def test_config_rejects_a_non_boolean_staging_env(store_config, monkeypatch):
    monkeypatch.setenv("TRTLLM_MOONCAKE_STORE_STAGE_THROUGH_HOST", "sometimes")
    with pytest.raises(ValueError, match="not a boolean"):
        MooncakeStoreConnectorConfig.from_env()


# ---- validation ----


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("context_parallel_size", 2, "context parallelism"),
        ("pipeline_parallel_size", 2, "pipeline parallelism"),
    ],
)
def test_validate_llm_args_rejects_unsupported_parallelism(field, value, match):
    args = make_llm_args()
    setattr(args, field, value)
    with pytest.raises(NotImplementedError, match=match):
        validate_llm_args(args)


def test_validate_llm_args_rejects_m3_index_value_cache():
    args = make_llm_args()
    args.sparse_attention_config = SimpleNamespace(sparse_disable_index_value=False)
    with pytest.raises(NotImplementedError, match="sparse_disable_index_value"):
        validate_llm_args(args)

    args.sparse_attention_config = SimpleNamespace(sparse_disable_index_value=True)
    validate_llm_args(args)


# ---- connector identification ----
#
# py_executor_creator turns partial reuse off for this connector and finds it
# through uses_connector. Failing to recognize the config would silently cost
# the reuse the store exists to provide.


@pytest.mark.parametrize(
    "config, expected",
    [
        (KvCacheConnectorConfig(connector="mooncake-store"), True),
        (
            KvCacheConnectorConfig(
                connector_module="tensorrt_llm._torch.pyexecutor.connectors.mooncake_store",
                connector_scheduler_class="MooncakeStoreConnectorScheduler",
                connector_worker_class="MooncakeStoreConnectorWorker",
            ),
            True,
        ),
        (KvCacheConnectorConfig(connector="kvbm"), False),
        (None, False),
    ],
    ids=["preset", "hand_written_module", "another_connector", "no_connector"],
)
def test_uses_connector_recognizes_the_connector_however_it_is_spelled(config, expected):
    assert uses_connector(config, "mooncake-store") is expected


def test_uses_connector_rejects_an_unknown_preset():
    config = KvCacheConnectorConfig(connector="mooncake-store")
    with pytest.raises(ValueError, match="Unknown connector preset"):
        uses_connector(config, "mooncake-stroe")


def test_the_registered_preset_resolves_to_refusing_classes():
    """py_executor_creator resolves both by name from the package."""
    config = KvCacheConnectorConfig(connector="mooncake-store")
    module = importlib.import_module(config.connector_module)

    for class_name in (config.connector_scheduler_class, config.connector_worker_class):
        with pytest.raises(NotImplementedError, match="not available in this build"):
            getattr(module, class_name)(llm_args=None)


# ---- host staging ----


@pytest.mark.parametrize(
    "page_bytes,batch,budget,expected_slots",
    [
        (1024, 64, 1 << 20, 64),  # budget is ample: the full batch stages
        (1024, 64, 8 * 1024, 8),  # budget binds before the batch does
        (1024, 8, 1 << 20, 8),  # batch binds before the budget does
        (1024, 64, 1024, 1),  # exactly one page fits
        (1024, 64, 1, 1),  # below one page, raised to one rather than refused
    ],
)
def test_plan_slot_geometry(page_bytes, batch, budget, expected_slots):
    slot_bytes, num_slots = plan_slot_geometry(page_bytes, batch, budget)
    # A slot always holds a whole page: the budget bounds the count, not the width.
    assert slot_bytes == page_bytes
    assert num_slots == expected_slots


@pytest.mark.parametrize("bad", [(0, 8, 1024), (-1, 8, 1024), (1024, 0, 1024)])
def test_plan_slot_geometry_rejects_degenerate_inputs(bad):
    with pytest.raises(ValueError):
        plan_slot_geometry(*bad)


def test_staging_pool_registers_its_whole_buffer_once():
    """One registration covering every slot, since the store addresses bytes."""
    store = FakeStore()
    pool = make_pool(slot_bytes=256, num_slots=4, store=store)

    assert pool.slot_bytes == 256
    assert pool.num_slots == 4
    assert store.registered == [(pool.slot_address(0), 256 * 4)]


def test_staging_pool_refuses_to_start_when_registration_fails():
    # Host memory that cannot be registered is the one failure staging exists to
    # rule out, so it has to be loud rather than fall back to the pools.
    with pytest.raises(RuntimeError, match="register_buffer failed"):
        make_pool(store=FakeStore(register_status=-1))


def test_staging_pool_close_unregisters_before_the_buffer_goes():
    store = FakeStore()
    pool = make_pool(slot_bytes=256, num_slots=4, store=store)
    base = pool.slot_address(0)

    pool.close()
    pool.close()

    assert store.unregistered == [base]


def test_staging_pool_close_keeps_the_buffer_when_unregistration_fails():
    store = FakeStore(unregister_status=-1)
    pool = make_pool(slot_bytes=256, num_slots=4, store=store)

    with pytest.raises(RuntimeError, match="unregister_buffer failed"):
        pool.close()

    assert pool._buffer is not None


def test_staging_pool_slots_are_contiguous_and_bounds_checked():
    pool = make_pool(slot_bytes=256, num_slots=4)
    base = pool.slot_address(0)

    assert [pool.slot_address(i) for i in range(4)] == [base + 256 * i for i in range(4)]
    with pytest.raises(IndexError):
        pool.slot_address(4)
    with pytest.raises(IndexError):
        pool.slot_address(-1)


def test_staging_pool_gather_concatenates_regions_in_region_order(staged_copies):
    """The slot layout is what the zero-copy path would have written."""
    pool = make_pool(num_slots=4)
    slot = pool.slot_address(1)

    address, total = pool.gather(1, PAGE_ADDRESSES, PAGE_SIZES, stream=0)

    assert (address, total) == (slot, PAGE_BYTES)
    assert staged_copies == [
        (slot, PAGE_ADDRESSES[0], PAGE_SIZES[0]),
        (slot + PAGE_SIZES[0], PAGE_ADDRESSES[1], PAGE_SIZES[1]),
    ]


def test_staging_pool_scatter_is_the_inverse_of_gather(staged_copies):
    pool = make_pool(num_slots=4)
    slot = pool.slot_address(1)

    pool.scatter(1, PAGE_ADDRESSES, PAGE_SIZES, stream=0)

    # Same split in the same order, with source and destination swapped.
    assert staged_copies == [
        (PAGE_ADDRESSES[0], slot, PAGE_SIZES[0]),
        (PAGE_ADDRESSES[1], slot + PAGE_SIZES[0], PAGE_SIZES[1]),
    ]


@pytest.mark.parametrize("operation", ["gather", "scatter", "reserve"])
def test_staging_pool_rejects_a_page_wider_than_a_slot(staged_copies, operation):
    # The pool is sized from the layout's largest page, so an overflow means the
    # layout changed after registration rather than a bad argument.
    pool = make_pool(slot_bytes=PAGE_BYTES - 1)

    with pytest.raises(ValueError, match="does not fit"):
        if operation == "reserve":
            pool.reserve(PAGE_BYTES)
        else:
            getattr(pool, operation)(0, PAGE_ADDRESSES, PAGE_SIZES, stream=0)


def test_stage_batch_for_put_hands_the_store_one_buffer_per_page(staged_copies):
    pool = make_pool(num_slots=4)
    addresses, sizes = stage_batch_for_put(
        pool, [PAGE_ADDRESSES, PAGE_ADDRESSES], [PAGE_SIZES, PAGE_SIZES], stream=0
    )

    # Two regions per page collapse into the one slot the store reads.
    assert addresses == [[pool.slot_address(0)], [pool.slot_address(1)]]
    assert sizes == [[PAGE_BYTES], [PAGE_BYTES]]
    assert len(staged_copies) == 4


def test_describe_batch_for_get_points_at_slots_without_copying(staged_copies):
    """Nothing to gather in this direction: the slots are the destination."""
    pool = make_pool(num_slots=4)

    addresses, sizes = describe_batch_for_get(pool, [PAGE_SIZES, PAGE_SIZES])

    assert addresses == [[pool.slot_address(0)], [pool.slot_address(1)]]
    assert sizes == [[PAGE_BYTES], [PAGE_BYTES]]
    assert staged_copies == []


@pytest.mark.parametrize("helper", ["put", "get"])
def test_batch_helpers_reject_a_batch_wider_than_the_pool(staged_copies, helper):
    pool = make_pool(num_slots=1)
    batch_sizes = [PAGE_SIZES, PAGE_SIZES]

    with pytest.raises(ValueError, match="exceeds 1 staging slots"):
        if helper == "put":
            stage_batch_for_put(pool, [PAGE_ADDRESSES, PAGE_ADDRESSES], batch_sizes, stream=0)
        else:
            describe_batch_for_get(pool, batch_sizes)


def test_unstage_batch_after_get_scatters_every_page_by_default(staged_copies):
    pool = make_pool(num_slots=4)
    pages = [[0xA000, 0xB000], [0xC000, 0xD000]]

    unstage_batch_after_get(pool, pages, [PAGE_SIZES, PAGE_SIZES], stream=0)

    assert [copy[0] for copy in staged_copies] == [0xA000, 0xB000, 0xC000, 0xD000]


def test_unstage_batch_after_get_leaves_the_pages_not_asked_for_alone(staged_copies):
    """A page whose read failed keeps its device bytes rather than slot garbage."""
    pool = make_pool(num_slots=4)
    pages = [[0xA000, 0xB000], [0xC000, 0xD000]]

    unstage_batch_after_get(pool, pages, [PAGE_SIZES, PAGE_SIZES], stream=0, only=[1])

    assert [copy[0] for copy in staged_copies] == [0xC000, 0xD000]


# ---- metadata ----


def test_metadata_is_falsy_until_there_is_work():
    """The worker skips the iteration entirely on an empty work list."""
    assert not MooncakeStoreMetadata()
    assert MooncakeStoreMetadata(loads=[RequestTransfers(request_id=1)])
    assert MooncakeStoreMetadata(saves=[RequestTransfers(request_id=1)])


def test_request_transfers_do_not_share_a_page_list():
    first = RequestTransfers(request_id=1)
    second = RequestTransfers(request_id=2)

    first.pages.append(PageTransfer(block_hash=b"\x01" * 16, layer_group_id=0, page_index=3))

    assert second.pages == []
