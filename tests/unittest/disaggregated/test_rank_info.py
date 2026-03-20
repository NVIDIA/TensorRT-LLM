from tensorrt_llm._torch.disaggregation.native.rank_info import InstanceInfo, RankInfo
from tensorrt_llm._torch.disaggregation.native.region.aux_ import AuxBufferMeta


def test_instance_info_construction():
    info = InstanceInfo(
        instance_name="ctx_0",
        tp_size=4,
        pp_size=2,
        dp_size=1,
        cp_size=1,
        kv_heads_per_rank=8,
        tokens_per_block=64,
        dims_per_head=128,
        element_bytes=2,
        enable_attention_dp=False,
        is_mla=False,
        layer_num_per_pp=[16, 16],
        sender_endpoints=["tcp://10.0.0.1:5000"],
    )
    assert info.instance_name == "ctx_0"
    assert info.tp_size == 4
    assert info.pp_size == 2
    assert info.layer_num_per_pp == [16, 16]
    assert info.sender_endpoints == ["tcp://10.0.0.1:5000"]


def test_instance_info_msgpack_roundtrip():
    info = InstanceInfo(
        instance_name="ctx_0",
        tp_size=4,
        pp_size=2,
        dp_size=1,
        cp_size=1,
        kv_heads_per_rank=8,
        tokens_per_block=64,
        dims_per_head=128,
        element_bytes=2,
        enable_attention_dp=False,
        is_mla=True,
        layer_num_per_pp=[16, 16],
        sender_endpoints=["tcp://10.0.0.1:5000", "tcp://10.0.0.2:5000"],
    )
    data = info.to_bytes()
    restored = InstanceInfo.from_bytes(data)
    assert restored.instance_name == info.instance_name
    assert restored.tp_size == info.tp_size
    assert restored.is_mla == info.is_mla
    assert restored.layer_num_per_pp == info.layer_num_per_pp
    assert restored.sender_endpoints == info.sender_endpoints


def test_rank_info_msgpack_roundtrip():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        device_id=0,
        kv_heads_per_rank=8,
        tokens_per_block=64,
        dims_per_head=128,
        element_bytes=2,
        enable_attention_dp=False,
        is_mla=False,
        layer_num_per_pp=[32],
        kv_ptrs=[0x1000, 0x2000],
        aux_ptrs=[0x3000],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
        aux_meta=None,
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.instance_name == ri.instance_name
    assert restored.tp_size == ri.tp_size
    assert restored.kv_ptrs == ri.kv_ptrs
    assert restored.transfer_engine_info == ri.transfer_engine_info
    assert restored.aux_meta is None


def test_rank_info_roundtrip_with_aux_meta():
    meta = AuxBufferMeta(
        ptrs=[0x4000, 0x5000],
        size=[1024, 2048],
        item_sizes=[64, 128],
        device="cpu",
    )
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        device_id=0,
        kv_heads_per_rank=8,
        tokens_per_block=64,
        dims_per_head=128,
        element_bytes=2,
        enable_attention_dp=False,
        is_mla=False,
        layer_num_per_pp=[32],
        kv_ptrs=[0x1000],
        aux_ptrs=[0x3000],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        aux_meta=meta,
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.aux_meta is not None
    assert restored.aux_meta.ptrs == [0x4000, 0x5000]
    assert restored.aux_meta.size == [1024, 2048]
    assert restored.aux_meta.item_sizes == [64, 128]
    assert restored.aux_meta.device == "cpu"
