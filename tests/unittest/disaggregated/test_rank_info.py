from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo


def test_rank_info_construction():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
    )
    assert ri.instance_name == "gen_0"
    assert ri.tp_size == 2
    assert ri.pp_size == 1
    assert ri.layer_num_per_pp == [32]
    assert ri.sender_endpoints == ["tcp://10.0.0.1:5000"]


def test_rank_info_msgpack_roundtrip():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.instance_name == ri.instance_name
    assert restored.tp_size == ri.tp_size
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
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
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
