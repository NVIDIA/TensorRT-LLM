import pytest
import torch

from tensorrt_llm._torch.pyexecutor.green_ctx import green_ctx_split_percent


@pytest.mark.parametrize("sm_percent", [0.2, 0.4, 0.6, 0.8])
def test_green_ctx_split_percent(sm_percent):
    sm_count = torch.cuda.get_device_properties(device=0).multi_processor_count
    sm_major = torch.cuda.get_device_capability()[0]
    sm_align = 8 if sm_major >= 9 else (4 if sm_major == 8 else 2)

    _, (res_g1, res_g2) = green_ctx_split_percent(sm_percent)

    sm_g1, sm_g2 = res_g1.sm.smCount, res_g2.sm.smCount
    assert sm_g1 + sm_g2 == sm_count
    assert abs(sm_g1 - round(sm_count * sm_percent)) <= sm_align
