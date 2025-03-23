import pytest
import torch
import xgrammar

import tensorrt_llm  # noqa


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vocab_size", [128000, 128001])
@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("logits_dtype", ["float32", "float16", "bfloat16"])
def test_logits_bitmask_op(batch_size: int, vocab_size: int, stride: int,
                           logits_dtype: str):
    logits_dtype = getattr(torch, logits_dtype)
    logits = torch.randn(batch_size,
                         vocab_size,
                         dtype=logits_dtype,
                         device="cuda")
    bool_mask = torch.randint(0,
                              2,
                              size=(batch_size, vocab_size),
                              dtype=torch.bool,
                              device="cuda")

    logits_reference = logits.clone()
    logits_reference[::stride].masked_fill_(~bool_mask[::stride], -float('inf'))

    bitmask = xgrammar.testing._bool_mask_to_bitmask(bool_mask)
    torch.ops.trtllm.logits_bitmask(
        [logits[i] for i in range(0, batch_size, stride)],
        [bitmask[i] for i in range(0, batch_size, stride)])

    torch.testing.assert_close(logits, logits_reference)
