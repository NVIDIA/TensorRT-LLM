import pytest
import torch
import xgrammar

import tensorrt_llm  # noqa


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vocab_size", [128000, 128001])
@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("logits_dtype", ["float32", "float16", "bfloat16"])
def test_logits_bitmask(batch_size: int, vocab_size: int, stride: int,
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
    bitmask = xgrammar.testing.bool_mask_to_bitmask(bool_mask)
    token_mask = None
    if stride > 1:
        token_mask = torch.arange(batch_size, dtype=torch.int32,
                                  device="cuda") % stride == 0
        token_mask = token_mask.to(torch.int32)

    # Compute reference logits
    logits_reference = logits.clone()
    logits_reference[::stride].masked_fill_(~bool_mask[::stride], -float('inf'))

    # Call logits bitmask op and evaluate
    torch.ops.trtllm.logits_bitmask(logits, bitmask, token_mask=token_mask)
    torch.testing.assert_close(logits, logits_reference)


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vocab_size", [128000, 128001])
@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("logits_dtype", ["float32", "float16", "bfloat16"])
def test_logits_bitmask_with_d2t(batch_size: int, vocab_size: int, stride: int,
                                 logits_dtype: str):
    logits_dtype = getattr(torch, logits_dtype)
    logits = torch.randn(batch_size,
                         vocab_size // 4,
                         dtype=logits_dtype,
                         device="cuda")
    bool_mask = torch.randint(0,
                              2,
                              size=(batch_size, vocab_size),
                              dtype=torch.bool,
                              device="cuda")
    bitmask = xgrammar.testing.bool_mask_to_bitmask(bool_mask)
    token_mask = None
    if stride > 1:
        token_mask = torch.arange(batch_size, dtype=torch.int32,
                                  device="cuda") % stride == 0
        token_mask = token_mask.to(torch.int32)
    d2t = torch.randint(0, 3, size=(vocab_size // 4, ),
                        device="cuda").cumsum(dim=0, dtype=torch.int32)

    # Compute reference logits
    logits_reference = logits.clone()
    draft_logits = logits_reference
    d2t_mapping = d2t + torch.arange(d2t.size(0), device=d2t.device)
    target_logits = torch.empty(draft_logits.size(0),
                                vocab_size,
                                dtype=draft_logits.dtype,
                                device=draft_logits.device)
    target_logits.index_copy_(-1, d2t_mapping, draft_logits)
    target_logits[::stride].masked_fill_(~bool_mask[::stride], -float('inf'))
    torch.index_select(target_logits, -1, d2t_mapping, out=draft_logits)

    # Call logits bitmask op and evaluate
    torch.ops.trtllm.logits_bitmask(logits,
                                    bitmask,
                                    token_mask=token_mask,
                                    d2t=d2t)
    torch.testing.assert_close(logits, logits_reference)
