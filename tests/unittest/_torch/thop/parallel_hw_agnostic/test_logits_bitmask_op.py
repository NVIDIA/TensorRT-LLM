import pytest
import torch
import xgrammar

import tensorrt_llm  # noqa
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vocab_size", [128000, 128001])
@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("logits_dtype", ["float32", "float16", "bfloat16"])
def test_logits_bitmask(batch_size: int, vocab_size: int, stride: int, logits_dtype: str):
    logits_dtype = getattr(torch, logits_dtype)
    logits = torch.randn(batch_size, vocab_size, dtype=logits_dtype, device="cuda")
    bool_mask = torch.randint(0, 2, size=(batch_size, vocab_size), dtype=torch.bool, device="cuda")
    bitmask = xgrammar.testing.bool_mask_to_bitmask(bool_mask)
    token_mask = None
    if stride > 1:
        token_mask = torch.arange(batch_size, dtype=torch.int32, device="cuda") % stride == 0
        token_mask = token_mask.to(torch.int32)

    # Compute reference logits
    logits_reference = logits.clone()
    logits_reference[::stride].masked_fill_(~bool_mask[::stride], -float("inf"))

    # Call logits bitmask op and evaluate
    torch.ops.trtllm.logits_bitmask(logits, bitmask, token_mask=token_mask)
    torch.testing.assert_close(logits, logits_reference)


@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("vocab_size", [128000, 128001])
@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize("logits_dtype", ["float32", "float16", "bfloat16"])
def test_logits_bitmask_with_d2t(batch_size: int, vocab_size: int, stride: int, logits_dtype: str):
    logits_dtype = getattr(torch, logits_dtype)
    logits = torch.randn(batch_size, vocab_size // 4, dtype=logits_dtype, device="cuda")
    bool_mask = torch.randint(0, 2, size=(batch_size, vocab_size), dtype=torch.bool, device="cuda")
    bitmask = xgrammar.testing.bool_mask_to_bitmask(bool_mask)
    token_mask = None
    if stride > 1:
        token_mask = torch.arange(batch_size, dtype=torch.int32, device="cuda") % stride == 0
        token_mask = token_mask.to(torch.int32)
    d2t = torch.randint(0, 3, size=(vocab_size // 4,), device="cuda").cumsum(
        dim=0, dtype=torch.int32
    )

    # Compute reference logits
    logits_reference = logits.clone()
    draft_logits = logits_reference
    d2t_mapping = d2t + torch.arange(d2t.size(0), device=d2t.device)
    target_logits = torch.empty(
        draft_logits.size(0), vocab_size, dtype=draft_logits.dtype, device=draft_logits.device
    )
    target_logits.index_copy_(-1, d2t_mapping, draft_logits)
    target_logits[::stride].masked_fill_(~bool_mask[::stride], -float("inf"))
    torch.index_select(target_logits, -1, d2t_mapping, out=draft_logits)

    # Call logits bitmask op and evaluate
    torch.ops.trtllm.logits_bitmask(logits, bitmask, token_mask=token_mask, d2t=d2t)
    torch.testing.assert_close(logits, logits_reference)


# _apply_bitmask requires the vocab size to be divisible by the shard count,
# so 128001 (which exercises a partial last mask word) is only valid at tp 1.
@pytest.mark.parametrize("vocab_size, tp_size", [(128000, 1), (128000, 4), (128001, 1)])
def test_apply_bitmask_skips_zero_valid_token_row(vocab_size: int, tp_size: int):
    """A grammar row with no valid token must not be masked to all -inf.

    Masking the whole row makes softmax produce NaN, which trips the sampler's
    async NaN assert and takes down every rank (https://nvbugs/6625851).
    """
    batch_size, empty_row = 8, 3
    rank, local_vocab_size = tp_size - 1, vocab_size // tp_size
    # Bypass __init__ to avoid needing a tokenizer / compiled grammar: only the
    # attributes read by _apply_bitmask are required.
    guided_decoder = object.__new__(GuidedDecoder)
    guided_decoder.vocab_size_padded = vocab_size
    guided_decoder.rank = rank

    bool_mask = torch.randint(0, 2, size=(batch_size, vocab_size), dtype=torch.bool, device="cuda")
    # Keep one valid token in every shard of every other row, so that a fully
    # masked-out shard can only come from the empty row. The stride is the shard
    # width, not tp_size: at tp_size 1 the latter marks every token valid, so
    # nothing is left to reject and the reference masking becomes a no-op.
    bool_mask[:, ::local_vocab_size] = True
    bool_mask[empty_row] = False
    guided_decoder.bitmask = xgrammar.testing.bool_mask_to_bitmask(bool_mask)
    guided_decoder.token_mask = torch.ones(batch_size, dtype=torch.int32, device="cuda")

    logits = torch.randn(batch_size, local_vocab_size, dtype=torch.float32, device="cuda")
    logits_reference = logits.clone()
    shard = slice(rank * local_vocab_size, (rank + 1) * local_vocab_size)
    reject = ~bool_mask[:, shard]
    reject[empty_row] = False  # the empty row must be left untouched
    logits_reference.masked_fill_(reject, -float("inf"))

    guided_decoder._apply_bitmask(None, logits, num_bitmask_tokens=batch_size)

    torch.testing.assert_close(logits, logits_reference)
    assert not torch.isnan(torch.softmax(logits, dim=-1)).any()
