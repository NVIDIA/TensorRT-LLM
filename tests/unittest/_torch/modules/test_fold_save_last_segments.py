"""Layout of the folded save-last prefill (pure host logic, no GPU)."""

import pytest

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import build_fold_segments


def test_no_fold_is_the_plain_layout():
    scan_cu, scan_idx, s1, s2, conv_tok, b_rows, b_cu = build_fold_segments(
        ctx_seq_lens=[5, 3], ctx_state_indices=[10, 11], folds=[None, None],
        decode_state_indices=[20, 21])
    assert scan_cu == [0, 5, 8, 9, 10]
    assert scan_idx == [10, 11, 20, 21]
    assert s1 == [] and s2 == [] and conv_tok == [] and b_rows == []
    assert b_cu == [0]


def test_folded_chunk_splits_into_snapshot_and_tail_segments():
    # request 0: chunk of 40 tokens, fold 32 tokens in (tail of 8) -> S1=3, S2=7
    # request 1: plain 6-token chunk on slot 5
    # request 2: chunk of 70, fold at 64 (tail 6) -> S1=8, S2=9
    scan_cu, scan_idx, s1, s2, conv_tok, b_rows, b_cu = build_fold_segments(
        ctx_seq_lens=[40, 6, 70],
        ctx_state_indices=[3, 5, 8],
        folds=[(32, 7), None, (64, 9)],
        decode_state_indices=[12])
    # first launch: A0 (0-32) on S1=3, B0 (32-40) on S2=7 (discarded), req1,
    # A2 (46-110) on 8, B2 (110-116) on 9, then the decode token
    assert scan_cu == [0, 32, 40, 46, 110, 116, 117]
    assert scan_idx == [3, 7, 5, 8, 9, 12]
    assert s1 == [3, 8] and s2 == [7, 9]
    assert conv_tok == [32, 110]
    assert b_rows == list(range(32, 40)) + list(range(110, 116))
    assert b_cu == [0, 8, 14]
    # every packed prefill token belongs to exactly one first-launch segment
    assert scan_cu[-1] == 40 + 6 + 70 + 1


def test_fold_offset_must_be_inside_the_chunk():
    with pytest.raises(ValueError):
        build_fold_segments([32], [1], [(32, 2)], [])
    with pytest.raises(ValueError):
        build_fold_segments([32], [1], [(0, 2)], [])
