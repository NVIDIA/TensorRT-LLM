# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CuTe DSL MLA decode split policy: without TRTLLM_MLA_BALANCED_SPLIT /
TRTLLM_MLA_MMA_LOOKAHEAD the balanced split and its MMA lookahead run for
attention-DP decode only; set, either value applies everywhere. A runner's
max_split_kv caps its uniform split; only the caller that passes it is capped.
No GPU needed."""

import pytest

try:
    import cutlass

    from tensorrt_llm._torch.custom_ops import cute_dsl_custom_ops
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner
except ImportError as e:
    pytest.skip(f"deps: {e}", allow_module_level=True)

Runner = CuteDSLNVMlaDecodeBlackwellRunner


def _runner(num_heads, helix=False, in_dtype=None, max_split_kv=0, seq_len_q=4):
    return Runner(
        in_dtype=in_dtype or cutlass.Float8E4M3FN,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        page_size=64,
        max_batch_size=64,
        emit_softmax_stats=helix,
        max_split_kv=max_split_kv,
    )


@pytest.mark.parametrize(
    "num_heads, helix, adp",
    [
        (96, False, True),  # Kimi K3 attention DP: all heads on the rank
        (128, False, True),
        (12, False, False),  # TP8 / TEP8
        (6, False, False),  # TEP16
        (96, True, False),  # Helix: softmax stats for the CP merge
    ],
)
def test_unset_follows_attention_dp(num_heads, helix, adp, monkeypatch):
    monkeypatch.setattr(Runner, "_BALANCED_SPLIT", None)
    monkeypatch.setattr(Runner, "_MMA_LOOKAHEAD", None)
    runner = _runner(num_heads, helix)
    assert runner._uses_balanced_split(16) is adp
    assert runner._fp8_mma_lookahead(True) == (2 if adp else 1)
    assert runner._fp8_mma_lookahead(False) == 1


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("num_heads, helix", [(96, False), (12, False), (96, True)])
def test_set_value_applies_everywhere(value, num_heads, helix, monkeypatch):
    monkeypatch.setattr(Runner, "_BALANCED_SPLIT", value)
    monkeypatch.setattr(Runner, "_MMA_LOOKAHEAD", value)
    runner = _runner(num_heads, helix)
    assert runner._uses_balanced_split(16) is value
    assert runner._fp8_mma_lookahead(True) == (2 if value else 1)


def test_balanced_split_is_fp8_only_and_capped(monkeypatch):
    monkeypatch.setattr(Runner, "_BALANCED_SPLIT", True)
    assert not _runner(96, in_dtype=cutlass.BFloat16)._uses_balanced_split(16)
    fp8 = _runner(96)
    cap = fp8.kernel_class.BALANCED_MAX_REQUESTS
    assert fp8._uses_balanced_split(cap)
    assert not fp8._uses_balanced_split(cap + 1)


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        (" ", None),
        ("0", False),
        ("off", False),
        ("1", True),
        ("TRUE", True),
    ],
)
def test_env_parsing(raw, expected, monkeypatch):
    if raw is None:
        monkeypatch.delenv("TRTLLM_MLA_BALANCED_SPLIT", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_MLA_BALANCED_SPLIT", raw)
    assert cute_dsl_custom_ops._mla_decode_split_env("TRTLLM_MLA_BALANCED_SPLIT") is expected


@pytest.mark.parametrize("raw", ["2", "balanced"])
def test_env_parsing_rejects_other_values(raw, monkeypatch):
    monkeypatch.setenv("TRTLLM_MLA_BALANCED_SPLIT", raw)
    with pytest.raises(ValueError, match="TRTLLM_MLA_BALANCED_SPLIT"):
        cute_dsl_custom_ops._mla_decode_split_env("TRTLLM_MLA_BALANCED_SPLIT")


def test_overrides_are_part_of_the_autotuner_id(monkeypatch):
    monkeypatch.setattr(Runner, "_BALANCED_SPLIT", None)
    monkeypatch.setattr(Runner, "_MMA_LOOKAHEAD", None)
    unset = _runner(96).unique_id()
    monkeypatch.setattr(Runner, "_BALANCED_SPLIT", False)
    assert _runner(96).unique_id() != unset


@pytest.mark.parametrize("cap, split, expected", [(0, 16, 16), (2, 16, 2), (2, 1, 1), (4, 4, 4)])
def test_split_cap(cap, split, expected):
    batch = Runner._SPLIT_CAP_MIN_BATCH
    for in_dtype in (cutlass.BFloat16, cutlass.Float16, cutlass.Float8E4M3FN):
        runner = _runner(64, in_dtype=in_dtype, max_split_kv=cap)
        assert runner._capped_split_kv(split, batch) == expected


def test_split_cap_spares_small_batches():
    floor = Runner._SPLIT_CAP_MIN_BATCH
    # Compared against power-of-two tuning buckets.
    assert floor & (floor - 1) == 0
    bf16 = _runner(64, in_dtype=cutlass.BFloat16, max_split_kv=1)
    assert bf16._capped_split_kv(32, floor // 2) == 32
    assert bf16._capped_split_kv(32, floor) == 1


def test_split_cap_bounds_the_default_tactic(monkeypatch):
    monkeypatch.setattr(Runner, "_cute_dsl_max_active_blocks", 296, raising=False)
    batch = Runner._SPLIT_CAP_MIN_BATCH
    uncapped = _runner(64, in_dtype=cutlass.BFloat16).default_tactic(batch)[2]
    assert uncapped > 2
    capped = _runner(64, in_dtype=cutlass.BFloat16, max_split_kv=2)
    assert capped.default_tactic(batch)[2] == 2
    assert capped.default_tactic(1)[2] == capped.get_default_split_kv(
        1, capped.get_folded_seq_len_q(64, 4), 296
    )


def test_split_cap_is_part_of_the_autotuner_id():
    bf16 = _runner(64, in_dtype=cutlass.BFloat16)
    assert _runner(64, in_dtype=cutlass.BFloat16, max_split_kv=0).unique_id() == bf16.unique_id()
    capped = _runner(64, in_dtype=cutlass.BFloat16, max_split_kv=2).unique_id()
    assert capped != bf16.unique_id()
    assert capped != _runner(64, in_dtype=cutlass.BFloat16, max_split_kv=1).unique_id()


def test_split_cap_rejects_negative():
    with pytest.raises(ValueError, match="max_split_kv"):
        _runner(64, in_dtype=cutlass.BFloat16, max_split_kv=-1)


def test_drafter_split_cap_env_leaves_a_target_decode_alone(monkeypatch):
    """The drafter's TLLM_DSPARK_MLA_MAX_SPLIT_KV reaches only the runner it
    builds: a BF16-KV target decode (96 heads, S=1, B=32) keeps its split."""
    monkeypatch.setattr(Runner, "_cute_dsl_max_active_blocks", 296, raising=False)
    monkeypatch.setenv("TLLM_DSPARK_MLA_MAX_SPLIT_KV", "1")
    target = _runner(96, in_dtype=cutlass.BFloat16, seq_len_q=1)
    assert target.max_split_kv == 0
    assert target.default_tactic(32)[2] == target.get_default_split_kv(
        32, target.get_folded_seq_len_q(96, 1), 296
    )
