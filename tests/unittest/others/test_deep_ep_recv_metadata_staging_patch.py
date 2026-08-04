# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

_PATCH = (
    Path(__file__).resolve().parents[3]
    / "3rdparty"
    / "patches"
    / "deep_ep_intranode_combine_fix.patch"
)


def _added_source(patch: str) -> str:
    return "\n".join(
        line[1:]
        for line in patch.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def test_deep_ep_recv_metadata_staging_patch_contract() -> None:
    patch = _PATCH.read_text()
    source = _added_source(patch)
    env_name = "TRTLLM_DEEP_EP_LP_COMBINE_RECV_METADATA_STAGING"
    extension_diff = (
        "diff --git a/csrc/kernels/extension_kernels.cu b/csrc/kernels/extension_kernels.cu"
    )
    assert patch.count(extension_diff) == 1
    receive_delta = patch.rsplit(extension_diff, maxsplit=1)[1]

    # Keep the accepted b2 send-side dynamic-scale fusion in the combined patch.
    assert "if (global_scale_per_token != nullptr)" in source
    assert "getMaxAbs helper performs a block reduction" in source
    assert "rounded reciprocal" in source
    assert "constexpr float kNvfp4GlobalScaleNumerator = 448.f * 6.f;" in source
    assert "global_scale_val = __fmul_rn(" in source
    assert "kNvfp4GlobalScaleNumerator, __frcp_rn(per_token_max_abs_val));" in source
    assert "div.rn.f32" not in source

    # The fused dispatch helper shares one 16-value scale across adjacent
    # eight-BF16 lane payloads. Supported widths must keep every pair active.
    assert "kHiddenBf16VecAccessNum % 32 == 0" in source
    assert "exchanges lane-pair maxima" in source
    assert "if (lane_id % 2 == 0)" in source

    assert env_name not in source
    assert "bool stage_recv_metadata" in source
    assert "stage_recv_metadata: bool = False" in source
    assert "async_finish, return_recv_hook, out," in source
    assert "stage_recv_metadata)" in source
    assert "getenv" not in source

    assert "int kNumMaxTopk, bool kStageRecvMetadata" in source
    for precision in ("FP8", "NVFP4"):
        for enabled in ("true", "false"):
            specialization = (
                f"low_precision_combine<LowPrecisionType::{precision}, "
                f"hidden, kNumMaxTopk, {enabled}>"
            )
            assert specialization in source

    staged = source.split("if constexpr (kStageRecvMetadata) {", maxsplit=1)[1]
    staged = staged.split("    } else {", maxsplit=1)[0]
    assert "__shared__ int staged_topk_idx[kNumMaxTopk];" in staged
    assert "__shared__ float staged_topk_weights[kNumMaxTopk];" in staged
    assert "__shared__ float staged_global_scales[kNumMaxTopk];" in staged
    assert "if (thread_id < num_topk)" in staged
    assert "if (expert_idx >= 0)" in staged
    assert staged.count("__syncthreads();") == 2
    assert "if (token_idx + num_sms < num_combined_tokens)" in staged
    assert "global_scale_val = staged_global_scales[i];" in staged
    assert "token_idx is CTA-uniform" in staged
    assert "__shfl" not in staged

    # The integrated patch only wraps the byte-for-byte original receive loop
    # in `else`; it must not delete or regenerate that loop.
    assert "+    } else {" in receive_delta
    assert "-    for (int token_idx = sm_id;" not in receive_delta
    assert "-        int reg_topk_idx[kNumMaxTopk];" not in receive_delta
    assert "-        float reg_topk_weights[kNumMaxTopk];" not in receive_delta
