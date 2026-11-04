# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation

from ...kernels.attention import gvr_topk


@torch.library.custom_op(
    "trtllm::glm_rubin_high_throughput_gvr_topk",
    mutates_args=("indices", "workspace"),
    device_types="cuda",
)
def gvr_topk_decode(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    workspace: torch.Tensor,
    next_n: int,
) -> None:
    gvr_topk.run(scores, lengths, indices, workspace, next_n)


@gvr_topk_decode.register_fake
def _fake_gvr_topk_decode(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    indices: torch.Tensor,
    workspace: torch.Tensor,
    next_n: int,
) -> None:
    return None


class RubinTopK(TopK):
    def _forward_decode_gvr(
        self,
        scores: torch.Tensor,
        sequence_lengths: torch.Tensor,
        output_indices: torch.Tensor,
        next_n: int,
        max_seq_len: int | None,
        radix_aux_indices: torch.Tensor | None,
        radix_aux_logits: torch.Tensor | None,
        gvr_prior_indices: torch.Tensor | None = None,
        gvr_row_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if (
            self.decode_implementation == TopKImplementation.CUTE_DSL_GVR
            and self.gvr_self_sampling
            and not self.gvr_deterministic
            and self.gvr_sorting is None
            and self.top_k == 2048
            and self.compress_ratio == 1
            and gvr_topk.supports(scores, sequence_lengths, output_indices, next_n, max_seq_len)
        ):
            gvr_topk_decode(
                scores, sequence_lengths, output_indices, gvr_topk.workspace(scores), next_n
            )
            return output_indices
        return super()._forward_decode_gvr(
            scores,
            sequence_lengths,
            output_indices,
            next_n,
            max_seq_len,
            radix_aux_indices,
            radix_aux_logits,
            gvr_prior_indices,
            gvr_row_order,
        )
