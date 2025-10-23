# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch


def compute_retained_tokens_count(video_size: torch.LongTensor,
                                  spatial_merge_size: int,
                                  pruning_ratio: float) -> int:
    """
    Compute the number of retained tokens for a given video.
    Method ensures that we retain all the tokens from the first frame
    regardless of the pruning rate.

    Args:
        video_size: The size of the video in the format of (T, H, W).
        spatial_merge_size: The size of the spatial merge.
        pruning_ratio: The pruning ratio.

    Returns:
        The number of retained tokens.
    """
    # Note about why map(int,..) exists here.
    # In vLLM a rounding issue was observed when input was Tensor versus when input was tuple of integers.
    # Tuple of ints input came from Preprocessing stage, while in actual forward() it was a Tensor.
    # To make sure number of output tokens stays the case - an explicit cast was added.
    T, H, W = map(int, video_size)
    min_num_tokens = (H // spatial_merge_size) * (W // spatial_merge_size)
    evs_num_tokens = int(T * min_num_tokens * (1 - pruning_ratio))
    return max(min_num_tokens, evs_num_tokens)


def compute_retention_mask(
    video_embeds: torch.Tensor,
    video_size: torch.LongTensor,
    spatial_merge_size: int,
    pruning_ratio: float,
    flatten_output: bool = True,
) -> torch.Tensor:
    """
    Computes the retention mask for input video embeddings.

    Args:
        video_embeds (`torch.Tensor`): The input video embeddings
            of shape `(T * H * W // spatial_merge_size ^ 2, hidden_size)`
            or shape `(T, H * W // spatial_merge_size ^ 2, hidden_size)`.
        video_size (`torch.LongTensor` of shape `(3)`):
            The temporal, height and width of video.
        spatial_merge_size: Size reduction for rows & cols dimensions.
        pruning_ratio: (`float`): Pruning ratio factor [0,1)
        flatten_output: (`bool`): Whether to flatten the output mask.

    Returns:
        `torch.Tensor`: The retention mask for the video embeddings of
            `(T * H * W // spatial_merge_size ^ 2)` shape.
    """
    T, H, W = video_size

    # Use reshape instead of einops to avoid graph breaks
    video_embeds = video_embeds.reshape(
        T,
        H // spatial_merge_size,
        W // spatial_merge_size,
        video_embeds.size(-1),
    )

    # Core EVS
    similarity = torch.nn.functional.cosine_similarity(video_embeds[1:, ...],
                                                       video_embeds[:-1, ...],
                                                       dim=-1)
    dissimilarity = 1 - similarity

    # Always ensure we include all tokens from the first frame
    dissimilarity = torch.cat(
        [255 * torch.ones_like(video_embeds[:1, :, :, 0]), dissimilarity],
        dim=0)

    dissimilarity_flat = dissimilarity.view(-1)
    order = torch.argsort(dissimilarity_flat,
                          dim=-1,
                          descending=True,
                          stable=True)
    retain_num_tokens = compute_retained_tokens_count(video_size,
                                                      spatial_merge_size,
                                                      pruning_ratio)
    topk_indices = order[:retain_num_tokens]

    retention_mask = torch.zeros_like(dissimilarity_flat, dtype=torch.bool)
    retention_mask[topk_indices] = True
    retention_mask = retention_mask.reshape(dissimilarity.size())

    mask = retention_mask.view(-1) if flatten_output else retention_mask
    return mask
