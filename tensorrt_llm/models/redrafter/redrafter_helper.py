from typing import Tuple

import numpy as np

from tensorrt_llm._common import default_net
from tensorrt_llm._utils import numpy_array

# isort: off
from tensorrt_llm.functional import (
    Tensor, arange, argmax, cast, concat, constant, constant_to_tensor_, cumsum,
    div, eq, exp, expand, expand_dims, floordiv, gather, gather_nd,
    index_select, int32_array, log_softmax, lt, max, maximum, masked_select,
    minimum, nonzero, not_op, op_and, rand, relu, scatter, select, shape, slice,
    silu, softmax, squeeze, stack, sum, topk, transpose, unsqueeze, view, where)
# isort: on
from tensorrt_llm.layers import Embedding
from tensorrt_llm.module import Module

INT_DTYPE_STR = "int32"
'''
NOTE:
    Name differences from Apple's PyTorch Implementation:
        `num_candidates` is mapped to `num_beams` and
        `candidate_length` is mapped to `beam_length - 1`.
        So for each sequence, the paths/beams to verify will be [num_beams, beam_length] tokens where
        each beam is a path that includes the true token (1) and the candidate tokens (beam_length - 1).
'''


def _unpack_beams(x: Tensor, indices: Tensor, num_beams: int,
                  beam_length: int) -> Tensor:
    """
    x: [bs, S, V]
    indices: [bs, nb, bl]
    output:
    """
    assert x.rank() == 3
    d0 = shape(x, 0, INT_DTYPE_STR)
    dl = shape(x, -1, INT_DTYPE_STR)
    indices = view(indices, [-1, num_beams * beam_length, 1], False)
    res_shape = concat([d0, num_beams, beam_length, dl])
    res = view(gather_nd(x, indices), res_shape, False)  # [d0, nb, bl, dl]
    return res


def _validate_draft_tokens(draft_log_probs: Tensor,
                           draft_tokens: Tensor,
                           draft_indices: Tensor,
                           flattened_logits: Tensor,
                           num_beams: int,
                           beam_length: int,
                           greedy_search: bool,
                           rand_data: Tensor = None):
    '''
        draft_log_probs: [bs, nb, bl-1, V]
        draft_tokens: [bs, nb, bl]
        draft_indices: [bs, nb, bl]
        flattened_logits: [bs, S, V], we need to unflatten it using draft_indices.
        The unflattend_logits should be of shape [bs, nb, bl, V] by doing a gather on S.
    '''
    batch_size = shape(flattened_logits, 0, INT_DTYPE_STR)
    rand_shape = concat([batch_size, num_beams, beam_length - 1])
    if rand_data is None:
        rand_data = rand(rand_shape, low=0, high=1, dtype=draft_log_probs.dtype)

    flat_log_probs = log_softmax(flattened_logits, dim=-1)
    all_base_log_probs = _unpack_beams(flat_log_probs, draft_indices, num_beams,
                                       beam_length)  # [bs, nb, bl, V]
    if greedy_search:
        all_base_log_probs = _top_1_logits(all_base_log_probs)

    base_log_probs = index_select(all_base_log_probs,
                                  dim=2,
                                  index=constant(
                                      np.arange(beam_length - 1,
                                                dtype=np.int32)))
    last_base_log_probs = select(all_base_log_probs,
                                 dim=2,
                                 index=beam_length - 1)
    proposed_tokens = unsqueeze(slice(draft_tokens, [0, 0, 1], rand_shape), -1)

    token_base_log_probs = squeeze(
        gather(base_log_probs, dim=-1, indices=proposed_tokens), -1)
    token_draft_log_probs = squeeze(
        gather(draft_log_probs, dim=-1, indices=proposed_tokens), -1)
    diff_probs = exp(token_base_log_probs - token_draft_log_probs)
    cmp = cast(lt(rand_data, diff_probs), dtype='int32')
    ideal_sum = constant(np.arange(1, beam_length, dtype=np.int32))
    cum_sum = cumsum(cmp, dim=-1)
    equality = cast((cum_sum == ideal_sum), dtype='int32')
    num_accepted = sum(equality, dim=-1)
    max_num_accepted_tokens, accepted_beam_index = topk(
        num_accepted, k=1,
        dim=-1)  # need to use topk layer to get both value and index
    return squeeze(max_num_accepted_tokens, -1), squeeze(accepted_beam_index, -1),\
        base_log_probs, last_base_log_probs, rand_data


def _get_prefix_match_indices(beams, beam_length):
    '''
    beams: [bs, nb, bl]
    '''
    prefix_target = constant(
        np.expand_dims(np.arange(1, beam_length + 1, dtype=np.int32),
                       [0, 1, 2]))
    matches = cast(expand_dims(beams, 1) == expand_dims(beams, 2), beams.dtype)
    seq_matches = cast(cumsum(matches, dim=3) == prefix_target,
                       dtype=beams.dtype)
    prefix_match_indices = argmax(seq_matches, dim=2)
    return prefix_match_indices


def _get_draft_token_indices(prefix_match_indices, num_beams, beam_length):
    '''
    prefix_match_indices: [bs, nb, bl]
    '''
    pmi_dtype = prefix_match_indices.dtype
    segments = cast(
        constant(np.expand_dims(np.arange(0, num_beams, dtype=np.int32),
                                [0, 2])) == prefix_match_indices, pmi_dtype)
    segment_lengths = sum(segments, dim=-1)
    accum_lengths = cumsum(segment_lengths, dim=-1) - segment_lengths
    segment_index = gather(accum_lengths,
                           dim=1,
                           indices=view(prefix_match_indices,
                                        shape=[-1, num_beams * beam_length]))
    segment_index = view(segment_index, [-1, num_beams, beam_length])
    match = cast(
        expand_dims(segment_index, 3) == expand_dims(segment_index, 2),
        pmi_dtype)
    seq_index = constant(np.arange(beam_length, dtype=np.int32))
    lower_triangle = cast(
        expand_dims(seq_index, 1) > expand_dims(seq_index, 0), pmi_dtype)
    offset = sum(match * expand_dims(lower_triangle, [0, 1]), dim=-1)
    draft_token_indices = segment_index + offset
    return draft_token_indices


def _get_packed_position_ids(
    active_indices: Tensor,
    indices: Tensor,
    total_lengths: Tensor,
    position_ids_base: Tensor,
) -> Tensor:
    expand_shape = concat([shape(total_lengths, 0), shape(indices, 0)])
    expanded_indices = expand(unsqueeze(indices, 0), expand_shape)
    position_mask = expanded_indices < unsqueeze(total_lengths, 1)
    position_ids = active_indices + unsqueeze(position_ids_base, 1)
    packed_position_ids = masked_select(position_ids, position_mask)
    return packed_position_ids


def _get_draft_token_array(
    beams: Tensor,
    prefix_match_indices: Tensor,
    num_beams: int,
    beam_length: int,
    position_ids_base: Tensor = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    beams: [bs, nb, bl]
    prefix_match_indices: [bs, nb, bl]
    '''
    prefix_ideal_indices = constant(np.arange(num_beams, dtype=np.int32))
    prefix_ideal_indices = expand_dims(prefix_ideal_indices, [0, 2])
    segments = cast(eq(prefix_match_indices, prefix_ideal_indices),
                    dtype=beams.dtype)
    raw_draft_token_array = view(segments * beams + (segments - 1),
                                 [-1, num_beams * beam_length], False)
    raw_active_token_indices = transpose(
        nonzero(not_op(raw_draft_token_array == -1)), 0, 1)
    active_token_flattened = gather_nd(raw_draft_token_array,
                                       raw_active_token_indices, 0)

    total_lengths = sum(view(segments, [-1, num_beams * beam_length], False),
                        dim=1)
    slice_size = concat([shape(raw_active_token_indices, 0, INT_DTYPE_STR), 1])
    active_token_index_flattened = view(
        slice(raw_active_token_indices, starts=[0, 1], sizes=slice_size), [-1],
        False)

    max_len = max(total_lengths, dim=0)
    total_gen_len = sum(total_lengths, dim=0)
    # constant_0 = constant(int32_array(0))
    # offset = arange(constant_0, max_len, dtype='int32')
    offset = slice(constant(np.arange(num_beams * beam_length, dtype=np.int32)),
                   constant_to_tensor_(0), unsqueeze(max_len, 0))
    idx_starts = cumsum(total_lengths, 0) - total_lengths
    select_indices = unsqueeze(idx_starts, -1) + unsqueeze(offset, 0)
    max_index_allowed = shape(active_token_flattened, 0, INT_DTYPE_STR) - 1
    select_indices = minimum(view(select_indices, [-1], False),
                             max_index_allowed)
    compressed_shape = concat([shape(total_lengths, 0, INT_DTYPE_STR), max_len])
    # draft_token_array = view(
    #     gather(active_token_flattened, dim=0, indices=select_indices),
    #     compressed_shape, False)
    active_token_indices = view(
        gather(active_token_index_flattened, dim=0, indices=select_indices),
        compressed_shape, False)
    # adding position offsets here
    position_offsets = active_token_indices % beam_length
    packed_position_ids = constant_to_tensor_(0)  # dummy initialization
    if position_ids_base is not None:
        packed_position_ids = _get_packed_position_ids(position_offsets, offset,
                                                       total_lengths,
                                                       position_ids_base)
    return active_token_flattened, active_token_indices, total_lengths, max_len, total_gen_len, position_offsets, packed_position_ids


# FROM APPLE (minor changes by NV)
def _get_mask(draft_token_indices: Tensor, active_token_indices: Tensor,
              num_beams: int, beam_length: int) -> Tensor:
    """
    Return mask for candidates according to the flattened and compact index.
    Args:
        draft_token_indices: (batch_size, num_beams, beam_length)
            A Mapping of draft candidates index from a stacked representation to a
            flattened and compact representation.
        active_token_indices: (batch_size, max_len)
            A Mapping of draft candidates index from a flattened and compact representation
            to a stacked representation.
    Returns:
        compact_candidate_mask: (batch_size, max_len, max_len)
            Output a mask tensor for candidates with a flattened and compact indexing.
    """

    batch_size = shape(draft_token_indices, 0, INT_DTYPE_STR)
    max_len = shape(active_token_indices, 1, INT_DTYPE_STR)
    all_candidate_len = beam_length * num_beams

    arange_all_candidates = constant(
        np.arange(all_candidate_len, dtype=np.int32))
    active_token_beam = div(active_token_indices, beam_length)
    beam_blocks = div(arange_all_candidates, beam_length)

    lower_triangle_mask = (unsqueeze(arange_all_candidates, axis=-1) -
                           unsqueeze(arange_all_candidates, axis=0) >= 0)
    block_diagonal_mask = unsqueeze(beam_blocks, axis=-1) - unsqueeze(
        beam_blocks, axis=0) == 0
    # `candidates_mask` is the flattened candidates mask
    candidates_mask = expand(
        expand_dims(op_and(lower_triangle_mask, block_diagonal_mask), [0]),
        concat([batch_size, all_candidate_len, all_candidate_len]),
    )

    expanded_active_token_indices = expand(
        expand_dims(active_token_indices, [2]),
        concat([batch_size, max_len, all_candidate_len]))
    raw_token_mask = gather(candidates_mask,
                            dim=1,
                            indices=expanded_active_token_indices)

    src_idx = unsqueeze(active_token_beam, axis=-1) * beam_length + expand_dims(
        constant(np.arange(beam_length, dtype=np.int32)), [0, 1])
    src_mask = gather(raw_token_mask, dim=2, indices=src_idx)
    tgt_idx = gather(
        draft_token_indices,
        dim=1,
        indices=expand(expand_dims(active_token_beam, [2]),
                       concat([batch_size, max_len, beam_length])),
    )
    # `compact_candidate_mask` is the compact and flattened candidates mask
    compact_candidate_mask = expand(
        expand_dims(cast(constant_to_tensor_(0), dtype="bool"), [0, 1]),
        concat([batch_size, max_len, max_len]),
    )

    updated_compact_candidate_mask = scatter(
        compact_candidate_mask,
        dim=2,
        indices=tgt_idx,
        updates=src_mask,
    )

    return updated_compact_candidate_mask


def _beams2tree(
    beams: Tensor,
    num_beams: int,
    beam_length: int,
    position_ids_base: Tensor = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    beams: [bs, nb, bl]
    '''
    prefix_match_indices = _get_prefix_match_indices(beams, beam_length)
    draft_token_indices = _get_draft_token_indices(prefix_match_indices,
                                                   num_beams, beam_length)
    active_tokens_flattened, active_token_indices, total_lengths, max_gen_len, \
        total_gen_len, position_offsets, packed_position_ids = _get_draft_token_array(
        beams, prefix_match_indices, num_beams, beam_length, position_ids_base)
    mask = _get_mask(draft_token_indices, active_token_indices, num_beams,
                     beam_length)
    return active_tokens_flattened, draft_token_indices, mask, position_offsets, packed_position_ids, total_lengths, max_gen_len, total_gen_len


def _get_indices_for_gather_beams(batch_size: Tensor, beam_indices: Tensor,
                                  num_beams: int) -> Tensor:
    '''
    beam_indices: [bs, nb]
    Returns: [bs*nb, 2]
    '''
    constant_0 = constant(int32_array(0))
    batch_indices = arange(constant_0, batch_size * num_beams, dtype='int32')
    batch_indices = floordiv(batch_indices, num_beams)

    indices = concat([
        view(batch_indices, [-1, 1], False),
        view(beam_indices, [-1, 1], False)
    ],
                     dim=1)
    return indices


def _gather_beams(x: Tensor, indices: Tensor, batch_size: Tensor,
                  num_beams: int) -> Tensor:
    '''
    x: [bs, nb, X]
    beam_indices: [bs, nb]
    Returns: [bs, nb, X]
    '''
    target_shp = [batch_size, constant(int32_array(num_beams))]
    for i in range(2, x.ndim()):
        target_shp.append(shape(x, i, INT_DTYPE_STR))
    target_shp = concat(target_shp)
    return view(gather_nd(x, indices, batch_dims=0), target_shp, False)


def _add_decoding_dim(x: Tensor, num_beams: int) -> Tensor:
    assert x.ndim() == 1 or x.ndim() == 2
    x = unsqueeze(x, 1)
    new_shp = [shape(x, 0, INT_DTYPE_STR), num_beams] if x.ndim() == 2 else [
        shape(x, 0, INT_DTYPE_STR), num_beams,
        shape(x, 2, INT_DTYPE_STR)
    ]
    res = expand(x, concat(new_shp))
    return res


def _flatten_decoding_dim(x: Tensor) -> Tensor:
    if x.ndim() > 1:
        new_shp = [-1
                   ] + [shape(x, i, INT_DTYPE_STR) for i in range(2, x.ndim())]
        return view(x, concat(new_shp))
    return x


def _unflatten_decoding_dim(x: Tensor, num_beams: int) -> Tensor:
    '''
    Unflattens the first, flat batch*decoding dimension of a non-scalar array.
    x: [bs*num_beams, ...]
    '''
    if x.ndim() > 0:
        new_shp = [-1, num_beams
                   ] + [shape(x, i, INT_DTYPE_STR) for i in range(1, x.ndim())]
        return view(x, concat(new_shp))
    return x


def _beam_search_candidates(prompt_state: Tensor, init_token: Tensor,
                            embedding: Embedding, drafter: Module,
                            num_beams: int, beam_length: int,
                            is_rnn: bool) -> Tuple[Tensor, Tensor]:
    """
        This version of beam search matches with ReDrafter GitHub version as of 10/02/2024.
        Link: https://github.com/apple/ml-recurrent-drafter/releases/tag/v1.1
    """

    LOG_0 = -50000.0
    LOG_1 = 0.0

    def maintain_logits(logits: Tensor) -> Tensor:
        max_logits = max(logits, -1, keepdim=True)
        max_logits = expand(max_logits,
                            shape(logits, cast_to_dtype=INT_DTYPE_STR))
        return logits - max_logits

    def warp_logits(logits: Tensor,
                    top_k: int = 50,
                    mask_value: float = LOG_0) -> Tensor:
        top_k = minimum(top_k, shape(logits,
                                     dim=-1,
                                     cast_to_dtype=INT_DTYPE_STR))
        top_values, _ = topk(logits, k=top_k, dim=-1)  # [bs, nb, top_k]
        starts = concat([0, 0, top_k - 1])
        sizes = concat([shape(logits, 0), shape(logits, 1), 1])
        lt_mask = logits < slice(top_values, starts=starts, sizes=sizes)
        logits = where(lt_mask,
                       constant_to_tensor_(mask_value, dtype=logits.dtype),
                       logits)
        return logits

    def compute_logits(x: Tensor) -> Tensor:
        """
        x: [bs, nb, 2*H]
        """
        logits = drafter(x)  # [bs, nb, 2*H] => [bs, nb, V]
        logits = maintain_logits(logits)  # [bs, nb, V]
        logits = warp_logits(logits)  # [bs, nb, V]
        return logits

    assert prompt_state.ndim() == 2
    assert init_token.ndim() == 1
    assert beam_length > 1
    batch_size = shape(prompt_state, 0, INT_DTYPE_STR)
    vocab_size = embedding.num_embeddings
    dtype = prompt_state.dtype

    log_p_beam = expand(
        unsqueeze(
            constant(
                numpy_array([LOG_1] + [LOG_0] * (num_beams - 1),
                            trt_dtype=dtype)), 0),  # [1, nb]
        concat([batch_size, num_beams]))  # [bs, nb]
    context = _add_decoding_dim(prompt_state, num_beams)  # [bs, nb, H]
    if init_token.ndim() == 1:
        init_token = unsqueeze(init_token, -1)  # [bs] => [bs, 1]
    beams = _add_decoding_dim(init_token, num_beams)  # [bs, nb, 1]

    last_tokens = squeeze(beams, -1)  # [bs, nb]
    state_shape = shape(context, cast_to_dtype=INT_DTYPE_STR)  # [bs, nb, H]
    state = expand(expand_dims(constant_to_tensor_(0.0, dtype=dtype), [0, 1]),
                   state_shape)  # [bs, nb, H]
    log_p_token_in_beam = None
    candidate_length = beam_length - 1
    for _ in range(candidate_length):
        state = (
            silu(drafter.rnn_w(embedding(last_tokens)) +
                 drafter.rnn_u(state)) if is_rnn else embedding(last_tokens) +
            state)  # [bs, nb, H]

        logits_new_token = compute_logits(concat([context, state],
                                                 -1))  # [bs, nb, V]
        log_p_new_token = log_softmax(logits_new_token, -1)  # [bs, nb, V]

        log_p_beam_new_token = log_p_new_token + unsqueeze(log_p_beam,
                                                           2)  # [bs, nb, V]

        tokens_times_beams = view(log_p_beam_new_token,
                                  concat([batch_size, num_beams * vocab_size
                                          ]))  # [bs, nb*V]
        log_p_beam, topk_indices = topk(tokens_times_beams, k=num_beams,
                                        dim=-1)  # [bs, nb]
        top_beam_indices = topk_indices // vocab_size  # [bs, nb]
        # Avoid repeated division for: top_token_ids = topk_indices % vocab_size
        top_token_ids = topk_indices - (top_beam_indices * vocab_size
                                        )  # [bs, nb]

        # get the common indices to gather beams
        gather_indices = _get_indices_for_gather_beams(batch_size,
                                                       top_beam_indices,
                                                       num_beams)

        # update running beams, state, logits, and last_tokens
        prev_top_beams = _gather_beams(beams, gather_indices, batch_size,
                                       num_beams)  # [bs, nb] OR [bs, nb, 1+i]
        if prev_top_beams.ndim() == 2:
            prev_top_beams = unsqueeze(prev_top_beams, -1)  # [bs, nb, 1]
        new_tokens = unsqueeze(top_token_ids, -1)  # [bs, nb, 1]
        beams = concat([prev_top_beams, new_tokens], dim=-1)  # [bs, nb, 1+i+1]

        state = _gather_beams(state, gather_indices, batch_size,
                              num_beams)  # [bs, nb, H]

        cur_log_p_token_in_beam = unsqueeze(
            _gather_beams(log_p_new_token, gather_indices, batch_size,
                          num_beams), 2)  # [bs, nb, 1, V]
        if log_p_token_in_beam is None:  # first iteration
            log_p_token_in_beam = cur_log_p_token_in_beam
        else:
            log_p_token_in_beam = concat(
                [
                    _gather_beams(log_p_token_in_beam, gather_indices,
                                  batch_size,
                                  num_beams),  # prev_top_logits [bs, nb, i, V]
                    cur_log_p_token_in_beam
                ],
                dim=2)  # [bs, nb, i+1, V]
        last_tokens = top_token_ids  # [bs, nb]
    return beams, log_p_token_in_beam


def _top_1_logits(logits: Tensor, NINF=-50000.0) -> Tensor:
    '''
    logits: [bs, S, V]
    '''
    NEG_INF = constant_to_tensor_(NINF, logits.dtype)
    # TODO: WAR for bug in max reduction: https://nvbugs/4714485
    # max_values = max(logits, dim=-1, keepdim=True)  # [bs, S, 1]
    max_values, _ = topk(logits, k=1, dim=-1)  # [bs, S, 1]
    cmp = not_op(logits == max_values)
    res = cast(cmp, dtype=logits.dtype) * NEG_INF
    return res


def _ctx_logits2probs(logits: Tensor, greedy_search: bool) -> Tensor:
    """
    Inputs:
        logits: [bs_ctx, V]
    Returns:
        probs: [bs_ctx, V]
    """
    if greedy_search:
        logits = _top_1_logits(logits)
    probs = softmax(logits, dim=-1)
    return probs


# Jointly developed with Apple
def _batch_index_select(x: Tensor, batch_index: Tensor) -> Tensor:
    """select the tensor by index inside each batch

    Args:
        x (Tensor): [batch, ..]
        batch_index (Tensor): (batch_size)

    Returns:
        Tensor: [batch, ..] Tensors selected by the indices
    """
    expanded_shape = concat(
        [shape(x, 0, INT_DTYPE_STR), 1] +
        [shape(x, i, INT_DTYPE_STR) for i in range(2, x.rank())])
    batch_index = expand(
        expand_dims(batch_index, range(1,
                                       x.rank() - batch_index.rank() + 1)),
        expanded_shape)
    gathered_x = gather(x, dim=1, indices=batch_index)
    return squeeze(gathered_x, dim=1)


# Jointly developed with Apple
def _prepare_drafter_input(
    draft_log_probs: Tensor,
    base_log_probs: Tensor,
    last_base_log_probs: Tensor,
    accepted_beam_index: Tensor,
    num_accepted_tokens: Tensor,
) -> Tensor:
    """
    Args:
        num_accepted_tokens: (batch_size)
            Highest count of accepted tokens.
        accepted_beam_index: (batch_size)
            Beam index with highest count of accepted tokens.
        draft_log_probs: (batch_size, num_candidates, candidate_length, vocab_size)
            Draft head log probs for draft_tokens.
        base_log_probs: (batch_size, num_candidates, candidate_length, vocab_size)
            LM log probs for draft_tokens.
        last_base_log_probs: (batch_size, num_candidates, vocab_size)
            Last token log probs for all candidates to predict the next token beyond each candidate.
    Returns:
        probs: (batch_size, vocab_size):
            Predict next token probability.

    """
    # Select according to the chosen beam index.
    candidate_length = shape(draft_log_probs, 2, INT_DTYPE_STR)
    selected_draft_log_probs = _batch_index_select(draft_log_probs,
                                                   accepted_beam_index)
    selected_base_log_probs = _batch_index_select(base_log_probs,
                                                  accepted_beam_index)
    selected_last_base_log_probs = _batch_index_select(last_base_log_probs,
                                                       accepted_beam_index)

    # Check if the entire beam is accepted or not.
    entire_beam_accept = unsqueeze(num_accepted_tokens == candidate_length,
                                   axis=-1)

    # If the entire beam is accepted, we use maybe_last_probs to sample next token.
    maybe_last_probs = exp(selected_last_base_log_probs)

    # Note the shape of selected_draft_log_probs and selected_base_log_probs is the same
    # as [batch_size, candidate_length, vocab_size].
    # Thus, we clamp resample_index to be up to candidate_length - 1.
    # Since when num_accepted_tokens == candidate_length, we use maybe_last_probs above.
    resample_index = num_accepted_tokens - cast(
        eq(num_accepted_tokens, candidate_length), dtype='int32')
    sample_draft_log_probs = _batch_index_select(selected_draft_log_probs,
                                                 resample_index)
    sample_base_log_probs = _batch_index_select(selected_base_log_probs,
                                                resample_index)
    # Rejection sampling probs.
    probs = relu(exp(sample_base_log_probs) - exp(sample_draft_log_probs))
    probs = where(entire_beam_accept, maybe_last_probs, probs)

    return probs


def _process_gen_logits(logits: Tensor,
                        hidden: Tensor,
                        draft_probs: Tensor,
                        draft_tokens: Tensor,
                        draft_indices: Tensor,
                        num_beams: int,
                        beam_length: int,
                        greedy_search: bool,
                        rand_data: Tensor = None) -> Tensor:
    num_accepted_tokens, accepted_beam_index,\
        base_log_probs, last_base_log_probs, _ = _validate_draft_tokens(
        draft_probs, draft_tokens, draft_indices, logits, num_beams, beam_length,
        greedy_search, rand_data)

    # need to retrieve flattened index from accepted_beam_index and num_accepted_tokens
    indices = stack([accepted_beam_index, num_accepted_tokens], 1)
    flat_indices = unsqueeze(gather_nd(draft_indices, indices, batch_dims=1),
                             -1)
    filtered_probs = _prepare_drafter_input(draft_probs, base_log_probs,
                                            last_base_log_probs,
                                            accepted_beam_index,
                                            num_accepted_tokens)
    filtered_hidden = gather_nd(hidden, flat_indices, batch_dims=1)
    return filtered_probs, filtered_hidden, num_accepted_tokens, accepted_beam_index


def _get_gen_token_indices_for_unpack(
        num_gen_tokens: Tensor, num_beams: int, beam_length: int,
        max_index_allowed: Tensor) -> Tuple[Tensor, Tensor]:
    upper_bound = num_beams * beam_length - num_beams + 1
    max_gen_tokens = max(num_gen_tokens, dim=0)
    max_gen_tokens = minimum(max_gen_tokens, upper_bound)
    max_gen_tokens = maximum(max_gen_tokens, 0)
    cum_gen_tokens = cumsum(num_gen_tokens, 0)
    gen_token_starts = cum_gen_tokens - num_gen_tokens
    gen_unpack_indxs = arange(constant_to_tensor_(0, to_array=False),
                              max_gen_tokens,
                              dtype='int32')
    gen_unpack_indxs = unsqueeze(gen_unpack_indxs, 0) + unsqueeze(
        gen_token_starts, 1)
    gen_unpack_indxs = minimum(gen_unpack_indxs, max_index_allowed)
    return gen_unpack_indxs, max_gen_tokens


def _unpack_gen_data(x: Tensor, num_gen_tokens: Tensor,
                     gen_unpack_indxs: Tensor,
                     max_gen_tokens: Tensor) -> Tensor:
    """
    x: [sum(num_gen_tokens), V/H]
    num_gen_tokens: [gen_bs]
    gen_unpack_indxs: [bs, max(num_gen_tokens)]
    Returns:
        [gen_bs, max_gen_tokens, V/H] where max_gen_tokens = max(num_gen_tokens)
    """
    unpacked_x = index_select(x, dim=0, index=view(gen_unpack_indxs, [-1]))
    out_shape = concat([
        shape(num_gen_tokens, 0, INT_DTYPE_STR), max_gen_tokens,
        shape(x, -1, INT_DTYPE_STR)
    ])
    return unpacked_x.view(out_shape, zero_is_placeholder=False)


def _process_logits_and_hidden_states(
        model: Module, logits: Tensor, hidden_states: Tensor,
        kwargs: dict) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Process the logits and hidden_states correctly.
    For logits:
        Can be all context, all gen or mixed.
            For all context-phase:
                the shape is [bs, V], just process to probs
            For all gen-phase:
                the shape is [sum(num_gen_tokens), V]
                gather using num_gen_tokens => [gen_bs, max_gen_tokens, V]
                then typical processing as above
            For mixed case:
                split the logits, do both ctx and gen phase processing
    For hidden_states:
        context phase: similar processing
        gen-phase: filter based on accepted beams and their lengths.
    """
    if model is not None:
        num_beams = model.num_beams
        beam_length = model.beam_length
        greedy_search = model.greedy_search
    else:
        num_beams = kwargs['num_beams']
        beam_length = kwargs['beam_length']
        greedy_search = kwargs.get('greedy_search', False)
    device_request_types = kwargs['device_request_types']
    inverted_temperature = kwargs['redrafter_inverted_temperature']  # [bs]
    num_gen_tokens = kwargs[
        'spec_decoding_params'].spec_decoding_generation_lengths
    assert default_net(
    ).plugin_config.remove_input_padding, "ReDrafter is only supported without input padding."
    """
        Split the flattened data: context and generation
        Process them separately.
        NOTE: Involves processing 0-shaped tensors (if all context or all generation)
    """
    # process context
    const_0 = constant_to_tensor_(0, to_array=False)
    bs = shape(device_request_types, 0, INT_DTYPE_STR)
    num_gen = sum(device_request_types, -1)
    num_gen = maximum(constant_to_tensor_(0, to_array=False), num_gen)
    num_gen = minimum(bs, num_gen)
    bs_ctx = bs - num_gen
    ctx_idxs = arange(const_0, bs_ctx, dtype='int32')
    assert bs_ctx.rank() == 0
    ctx_logits = index_select(logits, dim=0, index=ctx_idxs)
    if not greedy_search:
        ctx_temperature = index_select(inverted_temperature,
                                       dim=0,
                                       index=ctx_idxs)
        ctx_temperature = unsqueeze(ctx_temperature, 1)
        ctx_logits = ctx_logits * ctx_temperature
    ctx_probs = _ctx_logits2probs(ctx_logits, greedy_search)
    ctx_hidden_states = index_select(hidden_states, dim=0, index=ctx_idxs)
    # we accept zero draft tokens for ctx-phase
    ctx_num_accepted = expand(constant_to_tensor_(0), unsqueeze(bs_ctx, 0))
    ctx_accepted_beam_index = expand(constant_to_tensor_(0),
                                     unsqueeze(bs_ctx, 0))

    # process generation
    # get the logits[bs_ctx:, :] and hidden_states[bs_ctx:, :]
    gen_token_idxs = arange(bs_ctx,
                            shape(logits, 0, INT_DTYPE_STR),
                            dtype='int32')
    gen_logits = index_select(logits, dim=0, index=gen_token_idxs)
    gen_hidden = index_select(hidden_states, dim=0, index=gen_token_idxs)
    max_index_allowed = shape(gen_logits, 0, INT_DTYPE_STR) - 1
    gen_unpack_idxs, max_gen_tokens = _get_gen_token_indices_for_unpack(
        num_gen_tokens, num_beams, beam_length, max_index_allowed)
    gen_logits = _unpack_gen_data(gen_logits, num_gen_tokens, gen_unpack_idxs,
                                  max_gen_tokens)
    if not greedy_search:
        gen_temperature = index_select(inverted_temperature,
                                       dim=0,
                                       index=gen_token_idxs)
        gen_temperature = expand_dims(gen_temperature, dim=[1, 2])
        expanded_gen_temperature = expand(gen_temperature, shape(gen_logits))
        gen_logits = gen_logits * expanded_gen_temperature
    gen_hidden = _unpack_gen_data(gen_hidden, num_gen_tokens, gen_unpack_idxs,
                                  max_gen_tokens)

    # verify the input draft tokens (from last step) using the gen_logits
    gen_probs, gen_hidden_states, gen_num_accepted, gen_accepted_beam_index\
        = _process_gen_logits(
        gen_logits, gen_hidden, kwargs['draft_probs'],
        kwargs['draft_tokens'], kwargs['draft_indices'],
        num_beams, beam_length, greedy_search,
        kwargs.get('rand_data_validation', None)
    )

    # combine ctx and gen phase outputs
    probs = concat([ctx_probs, gen_probs], dim=0)
    drafter_input = concat([ctx_hidden_states, gen_hidden_states], dim=0)
    num_accepted_tokens = concat([ctx_num_accepted, gen_num_accepted], dim=0)
    accepted_beam_index = concat(
        [ctx_accepted_beam_index, gen_accepted_beam_index], dim=0)

    # NOTE: This is needed with shape inference of data-dependent tensors
    bs = shape(device_request_types, 0, INT_DTYPE_STR)
    const_0 = constant_to_tensor_(0, to_array=False)
    bidxs = arange(const_0, bs, dtype='int32')
    probs = index_select(probs, dim=0, index=bidxs)
    drafter_input = index_select(drafter_input, dim=0, index=bidxs)
    num_accepted_tokens = index_select(num_accepted_tokens, dim=0, index=bidxs)
    accepted_beam_index = index_select(accepted_beam_index, dim=0, index=bidxs)
    return probs, drafter_input, num_accepted_tokens, accepted_beam_index
