from typing import List

import torch

REDRAFTER_DEFAULT_SEED = 0


def get_redrafter_specific_tensor_names() -> List[str]:
    return [
        # inputs
        'device_request_types',
        'draft_tokens',
        'draft_indices',
        'draft_probs',
        'redrafter_inverted_temperature',
        'rand_data_sample',
        'rand_data_validation',
        'position_ids_base',
        # outputs
        'next_spec_decoding_generation_lengths',
        'next_spec_decoding_position_offsets',
        'spec_decoding_mask',
        'next_draft_tokens',
        'next_draft_indices',
        'next_draft_probs',
        'next_flat_tokens',
        'num_accepted_tokens',
        'accepted_beam_index',
        'max_gen_token',
        'total_gen_token',
        'packed_position_ids',
    ]


def get_redrafter_tensor_names() -> List[str]:
    return [
        # inputs
        'spec_decoding_generation_lengths',
        'spec_decoding_position_offsets',
        'spec_decoding_packed_mask',
    ] + get_redrafter_specific_tensor_names()


def init_allocate_redrafter_tensors(session, batch_size):
    # define the buffers for ReDrafter
    session.flat_tokens = torch.zeros(
        [batch_size * (session.max_draft_tokens + 1)],
        dtype=torch.int32,
        device=session.device)
    session.next_flat_tokens = torch.zeros(
        [batch_size * (session.max_draft_tokens + 1)],
        dtype=torch.int32,
        device=session.device)
    session.position_ids_base = torch.zeros([batch_size],
                                            dtype=torch.int32,
                                            device=session.device)
    session.packed_position_ids = torch.zeros(
        [batch_size * (session.max_draft_tokens + 1)],
        dtype=torch.int32,
        device=session.device)
    session.accept_lengths = torch.ones([batch_size],
                                        dtype=torch.int32,
                                        device=session.device)
    session.draft_tokens = torch.zeros([
        batch_size, session._model_config.redrafter_num_beams,
        session._model_config.redrafter_draft_len_per_beam + 1
    ],
                                       dtype=torch.int32,
                                       device=session.device)
    session.draft_indices = torch.zeros([
        batch_size, session._model_config.redrafter_num_beams,
        session._model_config.redrafter_draft_len_per_beam + 1
    ],
                                        dtype=torch.int32,
                                        device=session.device)
    session.draft_probs = torch.zeros([
        batch_size, session._model_config.redrafter_num_beams,
        session._model_config.redrafter_draft_len_per_beam, session.vocab_size
    ],
                                      dtype=session.dtype,
                                      device=session.device)
    session.next_draft_tokens = torch.zeros([
        batch_size, session._model_config.redrafter_num_beams,
        session._model_config.redrafter_draft_len_per_beam + 1
    ],
                                            dtype=torch.int32,
                                            device=session.device)
    session.next_draft_indices = torch.zeros([
        batch_size, session._model_config.redrafter_num_beams,
        session._model_config.redrafter_draft_len_per_beam + 1
    ],
                                             dtype=torch.int32,
                                             device=session.device)
    session.next_draft_probs = torch.zeros([
        batch_size, session._model_config.redrafter_num_beams,
        session._model_config.redrafter_draft_len_per_beam, session.vocab_size
    ],
                                           dtype=session.dtype,
                                           device=session.device)
    session.next_spec_decoding_position_offsets = torch.zeros(
        [batch_size, session.max_draft_tokens + 1],
        dtype=torch.int32,
        device=session.device)
    session.next_spec_decoding_generation_lengths = torch.zeros(
        [batch_size], dtype=torch.int32, device=session.device)

    session.spec_decoding_generation_lengths = torch.zeros(
        [batch_size], dtype=torch.int32, device=session.device)
    session.spec_decoding_mask = torch.zeros([
        batch_size, session.max_draft_tokens + 1, session.max_draft_tokens + 1
    ],
                                             dtype=torch.bool,
                                             device=session.device)
    session.spec_decoding_packed_mask = torch.zeros([
        batch_size * session.max_draft_tokens + 1,
        (session.max_draft_tokens + 1 + 31) // 32
    ],
                                                    dtype=torch.int32,
                                                    device=session.device)
    session.spec_decoding_position_offsets = torch.zeros(
        [batch_size, session.max_draft_tokens + 1],
        dtype=torch.int32,
        device=session.device)
    session.accepted_beam_index = torch.zeros([batch_size],
                                              dtype=torch.int32,
                                              device=session.device)
    session.max_gen_token = torch.zeros(1,
                                        dtype=torch.int32,
                                        device=session.device)
    session.total_gen_token = torch.zeros(1,
                                          dtype=torch.int32,
                                          device=session.device)

    session.buffer['flat_tokens'] = session.flat_tokens
    session.buffer['next_flat_tokens'] = session.next_flat_tokens
    session.buffer['num_accepted_tokens'] = session.accept_lengths
    session.buffer['draft_tokens'] = session.draft_tokens
    session.buffer['draft_indices'] = session.draft_indices
    session.buffer['draft_probs'] = session.draft_probs
    session.buffer['accepted_beam_index'] = session.accepted_beam_index
    session.buffer[
        'spec_decoding_generation_lengths'] = session.spec_decoding_generation_lengths
    session.buffer['spec_decoding_mask'] = session.spec_decoding_mask
    session.buffer[
        'spec_decoding_position_offsets'] = session.spec_decoding_position_offsets
    session.buffer[
        'spec_decoding_packed_mask'] = session.spec_decoding_packed_mask
    session.buffer[
        'next_spec_decoding_generation_lengths'] = session.next_spec_decoding_generation_lengths
    session.buffer['next_draft_tokens'] = session.next_draft_tokens
    session.buffer['next_draft_indices'] = session.next_draft_indices
    session.buffer['next_draft_probs'] = session.next_draft_probs
    session.buffer[
        'next_spec_decoding_position_offsets'] = session.next_spec_decoding_position_offsets
    session.buffer['max_gen_token'] = session.max_gen_token
    session.buffer['total_gen_token'] = session.total_gen_token
    session.buffer['position_ids_base'] = session.position_ids_base
    session.buffer['packed_position_ids'] = session.packed_position_ids
    # NOTE: device_request_types is created with host_request_types
    return


def set_redrafter_ctx_tensors(session, add_tensor, add_tensor_with_bs):
    # Add all output tensors
    add_tensor(session.buffer['next_spec_decoding_generation_lengths'],
               'next_spec_decoding_generation_lengths')
    add_tensor(session.buffer['next_spec_decoding_position_offsets'],
               'next_spec_decoding_position_offsets')
    add_tensor(session.buffer['spec_decoding_mask'], 'spec_decoding_mask')
    add_tensor(session.buffer['next_flat_tokens'], 'next_flat_tokens')
    add_tensor(session.buffer['next_draft_tokens'], 'next_draft_tokens')
    add_tensor(session.buffer['next_draft_indices'], 'next_draft_indices')
    add_tensor(session.buffer['next_draft_probs'], 'next_draft_probs')
    add_tensor(session.buffer['num_accepted_tokens'], 'num_accepted_tokens')
    add_tensor(session.buffer['accepted_beam_index'], 'accepted_beam_index')
    add_tensor(session.buffer['packed_position_ids'], 'packed_position_ids')
    # add all input tensors
    add_tensor_with_bs(session.buffer['spec_decoding_generation_lengths'],
                       'spec_decoding_generation_lengths', 0)
    add_tensor_with_bs(session.buffer['spec_decoding_position_offsets'],
                       'spec_decoding_position_offsets', 0)
    add_tensor_with_bs(session.buffer['spec_decoding_packed_mask'],
                       'spec_decoding_packed_mask', 0)
    add_tensor_with_bs(session.buffer['draft_tokens'], 'draft_tokens', 0)
    add_tensor_with_bs(session.buffer['draft_indices'], 'draft_indices', 0)
    add_tensor_with_bs(session.buffer['draft_probs'], 'draft_probs', 0)
    add_tensor_with_bs(session.buffer['rand_data_validation'],
                       'rand_data_validation', 0)
    add_tensor(session.buffer['rand_data_sample'], 'rand_data_sample')
    add_tensor(session.buffer['redrafter_inverted_temperature'],
               'redrafter_inverted_temperature')
    add_tensor(session.buffer['max_gen_token'], 'max_gen_token')
    add_tensor(session.buffer['total_gen_token'], 'total_gen_token')
    add_tensor(session.buffer['position_ids_base'], 'position_ids_base')
    return


def set_redrafter_gen_tensors(session, batch_size, add_tensor,
                              add_tensor_with_shape):
    torch.cuda.nvtx.range_push("set_redrafter_gen_tensors")
    # add output tensors
    add_tensor(session.buffer['next_spec_decoding_generation_lengths'],
               'next_spec_decoding_generation_lengths')
    add_tensor(session.buffer['next_flat_tokens'], 'next_flat_tokens')
    add_tensor(session.buffer['next_draft_tokens'], 'next_draft_tokens')
    add_tensor(session.buffer['next_draft_indices'], 'next_draft_indices')
    add_tensor(session.buffer['next_draft_probs'], 'next_draft_probs')
    add_tensor(session.buffer['next_spec_decoding_position_offsets'],
               'next_spec_decoding_position_offsets')
    add_tensor(session.buffer['spec_decoding_mask'], 'spec_decoding_mask')
    add_tensor(session.buffer['num_accepted_tokens'], 'num_accepted_tokens')
    add_tensor(session.buffer['accepted_beam_index'], 'accepted_beam_index')
    add_tensor(session.buffer['packed_position_ids'], 'packed_position_ids')
    # add all input tensors
    add_tensor(session.buffer['spec_decoding_generation_lengths'],
               'spec_decoding_generation_lengths')
    # position offsets vary for ReDrafter and should already be updated at this point.
    # Just need to provide the updated shape for ReDrafter.
    max_gen_len = session.host_max_gen_token
    position_offsets = session.buffer['spec_decoding_position_offsets'].view(
        -1)[:batch_size * max_gen_len]
    add_tensor_with_shape(position_offsets.view(batch_size, max_gen_len),
                          'spec_decoding_position_offsets',
                          (batch_size, max_gen_len))
    add_tensor(session.buffer['spec_decoding_packed_mask'],
               'spec_decoding_packed_mask')
    add_tensor(session.buffer['draft_tokens'], 'draft_tokens')
    add_tensor(session.buffer['draft_indices'], 'draft_indices')
    add_tensor(session.buffer['draft_probs'], 'draft_probs')
    add_tensor(session.buffer['rand_data_validation'], 'rand_data_validation')
    add_tensor(session.buffer['rand_data_sample'], 'rand_data_sample')
    add_tensor(session.buffer['redrafter_inverted_temperature'],
               'redrafter_inverted_temperature')
    add_tensor(session.buffer['max_gen_token'], 'max_gen_token')
    add_tensor(session.buffer['total_gen_token'], 'total_gen_token')
    add_tensor(session.buffer['position_ids_base'], 'position_ids_base')
    torch.cuda.nvtx.range_pop()
    return


def redrafter_convert_spec_decoding_mask_to_packed_mask(
        session, spec_decoding_generation_lengths):
    torch.cuda.nvtx.range_push("mask_conversion")
    torch.ops.tensorrt_llm.convert_spec_decoding_mask_to_packed_mask(
        spec_decoding_generation_lengths, session.spec_decoding_mask,
        session.max_draft_tokens, session.spec_decoding_packed_mask, None)
    torch.cuda.nvtx.range_pop()
    return


def exchange_redrafter_buffers(session):
    # NOTE: shouldn't incur any copies
    def swap_buffers(name: str):
        next_name = "next_" + name
        session.buffer[name], session.buffer[next_name] = session.buffer[
            next_name], session.buffer[name]

    torch.cuda.nvtx.range_push("exchange_redrafter_buffers")
    session.host_max_gen_token = session.buffer['max_gen_token'].cpu().item()
    session.host_total_gen_token = session.buffer['total_gen_token'].cpu().item(
    )
    swap_buffers('spec_decoding_generation_lengths')
    swap_buffers('spec_decoding_position_offsets')
    swap_buffers('draft_probs')
    swap_buffers('draft_indices')
    swap_buffers('draft_tokens')
    swap_buffers("flat_tokens")
    torch.cuda.nvtx.range_pop()
    return


def process_redrafter_outputs(session, step, batch_size, last_draft_tokens,
                              new_draft_tokens):
    torch.cuda.nvtx.range_push("process_redrafter_outputs")
    best_path = session.buffer["accepted_beam_index"]
    session.accept_lengths = best_path_lengths = session.buffer[
        "num_accepted_tokens"]
    accepted_tokens = [None] * batch_size
    # print(best_path, best_path_lengths)
    for b in range(batch_size):
        torch.cuda.nvtx.range_push(f"accept_tokens_{b}")
        # use new beam0 to get latest true token
        accepted_tokens[b] = new_draft_tokens[b, 0, :1]
        if step > 0:
            verified_tokens = last_draft_tokens[b, best_path[b],
                                                1:best_path_lengths[b]]
            accepted_tokens[b] = torch.concat(
                [verified_tokens, accepted_tokens[b]])
        torch.cuda.nvtx.range_pop()
    # print("Accept", accepted_tokens)
    session.new_tokens = torch.nested.to_padded_tensor(
        torch.nested.nested_tensor(accepted_tokens, dtype=torch.int32),
        session.end_ids[0])  #FIXME  end id padding.
    torch.cuda.nvtx.range_pop()
    return best_path, best_path_lengths


def redrafter_prepare_random_tensors(session, batch_size, initialize=False):
    torch.cuda.nvtx.range_push("torch_rand")

    def get_rand_tensors():
        rds = torch.rand([1], dtype=session.dtype, device=session.device)
        rdv = torch.rand([
            1, session._model_config.redrafter_num_beams,
            session._model_config.redrafter_draft_len_per_beam
        ],
                         dtype=session.dtype,
                         device=session.device)
        return rds, rdv

    rand_data_sample = []
    rand_data_validation = []
    if initialize:  # context phase
        random_seed = session.random_seed
        if random_seed is None:
            random_seed = torch.full([batch_size],
                                     REDRAFTER_DEFAULT_SEED,
                                     dtype=torch.int64)
        session.saved_rng_states = []
    for b in range(batch_size):
        if initialize:  # context phase
            torch.manual_seed(random_seed[b].item())
        else:  # generation phase
            assert session.saved_rng_states is not None, "Couldn't find random states."
            torch.set_rng_state(session.saved_rng_states[b])
        rds, rdv = get_rand_tensors()
        session.saved_rng_states.append(torch.get_rng_state())
        rand_data_sample.append(rds)
        rand_data_validation.append(rdv)
    session.rand_data_sample = torch.concat(rand_data_sample, dim=0)
    session.rand_data_validation = torch.concat(rand_data_validation, dim=0)
    session.buffer["rand_data_sample"] = session.rand_data_sample
    session.buffer["rand_data_validation"] = session.rand_data_validation
    torch.cuda.nvtx.range_pop()
    return
