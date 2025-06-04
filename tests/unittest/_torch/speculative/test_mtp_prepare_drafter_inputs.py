import unittest

import torch
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.speculative.mtp import (MTPConfig,
                                                 MTPHiddenStatesManager,
                                                 MTPSpecMetadata, MTPWorker)


def unittest_name_func(testcase_func, param_num, param):
    name = param.args[0]
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name(name),
    )


class TestMTPPrepareDrafterInputs(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def load_prepare_drafter_inputs_test_cases():

        def gen_data(batch_size, num_nextn_predict_layers, hidden_size):
            mtp_past_hidden_states_ptrs = []
            mtp_past_tokens_ptrs = []
            mtp_hidden_states_tensor_pool = torch.ones(
                (batch_size, num_nextn_predict_layers, hidden_size),
                device='cuda',
                dtype=torch.float32)
            mtp_tokens_tensor_pool = torch.ones(
                (batch_size, num_nextn_predict_layers),
                device='cuda',
                dtype=torch.int)

            for bix in range(batch_size):
                mtp_hidden_states_tensor_pool[
                    bix] = mtp_hidden_states_tensor_pool[
                        bix] * bix  # be different
                mtp_past_hidden_states_ptrs.append(
                    mtp_hidden_states_tensor_pool[bix].data_ptr())

                mtp_tokens_tensor_pool[
                    bix] = mtp_tokens_tensor_pool[bix] * bix  # be different
                mtp_past_tokens_ptrs.append(
                    mtp_tokens_tensor_pool[bix].data_ptr())
            return mtp_past_hidden_states_ptrs, mtp_past_tokens_ptrs, mtp_hidden_states_tensor_pool, mtp_tokens_tensor_pool

        test_cases = []

        ################# CASE 0 ##########################
        # MTP0, BS=1, 1 context request
        batch_size = 1
        num_nextn_predict_layers = 1
        num_contexts = 1
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two da ta,
        # One for is_thop=False, one for is_thop=True
        mtp_past_hidden_states_ptrs_v1, mtp_past_tokens_ptrs_v1, mtp_hidden_states_tensor_pool_v1, mtp_tokens_tensor_pool_v1 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)
        mtp_past_hidden_states_ptrs_v2, mtp_past_tokens_ptrs_v2, mtp_hidden_states_tensor_pool_v2, mtp_tokens_tensor_pool_v2 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor([5, 6, 7, 8, 9],
                                 dtype=torch.int,
                                 device="cuda")
        position_ids = torch.tensor([0, 1, 2, 3, 4],
                                    dtype=torch.int,
                                    device="cuda")

        seq_lens = torch.tensor([5], dtype=torch.int,
                                device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        accepted_tokens = torch.tensor([[0, -1]],
                                       dtype=torch.int,
                                       device="cuda")
        num_accepted_tokens = torch.tensor([1], dtype=torch.int, device="cuda")

        ref_input_ids = torch.tensor([6, 7, 8, 9, 0],
                                     dtype=torch.int,
                                     device="cuda")
        ref_previous_hidden_states = previous_layer_hidden_states
        attn_metadata = None

        test_cases += [[
            'case0_is_thop_false',
            num_nextn_predict_layers,
            input_ids,
            position_ids,
            seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1, device="cuda"),
            mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1,
            previous_layer_hidden_states,
            request_ids,
            num_contexts,
            accepted_tokens,
            num_accepted_tokens,
            attn_metadata,
            ref_input_ids,
            ref_previous_hidden_states,
            False,
        ]]

        test_cases += [[
            'case0_is_thop_true', num_nextn_predict_layers, input_ids,
            position_ids, seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v2,
                         device="cuda"), mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2, previous_layer_hidden_states,
            request_ids, num_contexts, accepted_tokens, num_accepted_tokens,
            attn_metadata, ref_input_ids, ref_previous_hidden_states, True
        ]]

        ################# CASE 1 ##########################
        # MTP0, BS=3, 3 context request
        batch_size = 3
        num_contexts = 3
        num_nextn_predict_layers = 1
        hidden_size = 16
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        mtp_past_hidden_states_ptrs_v1, mtp_past_tokens_ptrs_v1, mtp_hidden_states_tensor_pool_v1, mtp_tokens_tensor_pool_v1 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)
        mtp_past_hidden_states_ptrs_v2, mtp_past_tokens_ptrs_v2, mtp_hidden_states_tensor_pool_v2, mtp_tokens_tensor_pool_v2 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor(
            [
                5,
                6,
                7,
                8,
                9,  # request0
                10,
                11,
                12,
                13,  # request1
                20,
                21,
                22,
                23,
                24,
                25  # request2
            ],
            dtype=torch.int,
            device="cuda")
        position_ids = torch.tensor(
            [[
                0,
                1,
                2,
                3,
                4,  # request0
                0,
                1,
                2,
                3,  # request1
                0,
                1,
                2,
                3,
                4,
                5  # request2
            ]],
            dtype=torch.int,
            device="cuda")

        seq_lens = torch.tensor([5, 4, 6], dtype=torch.int,
                                device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.randn(
            (len(input_ids), hidden_size), dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        accepted_tokens = torch.tensor([[0, -1], [1, -1], [2, -1]],
                                       dtype=torch.int,
                                       device="cuda")
        num_accepted_tokens = torch.tensor([1, 1, 1],
                                           dtype=torch.int,
                                           device="cuda")

        ref_input_ids = torch.tensor(
            [6, 7, 8, 9, 0, 11, 12, 13, 1, 21, 22, 23, 24, 25, 2],
            dtype=torch.int,
            device="cuda")
        ref_previous_hidden_states = previous_layer_hidden_states

        attn_metadata = None

        test_cases += [[
            'case1_is_thop_false', num_nextn_predict_layers, input_ids,
            position_ids, seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, previous_layer_hidden_states,
            request_ids, num_contexts, accepted_tokens, num_accepted_tokens,
            attn_metadata, ref_input_ids, ref_previous_hidden_states, False
        ]]

        test_cases += [[
            'case1_is_thop_true', num_nextn_predict_layers, input_ids,
            position_ids, seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v2,
                         device="cuda"), mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2, previous_layer_hidden_states,
            request_ids, num_contexts, accepted_tokens, num_accepted_tokens,
            attn_metadata, ref_input_ids, ref_previous_hidden_states, True
        ]]

        ################## CASE 2 ##########################
        # BS=1, 1 generation request, num_nextn_predict_layers = 1
        batch_size = 1
        num_contexts = 0
        num_nextn_predict_layers = 1
        hidden_size = 12
        request_ids = range(batch_size)

        mtp_past_hidden_states_ptrs_v1 = []
        mtp_past_tokens_ptrs_v1 = []
        mtp_hidden_states_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers, hidden_size),
            device='cuda',
            dtype=torch.float32)
        mtp_tokens_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers),
            device='cuda',
            dtype=torch.int)

        mtp_hidden_states_tensor_pool_v1[0] = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
            device='cuda',
            dtype=torch.float32)
        mtp_past_hidden_states_ptrs_v1.append(
            mtp_hidden_states_tensor_pool_v1[0].data_ptr())
        mtp_tokens_tensor_pool_v1[0] = torch.tensor([42],
                                                    device='cuda',
                                                    dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[0].data_ptr())

        input_ids = torch.tensor([6, 42], dtype=torch.int, device="cuda")

        position_ids = torch.tensor([10, 11], dtype=torch.int, device="cuda")

        # already '-1' in 'update_mtp_hidden_states'
        seq_lens = torch.tensor([2], dtype=torch.int,
                                device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.randn(
            (len(input_ids), hidden_size), dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        accepted_tokens = torch.tensor([[43, -1]],
                                       dtype=torch.int,
                                       device="cuda")

        num_accepted_tokens = torch.tensor([1], dtype=torch.int, device="cuda")

        ref_input_ids = torch.tensor([43], dtype=torch.int, device="cuda")

        ref_previous_hidden_states = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
            device='cuda',
            dtype=torch.float32)

        attn_metadata = None

        test_cases += [[
            'case2_is_thop_false', num_nextn_predict_layers, input_ids,
            position_ids, seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, previous_layer_hidden_states,
            request_ids, num_contexts, accepted_tokens, num_accepted_tokens,
            attn_metadata, ref_input_ids, ref_previous_hidden_states, False
        ]]

        test_cases += [[
            'case2_is_thop_true', num_nextn_predict_layers, input_ids,
            position_ids, seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, previous_layer_hidden_states,
            request_ids, num_contexts, accepted_tokens, num_accepted_tokens,
            attn_metadata, ref_input_ids, ref_previous_hidden_states, True
        ]]

        ################## CASE 3 ##########################
        # BS=3, 3 generation request, num_nextn_predict_layers = 3
        batch_size = 3
        num_contexts = 0
        num_nextn_predict_layers = 3
        hidden_size = 12
        request_ids = range(batch_size)

        mtp_past_hidden_states_ptrs_v1 = []
        mtp_past_tokens_ptrs_v1 = []
        mtp_hidden_states_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers, hidden_size),
            device='cuda',
            dtype=torch.float32)
        mtp_tokens_tensor_pool_v1 = torch.ones(
            (batch_size, num_nextn_predict_layers),
            device='cuda',
            dtype=torch.int)

        mtp_hidden_states_tensor_pool_v1[0] = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
        ],
                                                           device='cuda',
                                                           dtype=torch.float32)
        mtp_past_hidden_states_ptrs_v1.append(
            mtp_hidden_states_tensor_pool_v1[0].data_ptr())

        mtp_hidden_states_tensor_pool_v1[1] = torch.tensor([
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
        ],
                                                           device='cuda',
                                                           dtype=torch.float32)
        mtp_past_hidden_states_ptrs_v1.append(
            mtp_hidden_states_tensor_pool_v1[1].data_ptr())

        mtp_hidden_states_tensor_pool_v1[2] = torch.tensor([
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
        ],
                                                           device='cuda',
                                                           dtype=torch.float32)
        mtp_past_hidden_states_ptrs_v1.append(
            mtp_hidden_states_tensor_pool_v1[2].data_ptr())

        mtp_tokens_tensor_pool_v1[0] = torch.tensor([19, 20, 21],
                                                    device='cuda',
                                                    dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[0].data_ptr())

        mtp_tokens_tensor_pool_v1[1] = torch.tensor([29, 30, 31],
                                                    device='cuda',
                                                    dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[1].data_ptr())

        mtp_tokens_tensor_pool_v1[2] = torch.tensor([39, 40, 41],
                                                    device='cuda',
                                                    dtype=torch.int)
        mtp_past_tokens_ptrs_v1.append(mtp_tokens_tensor_pool_v1[2].data_ptr())

        input_ids = torch.tensor(
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            dtype=torch.int,
            device="cuda")  # useless
        position_ids = torch.tensor(
            [10, 11, 12, 13, 21, 22, 23, 24, 32, 33, 34, 35],
            dtype=torch.int,
            device="cuda")

        seq_lens = torch.tensor([4, 4, 4], dtype=torch.int,
                                device="cuda")  # [batch_size]

        previous_layer_hidden_states = torch.randn(
            (12, hidden_size), device='cuda', dtype=torch.float32)  # useless

        accepted_tokens = torch.tensor(
            [[22, -1, -1, -1], [31, 32, -1, -1], [0, 40, 41, 42]],
            dtype=torch.int,
            device="cuda")

        num_accepted_tokens = torch.tensor([1, 2, 4],
                                           dtype=torch.int,
                                           device="cuda")

        ref_input_ids = torch.tensor([20, 21, 22, 30, 31, 32, 40, 41, 42],
                                     dtype=torch.int,
                                     device="cuda")

        ref_previous_hidden_states = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
        ],
                                                  device='cuda',
                                                  dtype=torch.float32)
        attn_metadata = None

        test_cases += [[
            'case3_is_thop_false', num_nextn_predict_layers, input_ids,
            position_ids, seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, previous_layer_hidden_states,
            request_ids, num_contexts, accepted_tokens, num_accepted_tokens,
            attn_metadata, ref_input_ids, ref_previous_hidden_states, False
        ]]

        test_cases += [[
            'case3_is_thop_true', num_nextn_predict_layers, input_ids,
            position_ids, seq_lens,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, previous_layer_hidden_states,
            request_ids, num_contexts, accepted_tokens, num_accepted_tokens,
            attn_metadata, ref_input_ids, ref_previous_hidden_states, True
        ]]

        return test_cases

    @parameterized.expand(load_prepare_drafter_inputs_test_cases,
                          name_func=unittest_name_func)
    def test_prepare_drafter_inputs(
            self, test_case_name, num_nextn_predict_layers, input_ids,
            position_ids, seq_lens, mtp_past_hidden_states_ptrs,
            mtp_past_tokens_ptrs, mtp_hidden_states_tensor_pool,
            mtp_tokens_tensor_pool, previous_layer_hidden_states, request_ids,
            num_contexts, accepted_tokens, num_accepted_tokens, attn_metadata,
            ref_input_ids, ref_previous_hidden_states, is_thop):

        batch_size = len(request_ids)
        if previous_layer_hidden_states is not None:
            hidden_size = previous_layer_hidden_states.shape[1]
        else:
            hidden_size = 10
        spec_config = MTPConfig(
            num_nextn_predict_layers=num_nextn_predict_layers)

        if attn_metadata is None:
            attn_metadata = TrtllmAttentionMetadata(max_num_requests=batch_size,
                                                    max_num_tokens=1024,
                                                    kv_cache_manager=None)
            attn_metadata.seq_lens = seq_lens.to('cpu')
            attn_metadata.num_contexts = num_contexts
            # dummy kv cache param
            attn_metadata.kv_cache_params = KVCacheParams(
                use_cache=True, num_cached_tokens_per_seq=[10] * batch_size)

        spec_manager = MTPHiddenStatesManager(config=spec_config,
                                              dtype=torch.float32,
                                              hidden_size=hidden_size,
                                              max_num_requests=batch_size)
        for i in range(batch_size):
            # for the generation requests, we also need to manually add slot
            # because these generation requests are also first use
            spec_manager.slot_manager.add_slot(request_ids[i])

        spec_metadata = MTPSpecMetadata(
            max_num_requests=32,
            spec_dec_mode=spec_config.spec_dec_mode,
            max_draft_tokens=num_nextn_predict_layers,
            mtp_num_modules=num_nextn_predict_layers,
            mtp_hidden_states_manager=spec_manager)
        spec_metadata.request_ids = request_ids
        spec_metadata.mtp_hidden_states_ptrs = mtp_past_hidden_states_ptrs
        spec_metadata.mtp_past_tokens_ptrs = mtp_past_tokens_ptrs

        spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool = mtp_hidden_states_tensor_pool
        spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool = mtp_tokens_tensor_pool
        spec_metadata.prepare()

        mtpworker = MTPWorker(spec_config)
        mtpworker.is_thop = is_thop
        draft_inputs = mtpworker.prepare_drafter_inputs(
            input_ids=input_ids,
            position_ids=position_ids,
            hidden_states=previous_layer_hidden_states,
            accepted_tokens=accepted_tokens,
            num_accepted_tokens=num_accepted_tokens,
            spec_metadata=spec_metadata,
            attn_metadata=attn_metadata)

        torch.testing.assert_close(draft_inputs["input_ids"], ref_input_ids)
        torch.testing.assert_close(draft_inputs["hidden_states"],
                                   ref_previous_hidden_states)
