import unittest

import torch
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.mtp import (MTPConfig,
                                                 MTPHiddenStatesManager,
                                                 MTPSpecMetadata, MTPWorker)


def unittest_name_func(testcase_func, param_num, param):
    name = param.args[0]
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name(name),
    )


class TestMTPUpdateMTPHiddenStates(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def load_update_mtp_hidden_states_test_cases():

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
        # BS=1, 1 context request
        batch_size = 1
        num_context_request = 1
        num_nextn_predict_layers = 1
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        mtp_past_hidden_states_ptrs_v1, mtp_past_tokens_ptrs_v1, mtp_hidden_states_tensor_pool_v1, mtp_tokens_tensor_pool_v1 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)
        mtp_past_hidden_states_ptrs_v2, mtp_past_tokens_ptrs_v2, mtp_hidden_states_tensor_pool_v2, mtp_tokens_tensor_pool_v2 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor([5, 6, 7, 8, 9],
                                 dtype=torch.int,
                                 device="cuda")

        seq_lens = torch.tensor([5], dtype=torch.int,
                                device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor([1], dtype=torch.int,
                                           device="cuda")  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([9],
                                              dtype=torch.int,
                                              device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor([
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")

        test_cases += [[
            'case0_is_thop_false', num_nextn_predict_layers,
            num_context_request, input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, False
        ]]

        test_cases += [[
            'case0_is_thop_true', num_nextn_predict_layers, num_context_request,
            input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v2,
                         device="cuda"), mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, True
        ]]

        ################## CASE 1 ##########################
        # BS=3, 3 context request, num_nextn_predict_layers = 2
        batch_size = 3
        num_context_request = 3
        num_nextn_predict_layers = 2
        hidden_size = 12
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
                9,  # request 1
                11,
                12,
                13,
                14,  # request 2
                21,
                22,
                23,
                24,
                25,
                26  # request 3
            ],
            dtype=torch.int,
            device="cuda")

        seq_lens = torch.tensor([5, 4, 6], dtype=torch.int,
                                device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                # request 1
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],

                # request 2
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],

                # request 3
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 1, 1],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor([1, 1, 1],
                                           dtype=torch.int,
                                           device="cuda")  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([8, 9],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[1] = torch.tensor([13, 14],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[2] = torch.tensor([25, 26],
                                              dtype=torch.int,
                                              device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor([
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[1] = torch.tensor([
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[2] = torch.tensor([
            [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 1, 1],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")

        test_cases += [[
            'case1_is_thop_false', num_nextn_predict_layers,
            num_context_request, input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, False
        ]]

        test_cases += [[
            'case1_is_thop_true', num_nextn_predict_layers, num_context_request,
            input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v2,
                         device="cuda"), mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, True
        ]]

        ################## CASE 2 ##########################
        # BS=1, 1 generation request, num_nextn_predict_layers = 1
        batch_size = 1
        num_context_request = 0
        num_nextn_predict_layers = 1
        hidden_size = 12
        request_ids = range(batch_size)

        # Since we will update the data inplace, so we need two data,
        # One for is_thop=False, one for is_thop=True
        mtp_past_hidden_states_ptrs_v1, mtp_past_tokens_ptrs_v1, mtp_hidden_states_tensor_pool_v1, mtp_tokens_tensor_pool_v1 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)
        mtp_past_hidden_states_ptrs_v2, mtp_past_tokens_ptrs_v2, mtp_hidden_states_tensor_pool_v2, mtp_tokens_tensor_pool_v2 = gen_data(
            batch_size, num_nextn_predict_layers, hidden_size)

        input_ids = torch.tensor([5, 42], dtype=torch.int, device="cuda")

        seq_lens = torch.tensor([2], dtype=torch.int,
                                device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor([2], dtype=torch.int,
                                           device="cuda")  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([42],
                                              dtype=torch.int,
                                              device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor([
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")

        test_cases += [[
            'case2_is_thop_false', num_nextn_predict_layers,
            num_context_request, input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, False
        ]]

        test_cases += [[
            'case2_is_thop_true', num_nextn_predict_layers, num_context_request,
            input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v2,
                         device="cuda"), mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, True
        ]]

        ################## CASE 3 ##########################
        # BS=4, 4 generation request, num_nextn_predict_layers = 1
        batch_size = 4
        num_context_request = 0
        num_nextn_predict_layers = 1
        hidden_size = 12
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
                6,  # request 1
                7,
                8,  # request 2
                9,
                10,  # request 3
                11,
                12  # request 4
            ],
            dtype=torch.int,
            device="cuda")

        seq_lens = torch.tensor([2, 2, 2, 2], dtype=torch.int,
                                device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor([2, 1, 1, 2],
                                           dtype=torch.int,
                                           device="cuda")  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([6],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[1] = torch.tensor([7],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[2] = torch.tensor([9],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[3] = torch.tensor([12],
                                              dtype=torch.int,
                                              device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor([
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[1] = torch.tensor([
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[2] = torch.tensor([
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[3] = torch.tensor([
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")

        test_cases += [[
            'case3_is_thop_false', num_nextn_predict_layers,
            num_context_request, input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, False
        ]]

        test_cases += [[
            'case3_is_thop_true', num_nextn_predict_layers, num_context_request,
            input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v2,
                         device="cuda"), mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, True
        ]]

        ################## CASE 4 ##########################
        # BS=4, 2 context request, 2 generation request, num_nextn_predict_layers = 2
        num_context = 2
        num_generation = 2
        num_context_request = num_context
        batch_size = num_context + num_generation
        num_nextn_predict_layers = 2
        hidden_size = 12
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
                9,  # request 1
                10,
                11,
                12,
                13,  # request 2
                26,
                27,
                28,  # request 3
                31,
                32,
                33  # request 4
            ],
            dtype=torch.int,
            device="cuda")

        seq_lens = torch.tensor([5, 4, 3, 3], dtype=torch.int,
                                device="cuda")  # [batch_size]

        hidden_states = torch.tensor(
            [
                # request 1
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],

                # request 2
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 1, 1],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],

                # request 3
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 1],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
                [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],

                # request 4
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
            ],
            dtype=torch.float32,
            device="cuda")  # [prompt_length, hidden_size]

        num_accepted_tokens = torch.tensor([1, 1, 3, 1],
                                           dtype=torch.int,
                                           device="cuda")  # [batch_size]

        ref_mtp_tokens_dict = dict()
        ref_mtp_tokens_dict[0] = torch.tensor([8, 9],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[1] = torch.tensor([12, 13],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[2] = torch.tensor([27, 28],
                                              dtype=torch.int,
                                              device="cuda")
        ref_mtp_tokens_dict[3] = torch.tensor([3, 31],
                                              dtype=torch.int,
                                              device="cuda")

        ref_mtp_hidden_state_dict = dict()
        ref_mtp_hidden_state_dict[0] = torch.tensor([
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 1, 1],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[1] = torch.tensor([
            [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 1, 1],
            [80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[2] = torch.tensor([
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1, 1],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")
        ref_mtp_hidden_state_dict[3] = torch.tensor([
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 1, 1],
        ],
                                                    dtype=torch.float32,
                                                    device="cuda")

        test_cases += [[
            'case4_is_thop_false', num_nextn_predict_layers,
            num_context_request, input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v1, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v1,
                         device="cuda"), mtp_hidden_states_tensor_pool_v1,
            mtp_tokens_tensor_pool_v1, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, False
        ]]

        test_cases += [[
            'case4_is_thop_true', num_nextn_predict_layers, num_context_request,
            input_ids, seq_lens, hidden_states,
            torch.tensor(mtp_past_hidden_states_ptrs_v2, device="cuda"),
            torch.tensor(mtp_past_tokens_ptrs_v2,
                         device="cuda"), mtp_hidden_states_tensor_pool_v2,
            mtp_tokens_tensor_pool_v2, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, True
        ]]

        return test_cases

    @parameterized.expand(load_update_mtp_hidden_states_test_cases,
                          name_func=unittest_name_func)
    def test_mtp_update_mtp_hidden_states(
            self, test_case_name, num_nextn_predict_layers, num_context_request,
            input_ids, seq_lens, hidden_states, mtp_hidden_states_ptrs,
            mtp_past_tokens_ptrs, mtp_hidden_states_tensor_pool,
            mtp_tokens_tensor_pool, request_ids, num_accepted_tokens,
            ref_mtp_tokens_dict, ref_mtp_hidden_state_dict, is_thop):

        batch_size = len(request_ids)
        batch_size - num_context_request
        hidden_size = hidden_states.shape[1]
        spec_config = MTPConfig(
            num_nextn_predict_layers=num_nextn_predict_layers)

        attn_metadata = TrtllmAttentionMetadata(max_num_requests=batch_size,
                                                max_num_tokens=1024,
                                                kv_cache_manager=None)
        attn_metadata.seq_lens = seq_lens.to('cpu')
        attn_metadata.num_contexts = num_context_request

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
        spec_metadata.mtp_hidden_states_ptrs = mtp_hidden_states_ptrs
        spec_metadata.mtp_past_tokens_ptrs = mtp_past_tokens_ptrs

        spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool = mtp_hidden_states_tensor_pool
        spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool = mtp_tokens_tensor_pool
        spec_metadata.prepare()

        mtpworker = MTPWorker(spec_config)
        mtpworker.is_thop = is_thop

        mtpworker.update_mtp_hidden_states(
            input_ids=input_ids,
            hidden_states=hidden_states,
            num_accepted_tokens=num_accepted_tokens,
            spec_metadata=spec_metadata,
            attn_metadata=attn_metadata)

        # Verify
        mtp_past_hidden_states_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_hidden_states_pool
        mtp_past_tokens_pool = spec_metadata.mtp_hidden_states_manager.mtp_past_tokens_pool

        for bix in range(batch_size):
            torch.testing.assert_close(mtp_past_tokens_pool[bix],
                                       ref_mtp_tokens_dict[bix])
            torch.testing.assert_close(mtp_past_hidden_states_pool[bix],
                                       ref_mtp_hidden_state_dict[bix])
