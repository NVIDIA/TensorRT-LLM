import unittest

import torch
from parameterized import parameterized
from utils.util import unittest_name_func

import tensorrt_llm
from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.speculative.mtp import (MTPConfig, MTPSpecMetadata,
                                                 MTPWorker)


class TestMTPSampleAndAcceptDraftTokens(unittest.TestCase):

    def setUp(self):
        tensorrt_llm.logger.set_level('warning')

    def load_sample_and_accept_draft_tokens_test_cases():
        test_cases = []

        # '''
        ################# CASE 0 ##########################
        # BS=1, 1 context request
        mtp_num_modules = 1
        num_context_requests = 1
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
            ],
            dtype=torch.float32,
            device="cuda")  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        draft_len = torch.tensor([0], dtype=torch.int,
                                 device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 0]], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        ref_num_accepted_tokens = torch.tensor([1],
                                               dtype=torch.int,
                                               device="cuda")  # [batch_size]

        test_cases += [[
            mtp_num_modules, logits, draft_tokens, draft_len,
            num_context_requests, ref_accepted_tokens, ref_num_accepted_tokens
        ]]

        ################# CASE 1 ##########################
        # BS=4, 4 context request
        mtp_num_modules = 1
        num_context_requests = 4
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
            ],
            dtype=torch.float32,
            device="cuda")  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        draft_len = torch.tensor([0, 0, 0, 0], dtype=torch.int,
                                 device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 0], [3, 0], [3, 0], [6, 0]], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        ref_num_accepted_tokens = torch.tensor([1, 1, 1, 1],
                                               dtype=torch.int,
                                               device="cuda")  # [batch_size]

        test_cases += [[
            mtp_num_modules, logits, draft_tokens, draft_len,
            num_context_requests, ref_accepted_tokens, ref_num_accepted_tokens
        ]]

        ################# CASE 2 ##########################
        # BS=1, 1 generation request
        # Assume there are 3 MTP layers
        # For each generation request, there are four logits here: one from golden token + three draft tokens
        mtp_num_modules = 3
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
            ],
            dtype=torch.float32,
            device="cuda")  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 3, 4], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        draft_len = torch.tensor([3], dtype=torch.int,
                                 device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3, 2, 0]], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        ref_num_accepted_tokens = torch.tensor([3],
                                               dtype=torch.int,
                                               device="cuda")  # [batch_size]

        test_cases += [[
            mtp_num_modules, logits, draft_tokens, draft_len,
            num_context_requests, ref_accepted_tokens, ref_num_accepted_tokens
        ]]

        ################# CASE 3 ##########################
        # BS=2, 2 generation request
        # Assume there are 1 MTP layers
        # For each generation request, there are two logits here: one golden token + one draft token
        mtp_num_modules = 1
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
            ],
            dtype=torch.float32,
            device="cuda")  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 5], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        draft_len = torch.tensor([1, 1], dtype=torch.int,
                                 device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3], [4, 0]], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        ref_num_accepted_tokens = torch.tensor([2, 1],
                                               dtype=torch.int,
                                               device="cuda")  # [batch_size]

        test_cases += [[
            mtp_num_modules, logits, draft_tokens, draft_len,
            num_context_requests, ref_accepted_tokens, ref_num_accepted_tokens
        ]]

        ################# CASE 4 ##########################
        # BS=2, 2 generation request
        # Assume there are 3 MTP layers
        # For each generation request, there are four logits here: one golden token + three draft token
        mtp_num_modules = 3
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
                [-100, -100, -100, -100, -100, 0, -100, -100],  # Top1 id = 5
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
            ],
            dtype=torch.float32,
            device="cuda")  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 3, 4, 4, 7, 3], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        draft_len = torch.tensor([3, 3], dtype=torch.int,
                                 device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3, 2, 0], [4, 6, 0, 0]], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        ref_num_accepted_tokens = torch.tensor([3, 2],
                                               dtype=torch.int,
                                               device="cuda")  # [batch_size]

        test_cases += [[
            mtp_num_modules, logits, draft_tokens, draft_len,
            num_context_requests, ref_accepted_tokens, ref_num_accepted_tokens
        ]]

        ################# CASE 5 ##########################
        # BS=3, 3 generation request, 2 accept partial, 1 accept all, 1 accept none
        # Assume there are 3 MTP layers
        # For each generation request, there are four logits here: one golden token + three draft token
        mtp_num_modules = 3
        num_context_requests = 0
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
                [-100, -100, -100, -100, -100, 0, -100, -100],  # Top1 id = 5
                [-100, -100, 0, -100, -100, -100, -100, -100],  # Top1 id = 2
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, 0, -100, -100, -100, -100],  # Top1 id = 3
                [-100, -100, -100, -100, -100, 0, -100, -100],  # Top1 id = 5
                [-100, -100, -100, -100, -100, -100, -100, 0],  # Top1 id = 7
            ],
            dtype=torch.float32,
            device="cuda")  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [1, 3, 5, 4, 6, 5, 5, 7, 4], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        draft_len = torch.tensor([3, 3, 3], dtype=torch.int,
                                 device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 3, 2, 0], [4, 6, 5, 2], [4, 0, 0, 0]],
            dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        ref_num_accepted_tokens = torch.tensor([3, 4, 1],
                                               dtype=torch.int,
                                               device="cuda")  # [batch_size]

        test_cases += [[
            mtp_num_modules, logits, draft_tokens, draft_len,
            num_context_requests, ref_accepted_tokens, ref_num_accepted_tokens
        ]]

        ################# CASE 6 ##########################
        # BS=2, 1 context request, 1 generation request
        # request0 is context request, and request1 is generation request
        # Assume there are 1 MTP layers
        # For each generation request, there are two logits here: one golden token + one draft token
        mtp_num_modules = 1
        num_context_requests = 1
        logits = torch.tensor(
            [
                [-100, 0, -100, -100, -100, -100, -100, -100],  # Top1 id = 1
                [-100, -100, -100, -100, 0, -100, -100, -100],  # Top1 id = 4
                [-100, -100, -100, -100, -100, -100, 0, -100],  # Top1 id = 6
            ],
            dtype=torch.float32,
            device="cuda")  # [num_tokens, vocab_size]

        draft_tokens = torch.tensor(
            [4], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        draft_len = torch.tensor([0, 1], dtype=torch.int,
                                 device="cuda")  # [batch_size]

        ref_accepted_tokens = torch.tensor(
            [[1, 0], [4, 6]], dtype=torch.int,
            device="cuda")  # [batch_size * max_draft_tokens]

        ref_num_accepted_tokens = torch.tensor([1, 2],
                                               dtype=torch.int,
                                               device="cuda")  # [batch_size]

        test_cases += [[
            mtp_num_modules, logits, draft_tokens, draft_len,
            num_context_requests, ref_accepted_tokens, ref_num_accepted_tokens
        ]]

        return test_cases

    @parameterized.expand(load_sample_and_accept_draft_tokens_test_cases,
                          name_func=unittest_name_func)
    def test_sample_and_accept_draft_tokens(self, mtp_num_modules, logits,
                                            draft_tokens, draft_len,
                                            num_context_requests,
                                            ref_accepted_tokens,
                                            ref_num_accepted_tokens):
        batch_size = len(draft_len)
        spec_config = MTPConfig(num_nextn_predict_layers=mtp_num_modules)

        # attention metedata
        attn_metadata = TrtllmAttentionMetadata(max_num_requests=batch_size,
                                                max_num_tokens=1024,
                                                kv_cache_manager=None)
        attn_metadata.seq_lens = torch.tensor(
            [1] * batch_size, dtype=torch.int)  # dummy sequence length
        attn_metadata.num_contexts = num_context_requests

        # speculative decoding metadata
        spec_metadata = MTPSpecMetadata(max_num_requests=32,
                                        max_draft_tokens=mtp_num_modules,
                                        mtp_num_modules=mtp_num_modules)
        spec_metadata.draft_tokens = draft_tokens

        # mtp worker
        mtpworker = MTPWorker(spec_config)

        # Test thop kernel
        # Test native torch op
        for is_thop in [True, False]:
            mtpworker.is_thop = is_thop
            accepted_tokens, num_accepted_tokens = mtpworker.sample_and_accept_draft_tokens(
                logits, spec_metadata, attn_metadata)

            torch.testing.assert_close(num_accepted_tokens,
                                       ref_num_accepted_tokens)
            for i in range(len(draft_len)):
                torch.testing.assert_close(
                    accepted_tokens[i][0:ref_num_accepted_tokens[i]],
                    ref_accepted_tokens[i][0:ref_num_accepted_tokens[i]])
