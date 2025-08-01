# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Dict, Optional

import numpy as np
import pytest
# Use PYTHONPATH=../inflight_batcher_llm/tensorrt_llm_bls/1/
from lib.decode import *


class MockDecoder(Decoder):

    def __init__(self,
                 streaming=False,
                 accumulate=False,
                 data_dict: Optional[Dict] = None):
        super().__init__(streaming=streaming, accumulate=accumulate)
        self.data_dict = data_dict
        self.draft_step = -1
        self.target_step = -1

        self.draft_num_calls = 0
        self.target_num_calls = 0

    def preprocess(self, request: Request) -> PreprocResponse:
        return PreprocResponse(
            input_ids=np.array([self.data_dict["input_ids"]]),
            input_lengths=np.array([[len(self.data_dict["input_ids"])]]),
            stop_words_list=np.array([[[]]]))

    def _postprocess(self, tokens: np.ndarray,
                     sequence_lengths: Optional[np.ndarray],
                     gen_response: GenerationResponse) -> Response:
        target_output = self.data_dict["target_output"][self.target_step]
        return Response(text_output=np.array([target_output["output_text"]]))

    def _draft_generate_non_streaming(
            self, preproc: PreprocResponse, request: Request,
            num_draft_tokens: int) -> GenerationResponse:
        self.draft_num_calls += 1
        self.draft_step += 1
        draft_output = self.data_dict["draft_output"][self.draft_step]
        response = GenerationResponse(
            output_ids=np.array([[draft_output["output_ids"]]]),
            generation_logits=None,
            sequence_length=np.array([[draft_output["sequence_length"]]]))
        if self.data_dict.get("use_draft_logits", False):
            print("!!!")
            response.generation_logits = draft_output["generation_logits"]
        return response

    def _generate(
        self,
        preproc: PreprocResponse,
        request: Request,
        draft_request: Optional[DraftRequest] = None,
        multimodal_enc_response: Optional[MultimodalEncResponse] = None
    ) -> Generator[GenerationResponse, None, None]:
        for idx, target_output in enumerate(self.data_dict["target_output"]):
            self.target_num_calls += 1
            self.target_step = idx
            output_len = len(target_output["output_ids"])
            yield GenerationResponse(output_ids=np.array(
                [[target_output["output_ids"]]]),
                                     sequence_length=np.array([[output_len]]))

    def _generate_non_streaming(
        self,
        preproc: PreprocResponse,
        request: Request,
        draft_request: Optional[DraftRequest] = None,
        multimodal_enc_response: Optional[MultimodalEncResponse] = None
    ) -> GenerationResponse:
        self.target_num_calls += 1
        # Return the full completion (final step) if not using speculative decoding in non-streaming mode
        if not self.data_dict["use_speculative"]:
            self.target_step = (len(self.data_dict["target_output"]) - 2)
        else:
            print(draft_request)
            assert draft_request is not None
            if draft_request.draft_input_ids is not None:
                assert draft_request.draft_input_ids.shape[1] > 0
                if self.data_dict.get("use_draft_logits", False):
                    assert draft_request.draft_logits is not None
                    assert draft_request.draft_logits.shape[
                        1] == draft_request.draft_input_ids.shape[1]

        self.target_step += 1
        target_output = self.data_dict["target_output"][self.target_step]
        output_len = len(target_output["output_ids"])
        return GenerationResponse(output_ids=np.array(
            [[target_output["output_ids"]]]),
                                  sequence_length=np.array([[output_len]]))


decode_testcases = [
    {
        "text_input":
        "Deep learning is",
        "max_tokens":
        10,
        "use_speculative":
        False,
        "input_ids": [1, 10, 11, 23],
        "target_output": [{
            "output_ids": [1, 10, 11, 23, 7],
            "output_text": "Deep learning is a"
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9],
            "output_text": "Deep learning is a subset"
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21],
            "output_text": "Deep learning is a subset of"
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21, 22],
            "output_text": "Deep learning is a subset of Machine"
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21, 22, 11],
            "output_text":
            "Deep learning is a subset of Machine learning"
        }]
    },
    {
        "text_input":
        "Deep learning is",
        "max_tokens":
        10,
        "use_speculative":
        True,
        "num_draft_tokens":
        3,
        "use_draft_logits":
        False,
        "input_ids": [1, 10, 11, 23],
        "target_output": [{
            "output_ids": [1, 10, 11, 23, 7, 9, 21],
            "output_text": "Deep learning is a subset of"
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21, 22, 11],
            "output_text":
            "Deep learning is a subset of Machine learning"
        }],
        "draft_output": [{
            "output_ids": [1, 10, 11, 23, 7, 9, 22],
            "sequence_length": 7,
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21, 22, 11],
            "sequence_length": 9,
        }]
    },
    {
        "text_input":
        "Deep learning is",
        "max_tokens":
        10,
        "use_speculative":
        True,
        "num_draft_tokens":
        3,
        "use_draft_logits":
        True,
        "input_ids": [1, 10, 11, 23],
        "target_output": [{
            "output_ids": [1, 10, 11, 23, 7, 9, 21],
            "output_text": "Deep learning is a subset of"
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21, 22, 11],
            "output_text":
            "Deep learning is a subset of Machine learning"
        }],
        "draft_output": [{
            "output_ids": [1, 10, 11, 23, 7, 9, 22],
            "sequence_length": 7,
            "generation_logits": np.random.rand(1, 1, 7, 1024),
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21, 22, 11],
            "sequence_length": 9,
            "generation_logits": np.random.rand(1, 1, 7, 1024),
        }]
    },
]


@pytest.mark.parametrize("test_case", decode_testcases)
def test_decode(test_case):

    request = Request(
        text_input=np.array([[test_case["text_input"]]], dtype=object),
        max_tokens=np.array([[test_case["max_tokens"]]], dtype=np.int32),
        num_draft_tokens=(np.array([[test_case["num_draft_tokens"]]],
                                   dtype=np.int32)
                          if "num_draft_tokens" in test_case else None),
        use_draft_logits=(np.array([[test_case["use_draft_logits"]]],
                                   dtype=bool)
                          if "use_draft_logits" in test_case else None),
        stop_words=np.array([[[]]]))
    # Last index is the expected response
    expected_res = Response(text_output=np.array(
        [test_case["target_output"][-1]["output_text"]], dtype=object))

    if not test_case["use_speculative"]:
        # Test non speculative mode

        # non-streaming
        d = MockDecoder(data_dict=test_case, streaming=False)
        for res in d.decode(request):
            assert expected_res == res
        assert d.target_num_calls == 1

        # streaming
        d = MockDecoder(data_dict=test_case, streaming=True)
        final_res = None
        for res in d.decode(request):
            final_res = res
        assert final_res == expected_res
        assert d.target_num_calls == len(test_case["target_output"])
    else:
        # Test speculative decoding
        d = MockDecoder(data_dict=test_case)
        final_res = None
        for res in d.decode(request, speculative_decoding=True):
            final_res = res
        assert final_res == expected_res
        num_steps = len(test_case["draft_output"])
        assert d.target_num_calls == num_steps
        assert d.draft_num_calls == num_steps


length_stop_testcases = [{
    "text_input":
    "Deep learning is",
    "max_tokens":
    1,
    "use_speculative":
    True,
    "num_draft_tokens":
    3,
    "input_ids": [1, 10, 11, 23],
    "target_output": [{
        "output_ids": [1, 10, 11, 23],
        "output_text": "Deep learning is a"
    }, {
        "output_ids": "not important",
        "output_text": "not important"
    }],
    "draft_output": [{
        "output_ids": ["not important"],
        "sequence_length": 0
    }, {
        "output_ids": ["not important"],
        "sequence_length": 0
    }]
}]


@pytest.mark.parametrize("test_case", length_stop_testcases)
def test_length_stop(test_case):
    # Since max_tokens is 1, test if get the first output as the final output
    # and make sure the draft model is never called
    request = Request(
        text_input=np.array([[test_case["text_input"]]], dtype=object),
        max_tokens=np.array([[test_case["max_tokens"]]], dtype=np.int32),
        num_draft_tokens=(np.array([[test_case["num_draft_tokens"]]],
                                   dtype=np.int32)
                          if "num_draft_tokens" in test_case else None),
        stop_words=np.array([[[]]]))
    # Index 0 is the expected response
    expected_res = Response(text_output=np.array(
        [test_case["target_output"][0]["output_text"]], dtype=object))

    d = MockDecoder(data_dict=test_case)
    final_res = None
    for res in d.decode(request, speculative_decoding=True):
        final_res = res
    assert final_res == expected_res
    assert d.target_num_calls == 1
    assert d.draft_num_calls == 0


early_stopping_testcases = [
    {
        "text_input":
        "Deep learning is",
        "max_tokens":
        10,
        "use_speculative":
        True,
        "num_draft_tokens":
        3,
        "input_ids": [1, 10, 11, 23],
        "target_output": [{
            "output_ids": [1, 10, 11, 23, 7, 9, 21],
            "output_text": "Deep learning is a subset of"
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21],
            "output_text": "Deep learning is a subset of Machine"
        }, {
            "output_ids": ["not important"],
            "output_text": "not important"
        }],
        "draft_output": [{
            "output_ids": [1, 10, 11, 23, 7, 9, 22],
            "sequence_length": 7
        }, {
            "output_ids": [1, 10, 11, 23, 7, 9, 21, 22, 11],
            "sequence_length": 9
        }, {
            "output_ids": ["not important"],
            "sequence_length": 0
        }]
    },
]


@pytest.mark.parametrize("test_case", early_stopping_testcases)
def test_early_stopping(test_case):

    request = Request(
        text_input=np.array([[test_case["text_input"]]], dtype=object),
        max_tokens=np.array([[test_case["max_tokens"]]], dtype=np.int32),
        num_draft_tokens=(np.array([[test_case["num_draft_tokens"]]],
                                   dtype=np.int32)
                          if "num_draft_tokens" in test_case else None),
        stop_words=np.array([[[]]]))
    # Index 1 is the expected response
    expected_res = Response(text_output=np.array(
        [test_case["target_output"][1]["output_text"]], dtype=object))

    d = MockDecoder(data_dict=test_case)
    final_res = None
    for res in d.decode(request, speculative_decoding=True):
        final_res = res
    assert final_res == expected_res
    assert d.target_num_calls == 2
    assert d.draft_num_calls == 2


def test_request_validation():
    req = Request()
    with pytest.raises(RequestValidationError):
        req.validate()
    req.text_input = np.array([["input string"]], dtype=object)
    with pytest.raises(RequestValidationError):
        req.validate()
    req.max_tokens = np.array([[10]])
    req.validate()

    req.stream = np.array([[True]])
    req.num_draft_tokens = np.array([[5]])

    with pytest.raises(RequestValidationError):
        req.validate()
