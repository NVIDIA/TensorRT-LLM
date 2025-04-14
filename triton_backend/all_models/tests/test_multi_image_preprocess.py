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

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.modules["triton_python_backend_utils"] = MagicMock()
# Use PYTHONPATH=../inflight_batcher_llm/preprocessing/1/
from model import TritonPythonModel, VisionPreProcessor


class MockTokenizer:

    def __init__(self):
        self.vocab_size = 50000
        self.pad_token = 0
        self.eos_token = 2
        self.bos_token = 1

    def encode(self, text, **kwargs):
        # Simple mock to convert characters to their ASCII values starting from 100 to avoid conflicts
        # This ensures that the pad token (0) and other special tokens are not used
        return [100 + ord(c) for c in text]


class MockProcessor:

    def __call__(self, images=None, text=None, **kwargs):
        # Simple mock to convert a uint8 image into a float image
        return dict(pixel_values=images / 255.0)


@pytest.fixture
def triton_model():
    model = TritonPythonModel()
    model.tokenizer = MockTokenizer()
    model.add_special_tokens = True
    model.is_multimodal = True
    model.model_type = 'vila'
    model.max_num_images = 2
    model.vocab_size = model.tokenizer.vocab_size
    model.ptable_shape = (-1, 10, 768)
    model.tokenizer_pad_id = model.tokenizer.pad_token
    model.tokenizer_end_id = model.tokenizer.eos_token
    model.vision_preprocessor = VisionPreProcessor(model.model_type,
                                                   MockProcessor())
    return model


# Test for _process_multi_image_inputs()
@pytest.mark.parametrize(
    "query, expected_output",
    [
        # Test Case 1: Single image placeholder
        (np.array([[b"Hello <image> World"]]), [
            np.array([100 + ord(c) for c in "Hello "] + [-200] +
                     [100 + ord(c) for c in " World"])
        ]),
        # Test Case 2: Multiple image placeholders
        (np.array([[b"Image1 <image> Image2 <image> Image3"]]), [
            np.array([100 + ord(c) for c in "Image1 "] + [-200] +
                     [100 + ord(c) for c in " Image2 "] + [-200] +
                     [100 + ord(c) for c in " Image3"])
        ]),
        # Test Case 3: No image placeholders
        (np.array([[b"No images here"]
                   ]), [np.array([100 + ord(c) for c in "No images here"])]),
        # Test Case 4: Multiple image at start or end
        (np.array([[b"<image> Image1 Image2 <image>"]]), [
            np.array([-200] + [100 + ord(c)
                               for c in " Image1 Image2 "] + [-200])
        ])
    ])
def test_process_multi_image_inputs(triton_model, query, expected_output):
    output = triton_model._process_multi_image_inputs(query)
    assert len(output) == len(expected_output)
    for out, exp in zip(output, expected_output):
        assert np.array_equal(out, exp)


# Test for _split_prompt_by_images()
@pytest.mark.parametrize(
    "concatenated_ids, expected_splits",
    [
        # Test Case 1: Single image placeholder
        (
            # <pre> Hello <image> World <post>
            [
                np.array([100 + ord(c) for c in "Pre Hello "] + [-200] +
                         [100 + ord(c) for c in " World Post"])
            ],
            [[
                np.array([100 + ord(c) for c in "Pre Hello "]).reshape(1, -1),
                np.array([100 + ord(c) for c in " World Post"]).reshape(1, -1)
            ]]),
        # Test Case 2: Multiple image placeholders
        (
            # <pre> Image1 <image> Image2 <image> Image3 <post>
            [
                np.array([100 + ord(c) for c in "Pre Image1 "] + [-200] +
                         [100 + ord(c) for c in " Image2 "] + [-200] +
                         [100 + ord(c) for c in " Image3 Post"])
            ],
            [[
                np.array([100 + ord(c) for c in "Pre Image1 "]).reshape(1, -1),
                np.array([100 + ord(c) for c in " Image2 "]).reshape(1, -1),
                np.array([100 + ord(c) for c in " Image3 Post"]).reshape(1, -1)
            ]]),
        # Test Case 3: No image placeholders
        (
            # <pre> No images here <post>
            [np.array([100 + ord(c) for c in "Pre No images here Post"])],
            [[
                np.array([100 + ord(c)
                          for c in "Pre No images here Post"]).reshape(1, -1)
            ]]),
        # Test Case 4: Multiple image at start or end
        (
            # <pre> <image> Image1 Image2 <image> <post>
            [
                np.array([100 + ord(c) for c in "Pre "] + [-200] +
                         [100 + ord(c) for c in " Image1 Image2 "] + [-200] +
                         [100 + ord(c) for c in " Post"])
            ],
            [[
                np.array([100 + ord(c) for c in "Pre "]).reshape(1, -1),
                np.array([100 + ord(c)
                          for c in " Image1 Image2 "]).reshape(1, -1),
                np.array([100 + ord(c) for c in " Post"]).reshape(1, -1)
            ]])
    ])
def test_split_prompt_by_images(triton_model, concatenated_ids,
                                expected_splits):
    output = triton_model._split_prompt_by_images(concatenated_ids)
    assert len(output) == len(expected_splits)
    for out_splits, exp_splits in zip(output, expected_splits):
        assert len(out_splits) == len(exp_splits)
        for out, exp in zip(out_splits, exp_splits):
            assert np.array_equal(out, exp)


# Test for _setup_fake_prompts()
@pytest.mark.parametrize(
    "batch_size, batch_split_prompts, ptable_shape, vocab_size, expected_input_ids",
    [
        # Test Case 1: Single image placeholder in each sample
        (2, [[
            np.array([100 + ord(c) for c in "Pre Hello "]).reshape(1, -1),
            np.array([100 + ord(c) for c in " World Post"]).reshape(1, -1)
        ],
             [
                 np.array([100 + ord(c) for c in "Pre Foo "]).reshape(1, -1),
                 np.array([100 + ord(c) for c in " Bar Post"]).reshape(1, -1)
             ]], (-1, 10, 768), 50000, [
                 np.concatenate([[100 + ord(c) for c in "Pre Hello "],
                                 np.arange(50000, 50000 + 10),
                                 [100 + ord(c) for c in " World Post"]]),
                 np.concatenate([[100 + ord(c) for c in "Pre Foo "],
                                 np.arange(50000, 50000 + 10),
                                 [100 + ord(c)
                                  for c in " Bar Post"], [0, 0, 0, 0]])
             ]),
        # Test Case 2: Multiple image placeholders in a sample
        (1, [[
            np.array([100 + ord(c) for c in "Pre Image1 "]).reshape(1, -1),
            np.array([100 + ord(c) for c in " Image2 "]).reshape(1, -1),
            np.array([100 + ord(c) for c in " Image3 Post"]).reshape(1, -1)
        ]], (-1, 10, 768), 50000, [
            np.concatenate([[100 + ord(c) for c in "Pre Image1 "],
                            np.arange(50000, 50000 + 10),
                            [100 + ord(c) for c in " Image2 "],
                            np.arange(50010, 50010 + 10),
                            [100 + ord(c) for c in " Image3 Post"]])
        ]),
        # Test Case 3: No image placeholders
        (1, [[
            np.array([100 + ord(c)
                      for c in "Pre No image here Post"]).reshape(1, -1)
        ]], (-1, 10, 768), 50000,
         [[100 + ord(c) for c in "Pre No image here Post"]]),
        # Test Case 4: Multiple image at start or end
        (2, [[
            np.array([100 + ord(c) for c in "Pre "]).reshape(1, -1),
            np.array([100 + ord(c) for c in " Image1 Image2 "]).reshape(1, -1),
            np.array([100 + ord(c) for c in " Post"]).reshape(1, -1)
        ],
             [
                 np.array([100 + ord(c) for c in "Pre "]).reshape(1, -1),
                 np.array([100 + ord(c)
                           for c in " Image3 Image4 "]).reshape(1, -1),
                 np.array([100 + ord(c) for c in " Post"]).reshape(1, -1)
             ]], (-1, 10, 768), 50000, [
                 np.concatenate([[100 + ord(c) for c in "Pre "],
                                 np.arange(50000, 50000 + 10),
                                 [100 + ord(c) for c in " Image1 Image2 "],
                                 np.arange(50010, 50010 + 10),
                                 [100 + ord(c) for c in " Post"]]),
                 np.concatenate([[100 + ord(c) for c in "Pre "],
                                 np.arange(50000, 50000 + 10),
                                 [100 + ord(c) for c in " Image3 Image4 "],
                                 np.arange(50010, 50010 + 10),
                                 [100 + ord(c) for c in " Post"]])
             ])
    ])
def test_setup_fake_prompts(triton_model, batch_size, batch_split_prompts,
                            ptable_shape, vocab_size, expected_input_ids):
    triton_model.ptable_shape = ptable_shape
    triton_model.vocab_size = vocab_size
    output = triton_model._setup_fake_prompts(batch_size, batch_split_prompts)
    assert output.shape[0] == batch_size
    for out, exp in zip(output, expected_input_ids):
        assert np.array_equal(out, exp)


# Test for _process_multi_image_inputs()
@pytest.mark.parametrize(
    "query, image_bytes, expected_output",
    [
        # Test Case 1: Single image placeholder
        (np.array([[b"Hello <image> World"]
                   ]), np.ones((1, 1, 32, 32, 3), dtype=np.uint8) * 255,
         dict(PIXEL_VALUES=np.ones((1, 1, 32, 32, 3), dtype=np.float16)))
    ])
def test_process_image_for_encoder(triton_model, query, image_bytes,
                                   expected_output):
    output = triton_model.vision_preprocessor.mllama_process(
        query, image_bytes=image_bytes)
    assert output.keys() == expected_output.keys()
    for key in output.keys():
        assert np.array_equal(output[key], expected_output[key])
