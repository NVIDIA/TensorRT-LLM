from types import MethodType
from typing import Optional

import pytest
from api_stability_core import (ApiStabilityTestHarness, ClassSnapshot,
                                MethodSnapshot)

from tensorrt_llm.bindings import executor as tllme
from tensorrt_llm.llmapi import (LLM, CalibConfig, CompletionOutput,
                                 GuidedDecodingParams, QuantConfig,
                                 RequestOutput)
from tensorrt_llm.llmapi.llm_utils import LlmArgs
from tensorrt_llm.sampling_params import (BatchedLogitsProcessor,
                                          LogitsProcessor, SamplingParams)


class TestSamplingParams(ApiStabilityTestHarness):
    TEST_CLASS = SamplingParams
    REFERENCE_FILE = "sampling_params.yaml"

    def test_get_sampling_config(self):
        expected_fields = {
            "beam_width", "top_k", "top_p", "top_p_min", "top_p_reset_ids",
            "top_p_decay", "seed", "random_seed", "temperature", "min_tokens",
            "min_length", "beam_search_diversity_rate", "repetition_penalty",
            "presence_penalty", "frequency_penalty", "length_penalty",
            "early_stopping", "no_repeat_ngram_size", "num_return_sequences",
            "min_p"
        }
        found_fields = {
            f
            for f in dir(tllme.SamplingConfig) if not f.startswith('__')
        }
        error_msg = (
            "Found fields in `tllme.SamplingConfig` different than expected; "
            f"if `tllme.SamplingConfig` is changed, please update {self.TEST_CLASS.__name__} accordingly."
        )
        assert found_fields == expected_fields, error_msg

    def test_get_output_config(self):
        expected_fields = {
            "return_log_probs", "return_context_logits",
            "return_generation_logits", "exclude_input_from_output",
            "return_encoder_output", "return_perf_metrics",
            "additional_model_outputs"
        }
        found_fields = {
            f
            for f in dir(tllme.OutputConfig) if not f.startswith('__')
        }
        error_msg = (
            "Found fields in `tllme.OutputConfig` different than expected; "
            f"if `tllme.OutputConfig` is changed, please update {self.TEST_CLASS.__name__} accordingly."
        )
        assert found_fields == expected_fields, error_msg


class TestGuidedDecodingParams(ApiStabilityTestHarness):
    TEST_CLASS = GuidedDecodingParams
    REFERENCE_FILE = "guided_decoding_params.yaml"


class TestLogitsProcessor(ApiStabilityTestHarness):
    TEST_CLASS = LogitsProcessor
    REFERENCE_FILE = "logits_processor.yaml"

    def create_snapshot_from_inspect(self):
        method_snapshot = MethodSnapshot.from_inspect(
            MethodType(self.TEST_CLASS.__call__, object()))
        return ClassSnapshot(methods={"__call__": method_snapshot},
                             properties={})

    def create_snapshot_from_docstring(self):
        method_snapshot = MethodSnapshot.from_docstring(
            MethodType(self.TEST_CLASS.__call__, object()))
        return ClassSnapshot(methods={"__call__": method_snapshot},
                             properties={})


class TestBatchedLogitsProcessor(ApiStabilityTestHarness):
    TEST_CLASS = BatchedLogitsProcessor
    REFERENCE_FILE = "batched_logits_processor.yaml"

    def create_snapshot_from_inspect(self):
        method_snapshot = MethodSnapshot.from_inspect(
            MethodType(self.TEST_CLASS.__call__, object()))
        return ClassSnapshot(methods={"__call__": method_snapshot},
                             properties={})

    def create_snapshot_from_docstring(self):
        method_snapshot = MethodSnapshot.from_docstring(
            MethodType(self.TEST_CLASS.__call__, object()))
        return ClassSnapshot(methods={"__call__": method_snapshot},
                             properties={})


class TestLLM(ApiStabilityTestHarness):
    TEST_CLASS = LLM
    REFERENCE_FILE = "llm.yaml"

    def test_modified_init(self, mocker):
        mocker.patch.object(self.TEST_CLASS,
                            "__init__",
                            new=lambda self, x: None)
        with pytest.raises(AssertionError):
            self.test_signature()
        self.test_docstring()

    def test_new_method(self, mocker):
        mocker.patch.object(self.TEST_CLASS,
                            "new_method",
                            new=lambda self, x: None,
                            create=True)
        with pytest.raises(AssertionError):
            self.test_signature()
        with pytest.raises(AssertionError):
            self.test_docstring()

    def test_modified_method_with_same_signature(self, mocker):

        def new_save(self, engine_dir: str) -> None:
            pass

        new_save.__doc__ = self.TEST_CLASS.save.__doc__

        mocker.patch.object(self.TEST_CLASS, "save", new=new_save)
        self.test_signature()
        self.test_docstring()

    def test_modified_method_with_modified_signature(self, mocker):

        def new_save(self, engine_dir: Optional[str]) -> None:
            pass

        mocker.patch.object(self.TEST_CLASS, "save", new=new_save)
        with pytest.raises(AssertionError):
            self.test_signature()
        with pytest.raises(AssertionError):
            self.test_docstring()

    def test_modified_docstring(self, mocker):
        mocker.patch.object(self.TEST_CLASS, "__doc__", new="")
        self.test_signature()
        with pytest.raises(AssertionError):
            self.test_docstring()


class TestLlmArgs(ApiStabilityTestHarness):
    TEST_CLASS = LlmArgs
    REFERENCE_FILE = "llm_args.yaml"


class TestCompletionOutput(ApiStabilityTestHarness):
    TEST_CLASS = CompletionOutput
    REFERENCE_FILE = "completion_output.yaml"


class TestRequestOutput(ApiStabilityTestHarness):
    TEST_CLASS = RequestOutput
    REFERENCE_FILE = "request_output.yaml"


class TestQuantConfig(ApiStabilityTestHarness):
    TEST_CLASS = QuantConfig
    REFERENCE_FILE = "quant_config.yaml"


class TestCalibConfig(ApiStabilityTestHarness):
    TEST_CLASS = CalibConfig
    REFERENCE_FILE = "calib_config.yaml"
