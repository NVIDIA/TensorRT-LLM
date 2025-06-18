from types import MethodType
from typing import Optional

import pytest
from api_stability_core import (ApiStabilityTestHarness, ClassSnapshot,
                                MethodSnapshot)

from tensorrt_llm import LLM
from tensorrt_llm.bindings import executor as tllme
from tensorrt_llm.executor.result import IterationResult
from tensorrt_llm.llmapi import (CalibConfig, CompletionOutput,
                                 GuidedDecodingParams, QuantConfig,
                                 RequestOutput)
from tensorrt_llm.sampling_params import (BatchedLogitsProcessor,
                                          LogitsProcessor, SamplingParams)


class TestSamplingParams(ApiStabilityTestHarness):
    TEST_CLASS = SamplingParams
    REFERENCE_FILE = "sampling_params.yaml"

    def test_get_sampling_config(self):
        expected_fields = {
            "beam_width",
            "beam_width_array",
            "top_k",
            "top_p",
            "top_p_min",
            "top_p_reset_ids",
            "top_p_decay",
            "seed",
            "temperature",
            "min_tokens",
            "beam_search_diversity_rate",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
            "length_penalty",
            "early_stopping",
            "no_repeat_ngram_size",
            "num_return_sequences",
            "min_p",
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

        def new_get_stats_async(self,
                                timeout: Optional[float] = 2
                                ) -> IterationResult:
            pass

        new_get_stats_async.__doc__ = self.TEST_CLASS.get_stats_async.__doc__

        mocker.patch.object(self.TEST_CLASS,
                            "get_stats_async",
                            new=new_get_stats_async)
        self.test_signature()
        self.test_docstring()

    def test_modified_method_with_modified_signature(self, mocker):

        def new_get_stats_async(self,
                                timeout: Optional[int] = 2) -> IterationResult:
            pass

        mocker.patch.object(self.TEST_CLASS,
                            "get_stats_async",
                            new=new_get_stats_async)
        with pytest.raises(AssertionError):
            self.test_signature()
        with pytest.raises(AssertionError):
            self.test_docstring()

    def test_modified_docstring(self, mocker):
        mocker.patch.object(self.TEST_CLASS, "__doc__", new="")
        self.test_signature()
        with pytest.raises(AssertionError):
            self.test_docstring()

    def test_fine_grained_error(self):
        # change the dtype of max_batch_size to float to trigger a fine-grained error
        self.reference.methods["__init__"].parameters[
            "max_batch_size"].annotation = "float"
        with pytest.raises(AssertionError) as e:
            self.test_signature()
            assert "LLM.max_batch_size annotation: typing.Optional[int] != <class 'float'>" in str(
                e.value.__cause__)

        # restore the original dtype
        self.reference.methods["__init__"].parameters[
            "max_batch_size"].annotation = "int"


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
