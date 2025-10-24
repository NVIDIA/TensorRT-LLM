import copy
import functools
import os
import re
from contextlib import contextmanager
from typing import List

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import (EagleDecodingConfig, KvCacheConfig,
                                 MTPDecodingConfig)

from ..conftest import llm_models_root
from .accuracy_core import GSM8K, MMLU, LlmapiAccuracyTestHarness


def extract_markdown_tables(file_path: str) -> List[str]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at path '{file_path}'")
        return []

    # Assume there is only one table in docs/source/features/feature-combination-matrix.md
    all_tables = []
    current_table = []
    in_potential_table = False

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        if stripped_line.startswith('|') and stripped_line.endswith('|'):
            # This line belongs to a table
            current_table.append(line)
            in_potential_table = True
            continue

        if in_potential_table:
            if len(current_table) >= 2:
                # The crucial check: ensure the second line is a separator line (e.g., |---|---|)
                second_line = current_table[1].strip()
                if re.match(r'^\|[\s:-]*[-]+\s*\|.*\|$', second_line):
                    all_tables.append("".join(current_table).strip())

            # Reset the state for the next potential table
            current_table = []
            in_potential_table = False

    # Final check for a table that ends exactly at the end of the file
    if in_potential_table and len(current_table) >= 2:
        second_line = current_table[1].strip()
        if re.match(r'^\|[\s:-]*[-]+\s*\|.*\|$', second_line):
            all_tables.append("".join(current_table).strip())

    return all_tables


def parse_feature_combination_table(feature_combination_table):
    lines = feature_combination_table.strip().split('\n')
    processed_lines = [line.strip() for line in lines]
    header_line = processed_lines[0].strip('|')
    headers = [col.strip().lower() for col in header_line.split('|')][1:]
    ref_result = {row: {col: None for col in headers} for row in headers}
    data_lines = processed_lines[2:]
    for i in range(len(data_lines)):
        line = data_lines[i]
        _row_values = [val.strip() for val in line.strip('|').split('|')]

        row_header = _row_values[0].lower()
        row_values = _row_values[1:]

        assert len(row_values) == len(
            headers), "same number in one row as headers"

        for j in range(len(row_values)):
            column_header = headers[j].lower()
            value = row_values[j]
            if value.strip().lower() == "yes":
                ref_result[row_header][column_header] = True
                ref_result[column_header][row_header] = True
            elif value.strip().lower() == "no":
                ref_result[row_header][column_header] = False
                ref_result[column_header][row_header] = False

    return ref_result


@functools.lru_cache(None)
def get_ref_result():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ref_result_path = os.path.join(
        current_dir, '../../../../',
        'docs/source/features/feature-combination-matrix.md')
    all_tables = extract_markdown_tables(ref_result_path)

    assert len(
        all_tables) == 1, "Only one table for features combination is expected"

    return parse_feature_combination_table(all_tables[0])


def assert_passing(feature_list, result):
    assert len(
        feature_list
    ) == 2, "Only a combination of 2 features supported in this check function"
    ref_result = get_ref_result()
    assert ref_result[feature_list[0]][feature_list[
        1]] == result, f"Failed the feature combination test between {feature_list[0]} and {feature_list[1]}, expected result is {ref_result}, actual result is {result}"


@contextmanager
def temp_model_path(obj, model_name, model_path):
    orig_model_name = getattr(obj, "MODEL_NAME")
    orig_model_path = getattr(obj, "MODEL_PATH")
    setattr(obj, "MODEL_NAME", model_name)
    setattr(obj, "MODEL_PATH", model_path)
    try:
        yield
    finally:
        setattr(obj, "MODEL_NAME", orig_model_name)
        setattr(obj, "MODEL_PATH", orig_model_path)


class TestFeatureCombination(LlmapiAccuracyTestHarness):
    """
    Base class for testing feature combinations.
    Given test time constraints and assuming features enabled by default are already tested, this
    template focuses only on critical or opt-in features. Together with a derived class,
    it enables full combination tests; e.g., TestChunkedPrefill::test_mtp validates chunked prefill paired with MTP.
    """
    PartialLLM = None
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    ctx_server_config = {}
    gen_server_config = {}
    derived_feature = None

    def test_chunked_prefill(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        kv_cache_config = self.PartialLLM.keywords[
            "kv_cache_config"] if "kv_cache_config" in self.PartialLLM.keywords else KvCacheConfig(
                free_gpu_memory_fraction=0.5)

        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=kv_cache_config,
                enable_chunked_prefill=True,
                max_num_tokens=256,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)
        assert self.derived_feature is not None, "self.derived_feature shouldn't be None"
        assert_passing([self.derived_feature, "Chunked Prefill".lower()], True)

    def test_mtp(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        model_name = "deepseek-ai/DeepSeek-V3-Lite"
        model_path = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
        mtp_nextn = 2
        mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
        kv_cache_config = self.PartialLLM.keywords[
            "kv_cache_config"] if "kv_cache_config" in self.PartialLLM.keywords else KvCacheConfig(
                free_gpu_memory_fraction=0.75)
        with self.PartialLLM(
                model=model_path,
                kv_cache_config=kv_cache_config,
                speculative_config=mtp_config,
        ) as llm:
            task = GSM8K(model_name)
            task.evaluate(llm)
        assert self.derived_feature is not None, "self.derived_feature shouldn't be None"
        assert_passing([self.derived_feature, "MTP".lower()], True)

    def test_eagle3_one_model(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        draft_len = 4
        spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=True)
        kv_cache_config = self.PartialLLM.keywords[
            "kv_cache_config"] if "kv_cache_config" in self.PartialLLM.keywords else KvCacheConfig(
                free_gpu_memory_fraction=0.5)
        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=kv_cache_config,
                speculative_config=spec_config,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

        assert self.derived_feature is not None, "self.derived_feature shouldn't be None"
        assert_passing(
            [self.derived_feature, "EAGLE-3(One Model Engine)".lower()], True)

    def test_eagle3_two_model(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        draft_len = 4
        spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                          speculative_model_dir=eagle_model_dir,
                                          eagle3_one_model=False)
        kv_cache_config = self.PartialLLM.keywords[
            "kv_cache_config"] if "kv_cache_config" in self.PartialLLM.keywords else KvCacheConfig(
                free_gpu_memory_fraction=0.5)
        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=kv_cache_config,
                speculative_config=spec_config,
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

        assert self.derived_feature is not None, "self.derived_feature shouldn't be None"
        assert_passing(
            [self.derived_feature, "EAGLE-3(Two Model Engine)".lower()], True)

    def test_kvcache_reuse(self):
        if self.PartialLLM == None:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")

        with self.PartialLLM(
                model=self.MODEL_PATH,
                kv_cache_config=KvCacheConfig(enable_block_reuse=True,
                                              free_gpu_memory_fraction=0.5),
        ) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

        assert self.derived_feature is not None, "self.derived_feature shouldn't be None"
        assert_passing([self.derived_feature, "KV Cache Reuse".lower()], True)

    def test_disaggregated_serving(self,
                                   _ctx_server_config=None,
                                   _gen_server_config=None,
                                   _derived_feature=None):

        from .test_disaggregated_serving import launch_disaggregated_llm
        ctx_server_config = copy.deepcopy(
            _ctx_server_config if _ctx_server_config else self.ctx_server_config
        )
        gen_server_config = copy.deepcopy(
            _gen_server_config if _gen_server_config else self.gen_server_config
        )
        if not ctx_server_config and not gen_server_config:
            pytest.skip(
                "LLMs are not well-suited for feature combination testing.")
        ctx_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        gen_server_config["cache_transceiver_config"] = {"backend": "DEFAULT"}
        disaggregated_server_config = {
            "hostname": "localhost",
            "port": 8000,
            "backend": "pytorch",
            "context_servers": {
                "num_instances": 1,
                "urls": ["localhost:8001"]
            },
            "generation_servers": {
                "num_instances": 1,
                "urls": ["localhost:8002"]
            }
        }
        with launch_disaggregated_llm(disaggregated_server_config,
                                      ctx_server_config, gen_server_config,
                                      self.MODEL_PATH) as llm:
            task = MMLU(self.MODEL_NAME)
            task.evaluate(llm)

        derived_feature = _derived_feature if _derived_feature is not None else self.derived_feature
        assert derived_feature is not None, "derived_feature shouldn't be None"
        assert_passing([derived_feature, "Disaggregated Serving".lower()], True)


class TestChunkedPrefill(TestFeatureCombination):
    PartialLLM = functools.partial(LLM,
                                   enable_chunked_prefill=True,
                                   max_num_tokens=256)
    # Context only requests are not supported in pytorch backend when overlap is enabled
    ctx_server_config = {
        "enable_chunked_prefill": True,
        "disable_overlap_scheduler": True,
        "max_num_tokens": 256,
    }
    gen_server_config = {}
    derived_feature = "Chunked Prefill".lower()

    def test_chunked_prefill(self):
        pytest.skip("Invalid feature combination")


class TestMTP(TestFeatureCombination):
    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
    mtp_nextn = 2
    mtp_config = MTPDecodingConfig(num_nextn_predict_layers=mtp_nextn)
    PartialLLM = functools.partial(LLM, speculative_config=mtp_config)
    # Context only requests are not supported in pytorch backend when overlap is enabled
    ctx_server_config = {
        "speculative_config": {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": mtp_nextn
        },
        "disable_overlap_scheduler": True
    }
    gen_server_config = {
        "speculative_config": {
            "decoding_type": "MTP",
            "num_nextn_predict_layers": mtp_nextn
        }
    }
    derived_feature = "MTP".lower()

    def test_mtp(self):
        pytest.skip("Invalid feature combination")

    def test_eagle3_one_model(self):
        pytest.skip("Invalid feature combination")

    def test_eagle3_two_model(self):
        pytest.skip("Invalid feature combination")


class TestEagle3OneModel(TestFeatureCombination):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
    draft_len = 4
    spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                      speculative_model_dir=eagle_model_dir,
                                      eagle3_one_model=True)
    PartialLLM = functools.partial(LLM, speculative_config=spec_config)
    speculative_decoding_config = {
        "decoding_type": "Eagle",
        "max_draft_len": draft_len,
        "speculative_model_dir": eagle_model_dir,
        "eagle3_one_model": True
    }
    ctx_server_config = {
        "speculative_config": speculative_decoding_config,
        "disable_overlap_scheduler": True
    }
    gen_server_config = {"speculative_config": speculative_decoding_config}
    derived_feature = "EAGLE-3(One Model Engine)".lower()

    def test_mtp(self):
        pytest.skip("Invalid feature combination")

    def test_eagle3_one_model(self):
        pytest.skip("Invalid feature combination")

    def test_eagle3_two_model(self):
        pytest.skip("Invalid feature combination")


class TestEagle3TwoModel(TestFeatureCombination):
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL_PATH = f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct"
    eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
    draft_len = 4
    spec_config = EagleDecodingConfig(max_draft_len=draft_len,
                                      speculative_model_dir=eagle_model_dir,
                                      eagle3_one_model=False)
    PartialLLM = functools.partial(LLM, speculative_config=spec_config)
    speculative_decoding_config = {
        "decoding_type": "Eagle",
        "max_draft_len": draft_len,
        "speculative_model_dir": eagle_model_dir,
        "eagle3_one_model": False
    }
    ctx_server_config = {
        "speculative_config": speculative_decoding_config,
        "disable_overlap_scheduler": True
    }
    gen_server_config = {"speculative_config": speculative_decoding_config}
    derived_feature = "EAGLE-3(Two Model Engine)".lower()

    def test_mtp(self):
        pytest.skip("Invalid feature combination")

    def test_eagle3_one_model(self):
        pytest.skip("Invalid feature combination")

    def test_eagle3_two_model(self):
        pytest.skip("Invalid feature combination")

    def test_disaggregated_serving(self):
        pytest.skip("Failed and report in https://nvbugs/5556020")


class TestKVCacheReuse(TestFeatureCombination):
    kv_cache_config = KvCacheConfig(enable_block_reuse=True,
                                    free_gpu_memory_fraction=0.5)
    PartialLLM = functools.partial(LLM, kv_cache_config=kv_cache_config)
    ctx_server_config = {
        "kv_cache_config": {
            "enable_block_reuse": True
        },
        "disable_overlap_scheduler": True
    }
    gen_server_config = {"kv_cache_config": {"enable_block_reuse": True}}
    derived_feature = "KV Cache Reuse".lower()

    def test_kvcache_reuse(self):
        pytest.skip("Invalid feature combination")


class TestDisaggregatedServing(TestFeatureCombination):

    def test_chunked_prefill(self):
        ctx_server_config = {
            "enable_chunked_prefill": True,
            "disable_overlap_scheduler": True,
            "max_num_tokens": 256,
        }
        gen_server_config = {}
        super().test_disaggregated_serving(ctx_server_config, gen_server_config,
                                           "Chunked Prefill".lower())

    def test_mtp(self):
        mtp_nextn = 2
        ctx_server_config = {
            "speculative_config": {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": mtp_nextn
            },
            "disable_overlap_scheduler": True
        }
        gen_server_config = {
            "speculative_config": {
                "decoding_type": "MTP",
                "num_nextn_predict_layers": mtp_nextn
            }
        }
        with temp_model_path(self, "deepseek-ai/DeepSeek-V3-Lite",
                             f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"):
            super().test_disaggregated_serving(ctx_server_config,
                                               gen_server_config, "MTP".lower())

    def test_eagle3_one_model(self):
        draft_len = 4
        eagle_model_dir = f"{llm_models_root()}/EAGLE3-LLaMA3.1-Instruct-8B"
        speculative_decoding_config = {
            "decoding_type": "Eagle",
            "max_draft_len": draft_len,
            "speculative_model_dir": eagle_model_dir,
            "eagle3_one_model": True
        }
        ctx_server_config = {
            "speculative_config": speculative_decoding_config,
            "disable_overlap_scheduler": True
        }
        gen_server_config = {"speculative_config": speculative_decoding_config}
        super().test_disaggregated_serving(ctx_server_config, gen_server_config,
                                           "EAGLE-3(One Model Engine)".lower())

    def test_eagle3_two_model(self):
        pytest.skip("Failed and report in https://nvbugs/5556020")

    def test_kvcache_reuse(self):
        ctx_server_config = {
            "kv_cache_config": {
                "enable_block_reuse": True
            },
            "disable_overlap_scheduler": True
        }
        gen_server_config = {"kv_cache_config": {"enable_block_reuse": True}}
        super().test_disaggregated_serving(ctx_server_config, gen_server_config,
                                           "KV Cache Reuse".lower())

    def test_disaggregated_serving(self):
        pytest.skip("Invalid feature combination")
