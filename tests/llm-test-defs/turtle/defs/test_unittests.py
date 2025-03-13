# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import itertools
import os
from subprocess import CalledProcessError

import pytest


class LLMUnitTestCase:

    def __init__(self, name: str, filter: str, test_case_filter: str = None):
        self.name = name
        self.filter = filter
        self.test_case_filter = test_case_filter


TestCases = [
    LLMUnitTestCase("api-stability", "{TEST_ROOT}/api_stability"),
    LLMUnitTestCase("bindings", "{TEST_ROOT}/bindings"),
    LLMUnitTestCase("model-runner-cpp", "{TEST_ROOT}/test_model_runner_cpp.py"),
    LLMUnitTestCase("functional", "{TEST_ROOT}/functional"),
    LLMUnitTestCase("functional-moe", "{TEST_ROOT}/functional/test_moe.py"),
    LLMUnitTestCase(
        "unit-woq-percol",
        "{TEST_ROOT}/quantization/test_weight_only_quant_matmul.py"),
    LLMUnitTestCase(
        "unit-woq-group",
        "{TEST_ROOT}/quantization/test_weight_only_groupwise_quant_matmul.py"),
    LLMUnitTestCase(
        "unit-moe-woq-group",
        "{TEST_ROOT}/quantization/test_moe_weight_only_groupwise_quant_matmul.py"
    ),
    LLMUnitTestCase("qserve-gemm",
                    "{TEST_ROOT}/quantization/test_qserve_gemm.py"),
    LLMUnitTestCase("attention-bert",
                    "{TEST_ROOT}/attention/test_bert_attention.py"),
    LLMUnitTestCase("attention-gpt",
                    "{TEST_ROOT}/attention/test_gpt_attention.py"),
    LLMUnitTestCase("attention-gpt-partition0",
                    "{TEST_ROOT}/attention/test_gpt_attention.py",
                    "partition0"),
    LLMUnitTestCase("attention-gpt-partition1",
                    "{TEST_ROOT}/attention/test_gpt_attention.py",
                    "partition1"),
    LLMUnitTestCase("attention-gpt-partition2",
                    "{TEST_ROOT}/attention/test_gpt_attention.py",
                    "partition2"),
    LLMUnitTestCase("attention-gpt-partition3",
                    "{TEST_ROOT}/attention/test_gpt_attention.py",
                    "partition3"),
    LLMUnitTestCase("attention-gpt-xqa-generic",
                    "{TEST_ROOT}/attention/test_gpt_attention.py",
                    "xqa_generic"),
    LLMUnitTestCase("attention-gpt-trtllm-gen",
                    "{TEST_ROOT}/attention/test_gpt_attention.py",
                    "trtllm_gen"),
    LLMUnitTestCase("attention-gpt-no-cache",
                    "{TEST_ROOT}/attention/test_gpt_attention_no_cache.py"),
    LLMUnitTestCase("attention-gpt-plugin-ib-mode",
                    "{TEST_ROOT}/attention/test_gpt_attention_IFB.py"),
    LLMUnitTestCase("model-llama", "{TEST_ROOT}/model/test_llama.py"),
    LLMUnitTestCase("model-nemotron-nas",
                    "{TEST_ROOT}/model/test_nemotron_nas.py", "not fp8"),
    LLMUnitTestCase("model-mamba", "{TEST_ROOT}/model/test_mamba.py"),
    LLMUnitTestCase("model-mistral", "{TEST_ROOT}/model/test_mistral.py"),
    LLMUnitTestCase("model-deepseek-v2", "{TEST_ROOT}/model/test_deepseek.py"),
    LLMUnitTestCase("model-bloom", "{TEST_ROOT}/model/test_bloom.py"),
    LLMUnitTestCase("model-falcon", "{TEST_ROOT}/model/test_falcon.py"),
    LLMUnitTestCase("model-gpt-partition0", "{TEST_ROOT}/model/test_gpt.py",
                    "partition0"),
    LLMUnitTestCase("model-gpt-partition1", "{TEST_ROOT}/model/test_gpt.py",
                    "partition1"),
    LLMUnitTestCase("model-gpt-partition2", "{TEST_ROOT}/model/test_gpt.py",
                    "partition2"),
    LLMUnitTestCase("model-gpt-partition3", "{TEST_ROOT}/model/test_gpt.py",
                    "partition3"),
    LLMUnitTestCase("model-gpt-other", "{TEST_ROOT}/model/test_gpt.py",
                    "other"),
    LLMUnitTestCase("model-gpt-e2e", "{TEST_ROOT}/model/test_gpt_e2e.py"),
    LLMUnitTestCase("model-gptj", "{TEST_ROOT}/model/test_gptj.py"),
    LLMUnitTestCase("model-gptneox", "{TEST_ROOT}/model/test_gptneox.py"),
    LLMUnitTestCase("quantization", "{TEST_ROOT}/quantization"),
    # Exclude subfolders: modeling, multi_gpu, auto_deploy
    LLMUnitTestCase("_torch", "{TEST_ROOT}/_torch",
                    "not (modeling or multi_gpu or auto_deploy)"),
    LLMUnitTestCase("_torch-modeling-llama", "{TEST_ROOT}/_torch",
                    "modeling_llama"),
    LLMUnitTestCase("_torch-modeling-mixtral", "{TEST_ROOT}/_torch/modeling",
                    "modeling_mixtral"),
    LLMUnitTestCase("_torch-modeling-mllama", "{TEST_ROOT}/_torch/modeling",
                    "modeling_mllama"),
    LLMUnitTestCase("_torch-modeling-nvsmall", "{TEST_ROOT}/_torch/modeling",
                    "modeling_nvsmall"),
    LLMUnitTestCase("_torch-modeling-qwen", "{TEST_ROOT}/_torch/modeling",
                    "modeling_qwen"),
    LLMUnitTestCase("_torch-modeling-nemotron", "{TEST_ROOT}/_torch/modeling",
                    "modeling_nemotron"),
    LLMUnitTestCase("_torch-modeling-out-of-tree",
                    "{TEST_ROOT}/_torch/modeling", "modeling_out_of_tree"),
    LLMUnitTestCase("_torch-modeling-vila", "{TEST_ROOT}/_torch/modeling",
                    "modeling_vila"),
    LLMUnitTestCase("_torch-modeling-bert", "{TEST_ROOT}/_torch/modeling",
                    "modeling_bert"),
    LLMUnitTestCase("_torch-modeling-deepseek",
                    "{TEST_ROOT}/_torch/multi_gpu_modeling",
                    "deepseek and tp1 and nextn0"),
    LLMUnitTestCase("_torch-modeling-deepseek-mtp",
                    "{TEST_ROOT}/_torch/multi_gpu_modeling",
                    "deepseek and tp1 and not nextn0"),
    LLMUnitTestCase("_torch-multi-gpu", "{TEST_ROOT}/_torch/multi_gpu"),
    LLMUnitTestCase("_torch-multi-gpu-deepseek",
                    "{TEST_ROOT}/_torch/multi_gpu_modeling",
                    "deepseek and not tp1 and nextn0"),
    LLMUnitTestCase("_torch-multi-gpu-deepseek-mtp",
                    "{TEST_ROOT}/_torch/multi_gpu_modeling",
                    "deepseek and not tp1 and not nextn0"),
    LLMUnitTestCase("_torch-multi-gpu-llama",
                    "{TEST_ROOT}/_torch/multi_gpu_modeling",
                    "llama and not tp1"),
    LLMUnitTestCase("_torch-auto-deploy-unit-singlegpu",
                    "{TEST_ROOT}/_torch/auto_deploy/unit/singlegpu"),
    LLMUnitTestCase("_torch-auto-deploy-unit-multigpu",
                    "{TEST_ROOT}/_torch/auto_deploy/unit/multigpu"),
    LLMUnitTestCase(
        "_torch-auto-deploy-integration-build",
        "{TEST_ROOT}/_torch/auto_deploy/integration/test_ad_build.py"),
    LLMUnitTestCase(
        "_torch-auto-deploy-integration-eval",
        "{TEST_ROOT}/_torch/auto_deploy/integration/test_lm_eval.py"),
    LLMUnitTestCase("_torch-speculative", "{TEST_ROOT}/_torch/speculative"),
    LLMUnitTestCase("model_api-part1",
                    "{TEST_ROOT}/model_api/test_model_level_api.py"),
    LLMUnitTestCase("model_api-part2",
                    "{TEST_ROOT}/model_api/test_model_quantization.py"),
    LLMUnitTestCase("model_api-part3",
                    "{TEST_ROOT}/model_api/test_model_api_multi_gpu.py"),
    LLMUnitTestCase("llmapi-utils", "{TEST_ROOT}/llmapi/test_llm_utils.py"),
    LLMUnitTestCase("llmapi-build-cache",
                    "{TEST_ROOT}/llmapi/test_build_cache.py"),
    LLMUnitTestCase("llmapi-executor", "{TEST_ROOT}/llmapi/test_executor.py"),
    LLMUnitTestCase("llmapi-perf-evaluator",
                    "{TEST_ROOT}/llmapi/test_llm_perf_evaluator.py"),
    LLMUnitTestCase("llmapi-mpi-session",
                    "{TEST_ROOT}/llmapi/test_mpi_session.py"),
    LLMUnitTestCase("llmapi-quant", "{TEST_ROOT}/llmapi/test_llm_quant.py"),
    LLMUnitTestCase("llmapi-tp-multi-node-2gpu",
                    "{TEST_ROOT}/llmapi/test_llm_multi_node.py"),
    LLMUnitTestCase("llmapi-model-download",
                    "{TEST_ROOT}/llmapi/test_llm_download.py"),
    LLMUnitTestCase("llmapi-models-2gpu",
                    "{TEST_ROOT}/llmapi/test_llm_models_multi_gpu.py"),
    LLMUnitTestCase("model-eagle", "{TEST_ROOT}/model/eagle"),
    LLMUnitTestCase("pip-install-check", "{TEST_ROOT}/test_pip_install.py"),
    LLMUnitTestCase("allreduce_norm_fusion",
                    "{TEST_ROOT}/functional/test_allreduce_norm.py"),
    LLMUnitTestCase(
        "allreduce_prepost_residual_norm_fusion",
        "{TEST_ROOT}/functional/test_allreduceprepost_residual_norm.py"),
    LLMUnitTestCase("fp4_gemm", "{TEST_ROOT}/functional/test_fp4_gemm.py"),
]

# Partition for test_llm_multi_gpu.py
llm_2gpus_tests = [
    [  # part0
        'test_llm_loading_from_ckpt_for_tp2',
        'test_llm_generate_tp2',
        'test_llm_generate_async_tp2',
    ],
    [  # part1
        'test_llm_generate_mixtral_for_tp2',
        'test_llm_generate_mixtral_for_ep2',
    ],
    [  # part2
        'test_llm_pp2',
        'test_llm_end2end_tp2__',
        'test_llm_end2end_tp2__embedding_parallel_mode_NONE_',
        'test_llm_end2end_tp2__embedding_parallel_mode_SHARDING_ALONG_HIDDEN_',
    ],
    [  # part3
        'test_llama_v2_13b_lora_tp2',
        'test_llama_7b_multi_lora_tp2',
        'test_llama_v2_7b_prompt_adapter_tp2',
    ]
]

llm_4gpus_tests = [[  # part0
    'test_tinyllama_guided_decoding_tp2pp2',
    'test_tinyllama_logits_processor_tp2pp2',
]]

for i, tests in enumerate(llm_2gpus_tests):
    TestCases.append(
        LLMUnitTestCase(f"llmapi-2gpu-part{i}",
                        "{TEST_ROOT}/llmapi/test_llm_multi_gpu.py",
                        " or ".join(tests)))

for i, tests in enumerate(llm_4gpus_tests):
    TestCases.append(
        LLMUnitTestCase(f"llmapi-4gpu-part{i}",
                        "{TEST_ROOT}/llmapi/test_llm_multi_gpu.py",
                        " or ".join(tests)))

# others
# There are rough 4 tests
TestCases.append(
    LLMUnitTestCase(
        "llmapi-multigpu-others", "{TEST_ROOT}/llmapi/test_llm_multi_gpu.py",
        "not (" +
        " or ".join(itertools.chain(*llm_2gpus_tests, *llm_4gpus_tests)) + ")"))

# OpenAI API multi-GPU tests
openai_2gpu_part0 = [
    "test_chat_tp2", "test_completion_tp2", "test_chat_streaming_tp2",
    "test_completion_streaming_tp2"
]
TestCases += [
    LLMUnitTestCase("openai-2gpu-part0",
                    "{TEST_ROOT}/llmapi/apps/_test_openai_multi_gpu.py",
                    " or ".join(openai_2gpu_part0))
]

llm_models_part1 = ["gptj", "starcoder", "baichuan"]
llm_models_part1 = " or ".join(llm_models_part1)
llm_models_part2 = ["falcon", "gemma", "gpt2"]
llm_models_part2 = " or ".join(llm_models_part2)
llm_models_part3 = f"not ({llm_models_part1} or {llm_models_part2})"
TestCases += [
    LLMUnitTestCase("llmapi-models-part1",
                    "{TEST_ROOT}/llmapi/test_llm_models.py", llm_models_part1),
    LLMUnitTestCase("llmapi-models-part2",
                    "{TEST_ROOT}/llmapi/test_llm_models.py", llm_models_part2),
    LLMUnitTestCase("llmapi-models-part3",
                    "{TEST_ROOT}/llmapi/test_llm_models.py", llm_models_part3),
]

# partition for test_llm.py
test_llm_single_gpu_part0 = [
    "llm_build_config",
    "llm_args_invalid_usage",
    "llm_loading_from_hf",
    "llm_loading_from_ckpt",
    "llm_with_dummy_weights",
    "llm_with_customized_tokenizer",
    "llm_without_tokenizer",
    "llm_with_kv_cache_retention_config",
    "tokenizer_decode_incrementally",
    "llm_generate_async",
    "user_specify_workspace",
    "generate_with_sampling_params_per_prompt",
    "generate_with_SamplingConfig",
    "generate_with_seed",
    "generate_with_beam_search",
    "generate_with_streaming_llm",
    "parallel_config",
    "generate_with_OutputConfig",
    "generate_with_stop_words",
    "generate_with_bad_words",
    "generate_with_sampling_params_misc",
    "generate_with_embedding_bias",
    "invalid_embedding_bias",
    "generate_with_embedding_bias_fp8",
    "invalid_embedding_bias_fp8",
    "tinyllama_logits_processor",
    "tinyllama_logits_processor_batched",
    "tinyllama_guided_decoding",
    "test_llm_api_medusa",
    "test_llm_api_medusa_tp2",
    "test_llm_api_eagle",
    "test_llm_api_eagle_tp2",
]
test_llm_single_gpu_part0 = " or ".join(test_llm_single_gpu_part0)
test_llm_single_gpu_part1 = f"not ({test_llm_single_gpu_part0})"
TestCases += [
    LLMUnitTestCase("llmapi-single-gpu-part0", "{TEST_ROOT}/llmapi/test_llm.py",
                    test_llm_single_gpu_part0),
    LLMUnitTestCase("llmapi-single-gpu-part1", "{TEST_ROOT}/llmapi/test_llm.py",
                    test_llm_single_gpu_part1),
]

# "others" is by default a placeholder for all tests except the functional/model/attention/quantization listed above
# when you find the "others" case is running slower, causing one job to be too slow, you should find out the
# time consuming cases and listed it separately to load balance the different test jobs
AllOtherCases = LLMUnitTestCase("others",
                                ["--ignore=" + x.filter
                                 for x in TestCases] + ["{TEST_ROOT}"])
TestCases += [AllOtherCases]


def merge_report(base_file, extra_file, output_file, is_retry=False):
    import xml.etree.ElementTree as ElementTree

    base = ElementTree.parse(base_file)
    try:
        extra = ElementTree.parse(extra_file)
    except FileNotFoundError:
        return

    base_suite = base.getroot().find('testsuite')
    extra_suite = extra.getroot().find('testsuite')

    def merge_attr(name, type_=int):
        base_suite.attrib[name] = str(
            type_(base_suite.attrib[name]) + type_(extra_suite.attrib[name]))

    merge_attr("time", type_=float)

    if is_retry:
        base_suite.attrib['failures'] = extra_suite.attrib['failures']
        # pytest may generate testcase node without classname or name attribute when worker crashed catastrophically.
        # Simply ignore these nodes since they are not meaningful.
        extra_suite_nodes = [
            element for element in extra_suite if 'name' in element.attrib
        ]
        case_names = {(element.attrib['classname'], element.attrib['name'])
                      for element in extra_suite_nodes}
        base_suite[:] = [
            element for element in base_suite if 'name' in element.attrib
            if (element.attrib['classname'],
                element.attrib['name']) not in case_names
        ] + extra_suite_nodes
    else:
        merge_attr("errors")
        merge_attr("failures")
        merge_attr("skipped")
        merge_attr("tests")
        base_suite[:] = list(base_suite) + list(extra_suite)

    os.remove(extra_file)
    base.write(output_file, encoding="UTF-8", xml_declaration=True)


@pytest.mark.parametrize("case", TestCases, ids=lambda x: x.name)
def test_unittests(llm_root, llm_venv, case: LLMUnitTestCase, output_dir):
    import pandas as pd
    import pynvml
    pynvml.nvmlInit()

    test_root = os.path.join(llm_root, "tests")
    dry_run = False
    passed = True

    num_workers = 1

    # This dataframe is not manually edited. Infra team will regularly generate this dataframe based on test execution results.
    # If you need to override this policy, please use postprocess code as below.
    agg_unit_mem_df = pd.read_csv(
        f'{test_root}/llm-test-defs/turtle/defs/agg_unit_mem_df.csv')
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    agg_unit_mem_df = agg_unit_mem_df[agg_unit_mem_df['gpu'] == gpu_name]
    print(agg_unit_mem_df)

    parallel_dict = {}
    for _, row in agg_unit_mem_df.iterrows():
        key = (row['gpu'], row['unittest_case_name'])
        parallel_dict[key] = row['parallel_factor']

    print(parallel_dict)
    cur_key = (gpu_name, case.name)
    if cur_key in parallel_dict:
        num_workers = parallel_dict[cur_key]
        num_workers = min(num_workers, 8)
    else:
        print(
            f'unittest {case.name} on "{gpu_name}" is not recorded in parallel config. Need to profile.'
        )

    num_workers = max(1, num_workers)

    if parallel_override := os.environ.get("LLM_TEST_PARALLEL_OVERRIDE", None):
        num_workers = int(parallel_override)

    print('Parallel workers: ', num_workers)

    ignore_opt = f"--ignore={test_root}/llm-test-defs"

    output_xml = os.path.join(output_dir,
                              f'sub-results-unittests-{case.name}.xml')

    command = [
        '-m',
        'pytest',
        ignore_opt,
        "-v",
        "--timeout=1600",
    ]

    if dry_run:
        command += ['--collect-only']

    filter_args = case.filter
    if not isinstance(filter_args, list):
        filter_args = [filter_args]
    filter_args = [arg.format(TEST_ROOT=test_root) for arg in filter_args]
    if case.test_case_filter:
        filter_args += ["-k", case.test_case_filter]

    command += filter_args

    print(f"Running:{case.name}, cmd:'{command}'")

    def run_command(cmd):
        try:
            llm_venv.run_cmd(cmd)
        except CalledProcessError:
            return False
        return True

    if num_workers == 1:
        # Do not bother with pytest-xdist at all if we don't need parallel execution
        command += ["-p", "no:xdist", f"--junitxml={output_xml}"]
        passed = run_command(command)
    else:
        # Avoid .xml extension to prevent CI from reading failures from it
        parallel_output_xml = os.path.join(
            output_dir,
            f'parallel-sub-results-unittests-{case.name}.xml.intermediate')
        parallel_command = command + [
            "-n", f"{num_workers}", '--reruns', '3',
            f"--junitxml={parallel_output_xml}"
        ]
        passed = run_command(parallel_command)

        assert os.path.exists(
            parallel_output_xml
        ), "no report generated, fatal failure happened in unittests (parallel phase)"

        if dry_run or passed:
            os.rename(parallel_output_xml, output_xml)
        else:
            # Avoid .xml extension to prevent CI from reading failures from it
            retry_output_xml = os.path.join(
                output_dir,
                f'retry-sub-results-unittests-{case.name}.xml.intermediate')
            # Run failed case sequentially.
            command = [
                '-m', 'pytest', "-p", "no:xdist", ignore_opt, "-v", '--lf',
                f"--junitxml={retry_output_xml}"
            ] + filter_args
            passed = run_command(command)

            if os.path.exists(retry_output_xml):
                merge_report(parallel_output_xml, retry_output_xml, output_xml,
                             True)
            else:
                os.rename(parallel_output_xml, output_xml)
                assert False, "no report generated, fatal failure happened in unittests (retry phase)"

    assert passed, "failure reported in unittests"
