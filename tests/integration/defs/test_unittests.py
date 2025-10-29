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
import os
import warnings
from subprocess import CalledProcessError

from defs.conftest import tests_path


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


def test_unittests_v2(llm_root, llm_venv, case: str, output_dir, request):
    import pandas as pd
    import pynvml
    pynvml.nvmlInit()

    test_root = tests_path()
    dry_run = False

    my_test_prefix = request.config.getoption("--test-prefix")
    if my_test_prefix:
        test_prefix = f"{my_test_prefix}/unittest"
    else:
        test_prefix = "unittest"

    waives_file = request.config.getoption("--waives-file")

    num_workers = 1

    # This dataframe is not manually edited. Infra team will regularly generate this dataframe based on test execution results.
    # If you need to override this policy, please use postprocess code as below.
    agg_unit_mem_df = pd.read_csv(
        f'{test_root}/integration/defs/agg_unit_mem_df.csv')
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

    cur_key = (gpu_name, case)
    if cur_key in parallel_dict:
        num_workers = parallel_dict[cur_key]
        num_workers = min(num_workers, 8)
    else:
        warnings.warn(
            f'Cannot find parallel config entry for unittest {case} on "{gpu_name}". Fallback to serial test. Please add config entry to agg_unit_mem_df.csv.'
        )

    num_workers = max(1, num_workers)

    if parallel_override := os.environ.get("LLM_TEST_PARALLEL_OVERRIDE", None):
        num_workers = int(parallel_override)

    print('Parallel workers: ', num_workers)

    ignore_opt = f"--ignore={test_root}/integration"

    import shlex
    arg_list = shlex.split(case)
    case_fn = case.replace('/', '-')
    if len(case_fn) > 80:
        case_fn = case_fn[:80]
    output_xml = os.path.join(output_dir,
                              f'results-sub-unittests-{case_fn}.xml')

    command = [
        '-m', 'pytest', ignore_opt, "-v", "--timeout=2400",
        "--timeout-method=thread"
    ]
    if test_prefix:
        command += [f"--test-prefix={test_prefix}"]

    if dry_run:
        command += ['--collect-only']

    if waives_file:
        waives_file = os.path.abspath(waives_file)
        command += [f"--waives-file={waives_file}"]

    command += arg_list

    print(f"Running unit test:\"python {' '.join(command)}\"")

    def run_command(cmd, num_workers=1):
        try:
            pythonpath = os.environ.get("PYTHONPATH", "")
            env = {'PYTHONPATH': f"{llm_root}/tests/unittest:{pythonpath}"}
            if num_workers > 1:
                env['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
            llm_venv.run_cmd(
                cmd,
                cwd=test_root,
                env=env,
            )
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
            f'parallel-sub-results-unittests-{case_fn}.xml.intermediate')
        parallel_command = command + [
            "-n", f"{num_workers}", f"--junitxml={parallel_output_xml}"
        ]
        passed = run_command(parallel_command, num_workers)

        assert os.path.exists(
            parallel_output_xml
        ), "no report generated, fatal failure happened in unittests (parallel phase)"

        if dry_run or passed:
            os.rename(parallel_output_xml, output_xml)
        else:
            # Avoid .xml extension to prevent CI from reading failures from it
            retry_output_xml = os.path.join(
                output_dir,
                f'retry-sub-results-unittests-{case_fn}.xml.intermediate')
            # Run failed case sequentially.
            retry_command = command + [
                "-p", "no:xdist", '--lf', f"--junitxml={retry_output_xml}"
            ]
            passed = run_command(retry_command)

            if os.path.exists(retry_output_xml):
                merge_report(parallel_output_xml, retry_output_xml, output_xml,
                             True)
            else:
                os.rename(parallel_output_xml, output_xml)
                assert False, "no report generated, fatal failure happened in unittests (retry phase)"

    assert passed, "failure reported in unittests"
