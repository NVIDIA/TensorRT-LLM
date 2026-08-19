# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
import shlex
import warnings
from subprocess import CalledProcessError
from typing import NoReturn

from defs.conftest import tests_path

_MAX_CULPRITS = 20


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


def _junit_culprits(xml_path: str) -> list[str]:
    """Names of failed/errored tests in a JUnit report.

    So a whole-case failure still names the specific culprit test(s) for
    CI/triage attribution.
    """
    import xml.etree.ElementTree as ElementTree
    try:
        root = ElementTree.parse(xml_path).getroot()
    except (OSError, ElementTree.ParseError):
        return []
    culprits = []
    for tc in root.iter('testcase'):
        if not tc.get('name'):
            continue
        kind = 'FAILED' if tc.find('failure') is not None else (
            'ERROR' if tc.find('error') is not None else None)
        if kind:
            culprits.append(
                f"{kind} {tc.get('classname', '')}::{tc.get('name')}")
    return culprits


def _nodeid_matches_case(nodeid: str, case_selector: str) -> bool:
    """Whether a possibly stage-prefixed node ID belongs to a case selector."""
    node_path, node_separator, node_selection = nodeid.partition("::")
    case_path, case_separator, case_selection = case_selector.partition("::")

    if case_separator:
        path_matches = (node_path == case_path
                        or node_path.endswith("/" + case_path))
    else:
        path_matches = f"/{case_path}/" in f"/{node_path}/"
    if not path_matches:
        return False

    if not case_separator:
        return True
    if not node_separator:
        return False
    if node_selection == case_selection:
        return True
    if "[" in case_selection:
        return False
    return (node_selection.startswith(case_selection + "::")
            or node_selection.startswith(case_selection + "["))


def _fail_unittests(reason: str, output_xml: str, output_dir: str,
                    case: str) -> NoReturn:
    """Raise a unittest failure naming the specific culprit test(s).

    Culprits are failed/errored tests from the JUnit report, plus tests from
    this case that were still in flight when a fatal OOM/timeout killed the run
    (nodeids left by --periodic-save-unfinished-test).

    unfinished_test.txt is shared per node and only cleared on teardown, so it
    may retain stale entries from other cases; it is filtered to this case's
    test path(s). Reading it is best-effort and never masks the original
    failure.
    """
    junit_culprits = _junit_culprits(output_xml)
    in_flight_culprits = []
    unfinished = os.path.join(output_dir, "unfinished_test.txt")
    case_selectors = []
    for arg in shlex.split(case):
        selector = arg.rstrip("/")
        path = selector.split("::", 1)[0]
        if "/" in path or path.endswith(".py"):
            case_selectors.append(selector)
    try:
        with open(unfinished, encoding="utf-8") as f:
            for line in f:
                nodeid = line.strip()
                if not nodeid:
                    continue
                if not case_selectors or any(
                        _nodeid_matches_case(nodeid, selector)
                        for selector in case_selectors):
                    in_flight_culprits.append("IN-FLIGHT " + nodeid)
    except (OSError, UnicodeDecodeError):
        pass
    # A fatal retry's in-flight node is the strongest signal. Put it ahead of
    # the earlier JUnit failures so the display cap cannot hide it, and remove
    # duplicate unfinished records left by an interrupted cleanup.
    culprits = list(dict.fromkeys(in_flight_culprits + junit_culprits))
    if culprits:
        displayed_culprits = ", ".join(culprits[:_MAX_CULPRITS])
        omitted_culprits = len(culprits) - _MAX_CULPRITS
        if omitted_culprits > 0:
            displayed_culprits += f" (+{omitted_culprits} more)"
        reason += "; culprit test(s): " + displayed_culprits
    raise AssertionError(reason)


def test_unittests_v2(llm_root, llm_venv, case: str, output_dir, request):
    import pandas as pd
    import pynvml
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
    except pynvml.NVMLError:
        gpu_name = "CPU"

    test_root = tests_path()
    dry_run = False

    my_test_prefix = request.config.getoption("--test-prefix")
    if my_test_prefix:
        test_prefix = f"{my_test_prefix}/unittest"
    else:
        test_prefix = "unittest"

    waives_file = request.config.getoption("--waives-file")
    run_ray = request.config.getoption("--run-ray")
    unittest_markexpr = request.config.getoption("--unittest-markexpr")

    num_workers = 1

    # This dataframe is not manually edited. Infra team will regularly generate this dataframe based on test execution results.
    # If you need to override this policy, please use postprocess code as below.
    agg_unit_mem_path = f'{test_root}/integration/defs/agg_unit_mem_df.csv'
    print(f'Loading unittest parallel config from: {agg_unit_mem_path}')
    agg_unit_mem_df = pd.read_csv(agg_unit_mem_path)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    print(f'GPU name from NVML (index 0): {gpu_name!r}')
    print(f'GPU names available in parallel config: '
          f'{sorted(agg_unit_mem_df["gpu"].dropna().unique().tolist())}')
    agg_unit_mem_df = agg_unit_mem_df[agg_unit_mem_df['gpu'] == gpu_name]
    print(f'Matched parallel config rows for GPU {gpu_name!r}: '
          f'{len(agg_unit_mem_df)}')
    print(agg_unit_mem_df)

    parallel_dict = {}
    for _, row in agg_unit_mem_df.iterrows():
        key = (row['gpu'], row['unittest_case_name'])
        parallel_dict[key] = row['parallel_factor']

    print(parallel_dict)

    cur_key = (gpu_name, case)
    print(f'Parallel config lookup key: {cur_key!r}')
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
    if unittest_markexpr:
        if "-m" in arg_list:
            markexpr_index = arg_list.index("-m") + 1
            case_markexpr = arg_list[markexpr_index]
            arg_list[markexpr_index] = (
                f"({case_markexpr}) and ({unittest_markexpr})")
        else:
            arg_list += ["-m", unittest_markexpr]
    case_fn = re.sub(r'[/\s"\']+', '-', case)
    if len(case_fn) > 80:
        case_fn = case_fn[:80]
    # MoE entries expand to too many sub-tests, which introduces noise into
    # the overall TRT-LLM test quality metrics. Skip per-sub-test reporting
    # for MoE by prefixing with "moe-" (CI only collects files starting
    # with "results" for JUnit reporting).
    if case.startswith("unittest/_torch/modules/moe/"):
        output_xml = os.path.join(output_dir,
                                  f'moe-results-sub-unittests-{case_fn}.xml')
    else:
        output_xml = os.path.join(output_dir,
                                  f'results-sub-unittests-{case_fn}.xml')

    command = [
        '-m',
        'pytest',
        ignore_opt,
        "-vv",
        "--tb=short",
        "-rF",
        "--timeout=2400",
        "--timeout-method=thread",
        "--periodic-junit",
        "--periodic-batch-size=1",
        "--periodic-save-unfinished-test",
    ]
    if test_prefix:
        command += [f"--test-prefix={test_prefix}"]

    if dry_run:
        command += ['--collect-only']

    if waives_file:
        waives_file = os.path.abspath(waives_file)
        command += [f"--waives-file={waives_file}"]

    if run_ray:
        command += ["--run-ray"]

    s3_secret_key = None
    s3_output_module = None
    s3_upload_path = request.config.getoption("--s3-upload-path", default=None)
    if s3_upload_path:
        from test_common import s3_output as s3_output_module

        inner_output_dir = os.path.join(output_dir, "inner-s3", case_fn)
        inner_upload_path = os.path.join(s3_upload_path, "inner", case_fn)
        command += [
            "--capture=fd",
            f"--output-dir={inner_output_dir}",
            f"--s3-upload-path={inner_upload_path}",
            "--s3-upload-mode=deferred",
        ]
        inline_output_max_bytes = request.config.getoption(
            "--s3-inline-output-max-bytes", default=256)
        command += [f"--s3-inline-output-max-bytes={inline_output_max_bytes}"]
        for option in ("--s3-endpoint", "--s3-username", "--s3-bucket"):
            value = request.config.getoption(option, default=None)
            if value:
                command += [f"{option}={value}"]
        if request.config.getoption("--s3-skip-upload", default=False):
            command += ["--s3-skip-upload"]
        s3_secret_key = request.config.getoption("--s3-secret-key",
                                                 default=None)

    command += arg_list

    print(f"Running unit test:\"python {' '.join(command)}\"")

    def build_pythonpath():
        pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [f"{llm_root}/tests/unittest"]

        # MPI worker processes start from the tests directory and otherwise
        # may resolve an older site-packages triton_kernels package.
        triton_kernels_src = os.path.join(llm_root, "triton_kernels")
        if os.path.isdir(triton_kernels_src):
            pythonpath_overrides_dir = os.path.join(
                output_dir, f"pythonpath-overrides-{case_fn}")
            os.makedirs(pythonpath_overrides_dir, exist_ok=True)
            triton_kernels_link = os.path.join(pythonpath_overrides_dir,
                                               "triton_kernels")
            if os.path.lexists(triton_kernels_link):
                is_expected_link = False
                if os.path.islink(triton_kernels_link):
                    is_expected_link = (
                        os.readlink(triton_kernels_link) == triton_kernels_src)
                if not is_expected_link:
                    raise RuntimeError(
                        f"Unexpected triton_kernels path: {triton_kernels_link}"
                    )
            else:
                os.symlink(triton_kernels_src, triton_kernels_link)
            pythonpath_entries.insert(0, pythonpath_overrides_dir)

        if pythonpath:
            pythonpath_entries.append(pythonpath)
        return os.pathsep.join(pythonpath_entries)

    def run_command(cmd, num_workers=1):
        try:
            env = {'PYTHONPATH': build_pythonpath()}
            if s3_secret_key:
                env["S3_SECRET_KEY"] = s3_secret_key
            if num_workers > 1:
                env['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
            llm_venv.run_cmd(
                cmd,
                cwd=test_root,
                env=env,
            )
        except CalledProcessError as e:
            print(f"\n{'='*60}")
            print(f"UNITTEST FAILED with exit code: {e.returncode}")
            print(f"Command: {' '.join(cmd)}")
            if hasattr(e, 'stdout') and e.stdout:
                print(
                    f"STDOUT:\n{e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout}"
                )
            if hasattr(e, 'stderr') and e.stderr:
                print(
                    f"STDERR:\n{e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr}"
                )
            print(f"{'='*60}\n")
            return False
        finally:
            if s3_output_module is not None:
                try:
                    drained = s3_output_module.drain_pending_uploads(
                        inner_output_dir,
                        secret_key=s3_secret_key,
                    )
                except Exception as e:
                    print(f"Failed to drain pending S3 test logs: {e}")
                else:
                    if not drained:
                        print("Some pending S3 test logs could not be drained")
        return True

    if num_workers == 1:
        # Do not bother with pytest-xdist at all if we don't need parallel execution
        command += ["-p", "no:xdist", f"--periodic-junit-xmlpath={output_xml}"]
        passed = run_command(command)
    else:
        # Avoid .xml extension to prevent CI from reading failures from it
        parallel_output_xml = os.path.join(
            output_dir,
            f'parallel-sub-results-unittests-{case_fn}.xml.intermediate')
        parallel_command = command + [
            "-n", f"{num_workers}",
            f"--periodic-junit-xmlpath={parallel_output_xml}"
        ]
        passed = run_command(parallel_command, num_workers)

        if not os.path.exists(parallel_output_xml):
            _fail_unittests(
                "no report generated, fatal failure in unittests (parallel phase)",
                parallel_output_xml, output_dir, case)

        if dry_run or passed:
            os.rename(parallel_output_xml, output_xml)
        else:
            # Avoid .xml extension to prevent CI from reading failures from it
            retry_output_xml = os.path.join(
                output_dir,
                f'retry-sub-results-unittests-{case_fn}.xml.intermediate')
            # Run failed case sequentially.
            retry_command = command + [
                "-p", "no:xdist", '--lf',
                f"--periodic-junit-xmlpath={retry_output_xml}"
            ]
            passed = run_command(retry_command)

            if os.path.exists(retry_output_xml):
                merge_report(parallel_output_xml, retry_output_xml, output_xml,
                             True)
            else:
                os.rename(parallel_output_xml, output_xml)
                _fail_unittests(
                    "no report generated, fatal failure in unittests (retry phase)",
                    output_xml, output_dir, case)

    if not passed:
        _fail_unittests("failure reported in unittests", output_xml, output_dir,
                        case)
