#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _method(source: str, name: str) -> str:
    start = source.index(f"def {name}(")
    # Pipeline methods close at column zero; nested blocks are indented.
    end = source.index("\n}", start) + 2
    return source[start:end]


@pytest.fixture(scope="module")
def groovy_runner() -> list[str]:
    if executable := shutil.which("groovy"):
        return [executable]
    if classpath := os.environ.get("GROOVY_CLASSPATH"):
        return ["java", "-cp", classpath, "groovy.ui.GroovyMain"]
    pytest.skip("Install Groovy or set GROOVY_CLASSPATH to run pipeline behavior tests")


@pytest.mark.parametrize(
    "scenario",
    [
        pytest.param(
            """
def scope = ['H100-PyTorch-1': 1]
assert filterCbtsStageJobs(scope, [:]).is(scope)
assert filterCbtsStageJobs(scope, [cbts_result: null]).is(scope)
""",
            id="fallback_scope",
        ),
        pytest.param(
            """
def scope = ['A10-PyTorch-1': 1, 'A10-CPP-1': 2]
assert filterCbtsStageJobs(scope, [cbts_result: decision]) == ['A10-PyTorch-1': 1]
""",
            id="intersection",
        ),
        pytest.param(
            """
decision.sanity_required = true
decision.perfsanity_required = true
assert filterCbtsStageJobs(['H100-PyTorch-1': 1], [cbts_result: decision]).isEmpty()
""",
            id="sanity_scope",
        ),
        pytest.param(
            """
decision.sanity_required = true
decision.perfsanity_required = true
def scope = ['PackageSanityCheck': 1, 'A10-PerfSanity-1': 2,
             'A10-PerfSanity-Post-Merge-1': 3]
assert filterCbtsStageJobs(scope, [cbts_result: decision]).keySet() ==
    ['PackageSanityCheck', 'A10-PerfSanity-1'] as Set
""",
            id="sanity_selection",
        ),
        pytest.param(
            """
decision.enable_multi_gpu = true
def scope = ['DGX-4_GPUs-PyTorch-1': 1]
assert filterCbtsStageJobs(scope, [cbts_result: decision]) == scope
assert filterCbtsStageJobs([:], [cbts_result: decision]).isEmpty()
""",
            id="multi_gpu_scope",
        ),
        pytest.param(
            """
decision.affected_stages = ['A10-PyTorch-OnDemand-1', 'H100-PyTorch-Post-Merge-1']
def scope = ['A10-PyTorch-OnDemand-1': 1, 'H100-PyTorch-Post-Merge-1': 2]
assert filterCbtsStageJobs(scope, [cbts_result: decision]) == scope
""",
            id="explicit_special_stages",
        ),
        pytest.param(
            """
decision.affected_stages = ['A10-PyTorch-1', 'A10-PyTorch-6']
decision.affected_stage_split_counts = ['A10-PyTorch-1': 1, 'A10-PyTorch-6': 1]
decision.cbts_test_db_artifact_path = 'artifact'
testFilter.cbts_result = decision
def configs = cbtsResizeSplits(['A10-PyTorch-1': ['gpu', 'yaml', 1, 6],
                              'A10-PyTorch-6': ['gpu', 'yaml', 6, 6]])
def selected = configs.findAll { stageMatchesAnyPattern(it.key, ['A10-PyTorch-6']) }
assert selected.keySet() == ['A10-PyTorch-6-cbts'] as Set
def kept = filterCbtsStageJobs(selected, testFilter)
assert kept.keySet() == ['A10-PyTorch-6-cbts'] as Set
assert kept['A10-PyTorch-6-cbts'][2..3] == [1, 1]
assert configs['A10-PyTorch-6-cbts'].is(cbtsStageConfigs['A10-PyTorch-6-cbts'])
assert stageMatchesAnyPattern('A10-PyTorch-6-cbts', ['A10-*-6'])
checkStageNameSet(['A10-PyTorch-6'], ['A10-PyTorch-1', 'A10-PyTorch-6'], 'stage_list')
""",
            id="single_late_shard",
        ),
        pytest.param(
            """
decision.affected_stages = (1..6).collect { 'A10-PyTorch-' + it }
decision.affected_stage_split_counts = decision.affected_stages.collectEntries { [(it): 2] }
decision.cbts_test_db_artifact_path = 'artifact'
testFilter.cbts_result = decision
def configs = cbtsResizeSplits((1..6).collectEntries {
    [('A10-PyTorch-' + it): ['gpu', 'yaml', it, 6]]
})
def kept = filterCbtsStageJobs(configs, testFilter)
assert kept.keySet() == ['A10-PyTorch-1-cbts', 'A10-PyTorch-2-cbts'] as Set
assert kept.values().collect { it[2..3] } == [[1, 2], [2, 2]]
""",
            id="resize_all_shards",
        ),
        pytest.param(
            """
decision.affected_stage_split_counts = ['A10-PyTorch-1': 1]
testFilter.cbts_result = decision
def configs = ['A10-PyTorch-1': ['gpu', 'yaml', 1, 6]]
assert cbtsResizeSplits(configs).is(configs)
assert filterCbtsStageJobs(configs, testFilter) == configs
""",
            id="artifact_upload_fallback",
        ),
        pytest.param(
            """
def scope = ['H100-PyTorch-VisualGen-1': 1, 'A10-PyTorch-1': 2]
def filter = [cbts_result: decision, auto_trigger_tag_list: ['VisualGen']]
assert filterCbtsStageJobs(scope, filter) == scope
assert filterCbtsStageJobs(['A10-PyTorch-1': 2], filter) == ['A10-PyTorch-1': 2]
""",
            id="auto_trigger_within_scope",
        ),
        pytest.param(
            """
changedFiles = ['tensorrt_llm/example.py']
def flags = [test_stage_list: ['A10-*'], extra_stage_list: ['*Post-Merge*'],
             gpu_type_list: ['A10'], test_backend: ['pytorch'], enable_skip_test: true,
             add_multi_gpu_test: true, only_multi_gpu_test: true,
             disable_multi_gpu_test: true]
def messages = []
def pipeline = [echo: { messages << it }]
flags.each { key, value ->
    messages.clear()
    def result = getCbtsResult(pipeline, [(key): value], [:])
    assert result?.scope == 'test_list' : messages
    assert new groovy.json.JsonSlurper().parseText(inputText).post_merge == true
    assert !messages.any { it.contains('deferring') }
}
""",
            id="flags_activate_cbts",
        ),
        pytest.param(
            """
changedFiles = ['tensorrt_llm/example.py']
failSelector = true
def messages = []
def pipeline = [echo: { messages << it }]
def fallback = getCbtsResult(pipeline, [test_stage_list: ['A10-*']], [:])
assert fallback == null
assert messages.last().contains('falling back to command-selected stages')
def scope = ['A10-PyTorch-6': 1]
assert filterCbtsStageJobs(scope, [cbts_result: fallback]).is(scope)
""",
            id="selector_error_scope",
        ),
    ],
)
def test_cbts_command_scope(groovy_runner: list[str], tmp_path: Path, scenario: str) -> None:
    test_source = (ROOT / "jenkins/L0_Test.groovy").read_text()
    parent_source = (ROOT / "jenkins/L0_MergeRequest.groovy").read_text()
    methods = [
        _method(test_source, name)
        for name in (
            "cbtsResizeSplits",
            "filterCbtsStageJobs",
            "stageMatchesPattern",
            "stageMatchesAnyPattern",
            "checkStageNameSet",
        )
    ]
    methods.extend(
        _method(parent_source, name)
        for name in (
            "getCbtsResult",
            "_cbtsParseSelectionResult",
            "_cbtsMatchesAnyPattern",
            "_cbtsGlobToRegex",
        )
    )
    script = tmp_path / "scope.groovy"
    script.write_text(
        """import groovy.transform.Field
@Field def CBTS_RESULT = 'cbts_result'
@Field def CBTS_STAGE_SUFFIX = '-cbts'
@Field def DISABLE_CBTS = 'disable_cbts'
@Field def testFilter = [:]
@Field def cbtsStageConfigs = [:]
@Field def env = [JOB_NAME: 'L0_MergeRequest']
@Field def changedFiles = []
@Field def inputText = ''
@Field def failSelector = false
@Field def LLM_ROOT = '/workspace'
@Field def IS_POST_MERGE = 'is_post_merge'
@Field def MULTI_GPU_FILE_CHANGED = 'multi_gpu_file_changed'
@Field def AUTO_TRIGGER_TAG_LIST = 'auto_trigger_tag_list'
def echo(message) {}
def getMergeRequestChangedFileList(pipeline, globalVars) { changedFiles }
def _cbtsEnsureCliDependencies(pipeline) { true }
def sh(String command) {}
def sh(Map args) {
    if (failSelector) throw new IOException('selector failed')
    if (args.script.contains('--list-needed-diffs')) return ''
    return '{"scope":"test_list","affected_stages":["A10-PyTorch-1"]}'
}
def writeFile(Map args) { inputText = args.text }
def _cbtsReportDecision(Object... args) {}
"""
        + "\n".join(methods)
        + """
def decision = [scope: 'coverage', affected_stages: ['A10-PyTorch-1'],
    sanity_required: false, perfsanity_required: false, enable_multi_gpu: false]
"""
        + scenario
    )
    result = subprocess.run(
        [*groovy_runner, str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
