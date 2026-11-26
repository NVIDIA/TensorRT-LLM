/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import com.cloudbees.groovy.cps.Continuable
import com.cloudbees.groovy.cps.CpsTransformer
import org.codehaus.groovy.control.CompilerConfiguration

// Run from the repository root; see README.md for the test-only classpath.
// An optional source path lets the same assertions reproduce the pre-fix failure.
def pipelineFile = new File(args ? args[0] : 'jenkins/L0_Test.groovy')
def source = pipelineFile.text
def start = source.indexOf('// Explicit stage selection retains the original pytest-split assignment.')
def end = source.indexOf('// True when an exception indicates the K8s dispatcher pod', start)
assert start >= 0 && end > start
def methods = source.substring(start, end)
def compiler = new CompilerConfiguration()
compiler.addCompilationCustomizers(new CpsTransformer())
def shell = new GroovyShell(this.class.classLoader, new Binding(), compiler)
def script = shell.parse('''
import com.cloudbees.groovy.cps.NonCPS
import groovy.transform.Field
''' + methods + '''
CBTS_RESULT = 'cbts_result'
CBTS_STAGE_SUFFIX = '-cbts'
TEST_STAGE_LIST = 'stage_list'
EXTRA_STAGE_LIST = 'extra_stage_list'
AUTO_TRIGGER_TAG_LIST = 'auto_trigger_tag_list'

def fixture(Map options = [:]) {
    cbtsStageConfigs.clear()
    def counts = [:]
    // Deliberately unsorted, with a double-digit group and a gap in selected scope.
    [10, 3, 1, 2].each { counts["A10-PyTorch-${it}".toString()] = options.splits ?: 2 }
    counts['H100-PyTorch-2'] = 1
    testFilter = [(CBTS_RESULT): [
        scope: 'testsonly', cbts_test_db_artifact_path: 'cbts.tar.gz',
        affected_stage_split_counts: counts, affected_stages: counts.keySet().toList(),
        sanity_required: true, perfsanity_required: false, enable_multi_gpu: false,
    ]] + (options.flags ?: [:])
    def configs = [:]
    [10, 3, 1, 2].each {
        configs["A10-PyTorch-${it}".toString()] = ['a10', 'l0_a10', it, 10]
    }
    configs['H100-PyTorch-2'] = ['h100', 'l0_h100', 2, 2]
    configs = cbtsResizeSplits(configs)
    def jobs = configs.collectEntries { name, values ->
        // Like launchTestJobs, the stage closure captures the shared config tuple.
        [(name): { -> [values[2], values[3]] }]
    }
    jobs['PackageSanityCheck'] = { -> 'sanity' }
    jobs['Unrelated-PyTorch-1'] = { -> 'unrelated' }
    if (options.partial) {
        jobs.remove('A10-PyTorch-1-cbts')
        jobs.remove('A10-PyTorch-2-cbts')
    }
    return jobs
}

def jobs = fixture()
def filtered = filterCbtsStageJobs(jobs, testFilter)
assert filtered.keySet() == [
    'A10-PyTorch-1-cbts', 'A10-PyTorch-2-cbts', 'H100-PyTorch-2-cbts',
    'PackageSanityCheck',
] as Set
assert filtered['A10-PyTorch-1-cbts']() == [1, 2]
assert filtered['A10-PyTorch-2-cbts']() == [2, 2]
assert filtered['H100-PyTorch-2-cbts']() == [1, 1]

// No lexicographic sorting: group 10 must follow group 3. Also clamp the split
// count to available jobs and never restore groups excluded by command scope.
jobs = fixture(splits: 10, partial: true)
filtered = filterCbtsStageJobs(jobs, testFilter)
assert filtered['A10-PyTorch-3-cbts']() == [1, 2]
assert filtered['A10-PyTorch-10-cbts']() == [2, 2]
assert !filtered.containsKey('A10-PyTorch-1-cbts')
assert !filtered.containsKey('A10-PyTorch-2-cbts')

// Both explicit selectors bypass rebalancing and retain original shard IDs.
[TEST_STAGE_LIST, EXTRA_STAGE_LIST].each { flag ->
    jobs = fixture(flags: [(flag): ['A10-PyTorch-*']])
    filtered = filterCbtsStageJobs(jobs, testFilter)
    [10, 3, 1, 2].each { group ->
        assert filtered["A10-PyTorch-${group}-cbts".toString()]() == [group, 10]
    }
    assert filtered['H100-PyTorch-2-cbts']() == [2, 2]
}

// Selector fallback and decisions without case narrowing never enter sorting.
jobs = fixture()
testFilter[CBTS_RESULT] = null
assert filterCbtsStageJobs(jobs, testFilter).is(jobs)
cbtsStageConfigs.clear()
testFilter[CBTS_RESULT] = [affected_stages: ['PackageSanityCheck']]
assert filterCbtsStageJobs(jobs, testFilter).keySet() == ['PackageSanityCheck'] as Set
testFilter[CBTS_RESULT] = [affected_stages: []]
assert filterCbtsStageJobs(jobs, testFilter).isEmpty()
return 'CBTS stage sharding passed under Groovy CPS'

def echo(message) { }
''')
println new Continuable(script).run(null)
