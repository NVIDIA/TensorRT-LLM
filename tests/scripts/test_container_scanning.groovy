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

// Run with Groovy 2.4+ and the path to jenkins/L0_MergeRequest.groovy.
// Exercises the production Docker branch and launchJob with mocked Jenkins steps.
class Logger { Logger(Object pipeline) {} }
class JobBuilder {
    static Closure handler
    static build(Object... args) { handler.call() }
}
class TestFlowInterruptedException extends InterruptedException {}

def source = new File(args[0]).text
def launchStart = source.indexOf('def launchJob(')
def launchEnd = source.indexOf('\ndef launchInfraDryRunTestJob(', launchStart)
def branchStart = source.indexOf('    def dockerBuildJob = [')
def branchEnd = source.indexOf('\n    if (runMode == "nightly_release")', branchStart)
assert [launchStart, launchEnd, branchStart, branchEnd].every { it >= 0 }
def production = source.substring(launchStart, launchEnd) + '\n' +
    source.substring(branchStart, branchEnd) + '\ndockerBuildJob["Build-Docker-Images"]()'
def loader = new GroovyClassLoader(this.class.classLoader)
def parsed = new GroovyShell(loader).parse(production).class
int count = 0

def runCase = { String mode, boolean failFast, boolean scan, String outcome,
                boolean ignore = false, String scanOutcome = 'SUCCESS' ->
    def state = [result: 'SUCCESS', scans: 0, stages: [:], stack: [], thrown: null]
    def rank = ['SUCCESS': 0, 'UNSTABLE': 1, 'FAILURE': 2, 'ABORTED': 3]
    def worsen = { String result ->
        if (rank[result] > rank[state.result]) state.result = result
    }
    def binding = new Binding([
        runMode: mode, enableFailFast: failFast, runContainerScanning: scan,
        globalVars: [branch: 'main'], BUILD_BRANCH: 'branch',
        LLM_ARTIFACTS_PRIMARY_KEY: 'primary', CACHED_CHANGED_FILE_LIST: 'changed',
        ARTIFACT_PATH: 'artifacts/209', wheelSelected: true, buildAllDockerImages: true,
        BUILD_CHECK_CHOICE: ignore ? 'ignore' : 'normal', STAGE_CHOICE_IGNORE: 'ignore',
        pipeline: new Object(), params: [:], currentBuild: state,
        env: [JOB_NAME: mode == 'post_merge' ? 'LLM/main/L0_PostMerge' : 'LLM/main/L0_NightlyRelease',
              BUILD_NUMBER: '209', gitlabCommit: 'abcdef0123456789', NSPECT_RELEASE_VERSION: 'nightly'],
        script: { Closure body -> body() },
        echo: { Object message -> },
        error: { Object message -> throw new IllegalStateException(message.toString()) },
        string: { Map param -> param },
        getCommonParameters: { [:] },
        llmArtifactsPrimary: { 'artifacts' },
        writeJSON: { Map options -> '{}' },
        getPLCRepoUrlKey: { 'tensorrt_llm_github' },
    ])
    binding.stage = { String name, Closure body ->
        state.stack << name
        try { body() } finally { state.stack.remove(state.stack.size() - 1) }
    }
    binding.catchError = { Map options, Closure body ->
        try { body() } catch (Exception e) {
            if (e instanceof InterruptedException && options.catchInterruptions == false) throw e
            worsen(options.buildResult ?: 'FAILURE')
            state.stages[state.stack.last()] = options.stageResult ?: 'SUCCESS'
        }
    }
    binding.build = { Map options ->
        assert options.job == '/LLM/helpers/PLCScanningSetup'
        assert options.propagate == false
        def parameters = options.parameters.collectEntries { [(it.name): it.value] }
        assert parameters.postMergePipelineName == binding.env.JOB_NAME
        assert parameters.postMergeBuildNumber == '209'
        assert parameters.ref == 'abcdef0123456789'
        assert parameters.scanMode == (mode == 'nightly_release' ? 'release' : 'pre_merge')
        assert parameters.runSourceCodeScanning == 'false'
        assert parameters.runContainerOssScanning == 'true'
        state.scans++
        if (scanOutcome == 'INTERRUPTED') throw new TestFlowInterruptedException()
        [result: scanOutcome]
    }
    JobBuilder.handler = {
        if (outcome == 'INTERRUPTED') throw new TestFlowInterruptedException()
        ['https://example.invalid/build/1', outcome]
    }
    try { parsed.newInstance(binding).run() } catch (Exception e) {
        state.thrown = e
        worsen(e instanceof InterruptedException ? 'ABORTED' : 'FAILURE')
    }
    boolean shouldScan = scan && outcome != 'INTERRUPTED' &&
        (outcome == 'SUCCESS' || (mode == 'nightly_release' && !failFast))
    assert state.scans == (shouldScan ? 1 : 0): "scan mismatch: ${mode}, failFast=${failFast}, scan=${scan}, result=${outcome}, ignore=${ignore}; ${state}"
    if (outcome == 'FAILURE' && shouldScan && !ignore) {
        assert state.result == (scanOutcome == 'INTERRUPTED' ? 'ABORTED' : 'FAILURE')
        assert state.stages['[Build-Docker-Images] Remote Run'] == 'FAILURE'
    }
    if (outcome == 'UNSTABLE' && scanOutcome == 'SUCCESS') assert state.result == 'UNSTABLE'
    if (outcome == 'INTERRUPTED' || (shouldScan && scanOutcome == 'INTERRUPTED')) {
        assert state.thrown instanceof InterruptedException
        assert state.result == 'ABORTED'
    }
    if (shouldScan && mode == 'nightly_release' && scanOutcome == 'FAILURE') {
        assert state.result == 'FAILURE'
        assert state.thrown != null
    }
    if (outcome == 'SUCCESS' && scanOutcome == 'SUCCESS') assert state.result == 'SUCCESS'
    count++
}
// First case reproduces the reported regression on the unpatched source.
runCase('nightly_release', false, true, 'FAILURE')
['nightly_release', 'pre_merge', 'post_merge'].each { mode ->
    [false, true].each { failFast ->
        [false, true].each { scan ->
            ['SUCCESS', 'FAILURE', 'UNSTABLE', 'ABORTED', 'INTERRUPTED'].each { outcome ->
                [false, true].each { ignore -> runCase(mode, failFast, scan, outcome, ignore) }
            }
        }
    }
}
['SUCCESS', 'FAILURE', 'UNSTABLE'].each { outcome ->
    ['FAILURE', 'INTERRUPTED'].each { scanOutcome ->
        runCase('nightly_release', false, true, outcome, false, scanOutcome)
    }
}
println "PASS: ${count} scenarios (production Groovy, mocked Jenkins and downstream jobs)"
