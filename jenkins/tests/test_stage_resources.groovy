/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 */

// Run from the repository root: groovy -cp <snakeyaml.jar> jenkins/tests/test_stage_resources.groovy
import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import org.yaml.snakeyaml.DumperOptions
import org.yaml.snakeyaml.LoaderOptions
import org.yaml.snakeyaml.Yaml
import org.yaml.snakeyaml.constructor.SafeConstructor

def source = new File('jenkins/L0_Test.groovy').text
def method = { String name ->
    int start = source.indexOf("def ${name}(")
    source.substring(start, source.indexOf('\ndef ', start + 1))
}
def loader = new GroovyClassLoader(this.class.classLoader)
loader.parseClass('''
    package trtllm.exceptions
    class InfraFailure extends RuntimeException {
        static final String K8S = 'K8S'
        static final String PERSISTENT = 'PERSISTENT'
        String severity = PERSISTENT
        String detectedPattern = 'test'
    }
    class PipelineInterruption extends RuntimeException {}
''')
def infra = loader.loadClass('trtllm.exceptions.InfraFailure')
def yamlOptions = new LoaderOptions()
yamlOptions.setAllowDuplicateKeys(false)
def yaml = new Yaml(new SafeConstructor(yamlOptions))
def dumpOptions = new DumperOptions()
dumpOptions.setDefaultFlowStyle(DumperOptions.FlowStyle.BLOCK)
def yamlWriter = new Yaml(dumpOptions)
def messages = []
def checkouts = []
def launches = []
String configJson = new File('jenkins/config/stage_resources.json').text
def utils = new Expando(
    checkoutFile: { Object... args -> checkouts << args.toList() },
    generateNodeLabel: { prefix -> "${prefix}-test-label".toString() },
    withKubernetesHostNodeExclusion: { config, nodes ->
        if (!nodes) return config
        // The shared library matches a quoted key in the original template.
        assert config.yaml.contains('- key: "tensorrt/taints"')
        def text = config.yaml.replaceAll(/(?m)^([ ]*)- key: "tensorrt\/taints"[ ]*$/) { match, indent ->
            "${indent}- key: \"kubernetes.io/hostname\"\n${indent}  operator: NotIn\n" +
                "${indent}  values: [\"host-1\"]\n${match}"
        }
        config + [yaml: text]
    }
)
def pipeline = new Expando(
    pwd: { Map options -> assert options.tmp; '/tmp/stage-resource-test' },
    readJSON: { Map options ->
        assert options.returnPojo
        assert options.file.toString() == '/tmp/stage-resource-test/stage-resources/stage_resources.json'
        new JsonSlurper().parseText(configJson)
    },
    readYaml: { Map options -> yaml.load(options.text.toString()) },
    writeYaml: { Map options -> assert options.returnText; yamlWriter.dump(options.data) },
    echo: { String text -> messages << text },
    error: { String text -> throw new IllegalArgumentException(text) }
)
def variables = [
    stageResourceRequests: [:], trtllm_utils: utils, LLM_REPO: 'test-repo', env: [gitlabCommit: 'test-sha'],
    Constants: [DEFAULT_NVIDIA_DRIVER_VERSION: 'driver', DEFAULT_NVIDIA_OPEN_DRIVER_VERSION: 'open-driver'],
    KubernetesManager: [selectGPU: { type -> 'test-gpu' }],
    REQUIRED_OPEN_DRIVER_TYPES: [], REQUIRED_NO_DRIVER_TYPES: ['gb10x'],
    POD_TIMEOUT_SECONDS_TEST: '60', POD_TIMEOUT_SECONDS_SLURM: '60', POD_TIMEOUT_SECONDS_BUILD: '60',
    ARTIFACTORY_IMAGE_PULL_SECRET: 'test-secret', MODEL_EXPRESS_REDIS_IMAGE: 'redis', MODEL_EXPRESS_SERVER_IMAGE: 'mx',
    testFilter: [debug: false], DEBUG_MODE: 'debug', K8S_INFRA_RETRY_MAX: 2,
    resolveInfraRetryMax: { scope, max -> max == null ? 2 : max },
    echo: { text -> messages << text }, sleep: { seconds -> },
    classifyFailure: { p, error, scope, context = null -> error },
    rememberAvoidedKubernetesHostNodes: { nodes, node, stage -> nodes << node },
    hasBudgetForInfraRetry: { Object... ignored -> true },
    registerSlurmResource: { stage, info -> }
]
def script = new GroovyShell(loader, new Binding(variables)).parse(
    'import trtllm.exceptions.*\n' +
    source.substring(source.indexOf('BUILD_CORES_REQUEST ='), source.indexOf('CCACHE_DIR=')) +
    ['loadStageResourceRequests', 'applyStageResourceRequests', 'createKubernetesPodConfig',
     'runKubernetesPodWithInfraRetry'].collect { method(it) }.join('\n')
)
script.run()

assert script.loadStageResourceRequests(pipeline) == [:]
assert checkouts == [['test-repo', 'test-sha', 'jenkins/config/stage_resources.json',
                      '/tmp/stage-resource-test/stage-resources']]
configJson = JsonOutput.toJson([stages: ['test-stage': [cpu_m: 1001, memory_mi: 10241]]])
variables.stageResourceRequests = script.loadStageResourceRequests(pipeline)
assert variables.stageResourceRequests['test-stage'] == [cpu_m: 1001, memory_mi: 10241]
def goodConfig = configJson
[[cpu_m: 0, memory_mi: 1], [cpu_m: -1, memory_mi: 1], [cpu_m: 1.5, memory_mi: 1],
 [cpu_m: '1000', memory_mi: 1], [cpu_m: 1000], [memory_mi: 1024]].each { invalid ->
    configJson = JsonOutput.toJson([stages: ['bad-stage': invalid]])
    try {
        script.loadStageResourceRequests(pipeline)
        assert false: "Accepted invalid resources: ${invalid}"
    } catch (IllegalArgumentException e) {
        assert e.message.contains('bad-stage')
    }
}
configJson = goodConfig

def testContainer = { pod -> pod.spec.containers.find { it.name == 'trt-llm' } }
def expectedRequests = [cpu: '1001m', memory: '10241Mi']
def expectedLimits = [cpu: '1402m', memory: '14338Mi']
[['cpu', 'amd64', 1, false], ['a10', 'amd64', 1, false],
 ['dgx-h100-x4', 'amd64', 4, true], ['gb10x', 'arm64', 1, false]].each { type, arch, gpus, mx ->
    def original = script.createKubernetesPodConfig('test-image', type, arch, gpus, mx)
    def originalYaml = original.yaml
    def expected = yaml.load(original.yaml)
    testContainer(expected).resources.requests.putAll(expectedRequests)
    testContainer(expected).resources.limits.putAll(expectedLimits)
    def updated = script.applyStageResourceRequests(pipeline, original, 'test-stage')
    assert yaml.load(updated.yaml) == expected
    assert original.yaml == originalYaml
    assert updated.findAll { it.key != 'yaml' } == original.findAll { it.key != 'yaml' }
    assert script.applyStageResourceRequests(pipeline, original, 'unconfigured-stage').is(original)
}
assert messages.any { it.contains('unconfigured-stage: using default') }

def gpuConfig = script.createKubernetesPodConfig('test-image', 'a10')
def launch = { config, context, body ->
    launches << config
    if (context != null) context.lastHostNode = "host-${launches.size()}".toString()
    body()
}
utils.launchKubernetesPod = { p, config, container, body -> launch(config, null, body) }
utils.launchKubernetesPodWithPlacement = { p, config, container, context, body -> launch(config, context, body) }

[[:], [singleAttempt: true]].each { options ->
    launches.clear()
    script.runKubernetesPodWithInfraRetry(options, pipeline, gpuConfig, 'trt-llm', 'test-stage', { tag, isFinal, ctx -> })
    assert launches.size() == 1
    assert testContainer(yaml.load(launches[0].yaml)).resources.limits.cpu == '1402m'
}
variables.testFilter.debug = true
launches.clear()
script.runKubernetesPodWithInfraRetry([:], pipeline, gpuConfig, 'trt-llm', 'test-stage', { tag, isFinal, ctx -> })
assert testContainer(yaml.load(launches[0].yaml)).resources.requests.memory == '10241Mi'
variables.testFilter.debug = false

def slurmConfig = script.createKubernetesPodConfig('test-image', 'slurm')
launches.clear()
script.runKubernetesPodWithInfraRetry([singleAttempt: true, slurmDispatcher: true], pipeline, slurmConfig,
    'trt-llm', 'test-stage', { tag, isFinal, ctx -> })
assert launches[0].is(slurmConfig)

launches.clear()
def attemptTags = []
script.runKubernetesPodWithInfraRetry([:], pipeline, gpuConfig, 'trt-llm', 'test-stage', { tag, isFinal, ctx ->
    attemptTags << tag.toString()
    if (launches.size() == 1) throw infra.newInstance()
})
assert attemptTags == ['', '-pod-2']
assert launches.size() == 2
launches.each { config -> assert testContainer(yaml.load(config.yaml)).resources.requests.cpu == '1001m' }
def expressions = yaml.load(launches[1].yaml).spec.affinity.nodeAffinity
    .requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions
assert expressions.find { it.key == 'kubernetes.io/hostname' }.values == ['host-1']

println 'PASS: config loading and validation, CPU/GPU/Spark/sidecars, defaults, debug/single-attempt, SLURM exclusion, retry placement'
