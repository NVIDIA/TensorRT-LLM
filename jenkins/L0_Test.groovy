@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.InterruptedException
import groovy.transform.Field
import groovy.json.JsonSlurper
import groovy.json.JsonOutput
import com.nvidia.bloom.KubernetesManager
import com.nvidia.bloom.Constants
import org.jenkinsci.plugins.workflow.cps.CpsThread
import org.jsoup.Jsoup
import org.jenkinsci.plugins.pipeline.modeldefinition.Utils as jUtils

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}
LLM_ROOT = "llm"

ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
UPLOAD_PATH = env.uploadPath ? env.uploadPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"

X86_64_TRIPLE = "x86_64-linux-gnu"
AARCH64_TRIPLE = "aarch64-linux-gnu"

// default package name
linuxPkgName = ( env.targetArch == AARCH64_TRIPLE ? "tensorrt-llm-sbsa-release-src-" : "tensorrt-llm-release-src-" ) + (env.artifactCommit ? env.artifactCommit : env.gitlabCommit) + ".tar.gz"

// Container configuration
// available tags can be found in: https://urm.nvidia.com/artifactory/sw-tensorrt-docker/tensorrt-llm/
// [base_image_name]-[arch]-[os](-[python_version])-[trt_version]-[torch_install_type]-[stage]-[date]-[mr_id]
LLM_DOCKER_IMAGE = env.dockerImage
LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:cuda-12.8.1-devel-rocky8-x86_64-rocky8-py310-trt10.9.0.34-skip-devel-202504101610-3421"
LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:cuda-12.8.1-devel-rocky8-x86_64-rocky8-py312-trt10.9.0.34-skip-devel-202504101610-3421"

// DLFW torch image
DLFW_IMAGE = "nvcr.io/nvidia/pytorch:25.03-py3"

//Ubuntu base image
UBUNTU_22_04_IMAGE = "urm.nvidia.com/docker/ubuntu:22.04"
UBUNTU_24_04_IMAGE = "urm.nvidia.com/docker/ubuntu:24.04"

POD_TIMEOUT_SECONDS = env.podTimeoutSeconds ? env.podTimeoutSeconds : "21600"

// Literals for easier access.
@Field
def TARNAME = "tarName"

@Field
def VANILLA_CONFIG = "Vanilla"

@Field
def SINGLE_DEVICE_CONFIG = "SingleDevice"

@Field
def LLVM_CONFIG = "LLVM"

@Field
LINUX_AARCH64_CONFIG = "linux_aarch64"

@Field
def BUILD_CONFIGS = [
  // Vanilla TARNAME is used for packaging in runLLMPackage
  (VANILLA_CONFIG) : [(TARNAME) : "TensorRT-LLM.tar.gz"],
  (SINGLE_DEVICE_CONFIG) : [(TARNAME) : "single-device-TensorRT-LLM.tar.gz"],
  (LLVM_CONFIG) : [(TARNAME) : "llvm-TensorRT-LLM.tar.gz"],
  (LINUX_AARCH64_CONFIG) : [(TARNAME) : "TensorRT-LLM-GH200.tar.gz"],
]

// TODO: Move common variables to an unified location
BUILD_CORES_REQUEST = "8"
BUILD_CORES_LIMIT = "8"
BUILD_MEMORY_REQUEST = "48Gi"
BUILD_MEMORY_LIMIT = "64Gi"
BUILD_JOBS = "8"

TESTER_CORES = "12"
TESTER_MEMORY = "96Gi"

CCACHE_DIR="/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"
MODEL_CACHE_DIR="/scratch.trt_llm_data/llm-models"

def trimForStageList(stageNameList)
{
    if (stageNameList == null) {
        return null
    }
    trimedList = []
    stageNameList.each { stageName ->
        trimedList.add(stageName.trim().replaceAll('\\\\', ''))
    }
    return trimedList
}

// Test filter flags
@Field
def REUSE_STAGE_LIST = "reuse_stage_list"
@Field
def ENABLE_SKIP_TEST = "skip_test"
@Field
def TEST_STAGE_LIST = "stage_list"
@Field
def GPU_TYPE_LIST = "gpu_type"
@Field
def IS_POST_MERGE = "post_merge"
@Field
def ADD_MULTI_GPU_TEST = "add_multi_gpu_test"
@Field
def ONLY_MULTI_GPU_TEST = "only_multi_gpu_test"
@Field
def DISABLE_MULTI_GPU_TEST = "disable_multi_gpu_test"
@Field
def EXTRA_STAGE_LIST = "extra_stage"
@Field
def MULTI_GPU_FILE_CHANGED = "multi_gpu_file_changed"
@Field
def ONLY_PYTORCH_FILE_CHANGED = "only_pytorch_file_changed"
@Field
def DEBUG_MODE = "debug"
@Field
def testFilter = [
    (REUSE_STAGE_LIST): null,
    (ENABLE_SKIP_TEST): false,
    (TEST_STAGE_LIST): null,
    (GPU_TYPE_LIST): null,
    (IS_POST_MERGE): false,
    (ADD_MULTI_GPU_TEST): false,
    (ONLY_MULTI_GPU_TEST): false,
    (DISABLE_MULTI_GPU_TEST): false,
    (EXTRA_STAGE_LIST): null,
    (MULTI_GPU_FILE_CHANGED): false,
    (ONLY_PYTORCH_FILE_CHANGED): false,
    (DEBUG_MODE): false,
]

String getShortenedJobName(String path)
{
    static final nameMapping = [
        "L0_MergeRequest": "l0-mr",
        "L0_Custom": "l0-cus",
        "L0_PostMerge": "l0-pm",
        "L0_PostMergeDocker": "l0-pmd",
        "L1_Custom": "l1-cus",
        "L1_Nightly": "l1-nt",
        "L1_Stable": "l1-stb",
    ]
    def parts = path.split('/')
    // Apply nameMapping to the last part (jobName)
    def jobName = parts[-1]
    boolean replaced = false
    nameMapping.each { key, value ->
        if (jobName.contains(key)) {
            jobName = jobName.replace(key, value)
            replaced = true
        }
    }
    if (!replaced) {
        jobName = jobName.length() > 7 ? jobName.substring(0, 7) : jobName
    }
    // Replace the last part with the transformed jobName
    parts[-1] = jobName
    // Rejoin the parts with '-', convert to lowercase
    return parts.join('-').toLowerCase()
}

def cacheErrorAndUploadResult(stageName, taskRunner, finallyRunner, noResultIfSuccess=false)
{
    checkStageName([stageName])
    def Boolean stageIsInterrupted = false
    def Boolean stageIsFailed = true
    try {
        taskRunner()
        stageIsFailed = false
    } catch (InterruptedException e) {
        stageIsInterrupted = true
        throw e
    } finally {
        if (stageIsInterrupted) {
            echo "Stage is interrupted, skip to upload test result."
        } else {
            sh 'if [ "$(id -u)" -eq 0 ]; then dmesg; fi'
            if (noResultIfSuccess && !stageIsFailed) {
                return
            }
            echo "noResultIfSuccess: ${noResultIfSuccess}, stageIsFailed: ${stageIsFailed}"
            sh "mkdir -p ${stageName}"
            finallyRunner()
            if (stageIsFailed) {
                def stageXml = generateStageFailTestResultXml(stageName, "Stage Failed", "Stage run failed without result", "results*.xml")
                if (stageXml != null) {
                    sh "echo '${stageXml}' > ${stageName}/results-stage.xml"
                }
            }
            sh "STAGE_NAME=${stageName}"
            sh "STAGE_NAME=${stageName} && env | sort > ${stageName}/debug_env.txt"
            echo "Upload test results."
            sh "tar -czvf results-${stageName}.tar.gz ${stageName}/"
            trtllm_utils.uploadArtifacts(
                "results-${stageName}.tar.gz",
                "${UPLOAD_PATH}/test-results/"
            )
            junit(testResults: "${stageName}/results*.xml")
        }
    }
}

def createKubernetesPodConfig(image, type, arch = "amd64", gpuCount = 1, perfMode = false)
{
    def targetCould = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/arch: ${arch}
                  kubernetes.io/os: linux"""
    def containerConfig = ""
    def nodeLabelPrefix = ""
    def jobName = getShortenedJobName(env.JOB_NAME)
    def buildID = env.BUILD_ID

    switch(type)
    {
    case "agent":
        containerConfig = """
                  - name: alpine
                    image: urm.nvidia.com/docker/alpine:latest
                    command: ['cat']
                    tty: true
                    resources:
                      requests:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    case "build":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS}]
                    volumeMounts:
                    - name: sw-tensorrt-pvc
                      mountPath: "/mnt/sw-tensorrt-pvc"
                      readOnly: false
                    tty: true
                    resources:
                      requests:
                        cpu: ${BUILD_CORES_REQUEST}
                        memory: ${BUILD_MEMORY_REQUEST}
                        ephemeral-storage: 200Gi
                      limits:
                        cpu: ${BUILD_CORES_LIMIT}
                        memory: ${BUILD_MEMORY_LIMIT}
                        ephemeral-storage: 200Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    default:
        def hasMultipleGPUs = (gpuCount > 1)
        def memorySize = "${TESTER_MEMORY}"
        def storageSize = "300Gi"
        def driverVersion = Constants.DEFAULT_NVIDIA_DRIVER_VERSION
        def cpuCount = "${TESTER_CORES}"

        // Multi-GPU only supports DGX-H100 and DGX-H200 due to the hardware stability.
        if ((type.contains("dgx-h100") || type.contains("dgx-h200")) && hasMultipleGPUs)
        {
            // Not a hard requirement, but based on empirical values.
            memorySize = "${gpuCount * 150}" + "Gi"
            storageSize = "${gpuCount * 150}" + "Gi"
            cpuCount = "${gpuCount * 12}"
        }

        def gpuType = KubernetesManager.selectGPU(type)
        nodeLabelPrefix = type

        targetCould = "kubernetes"

        // The following GPU types doesn't support dynamic driver flashing.
        if (type == "b100-ts2" || type.contains("dgx-h100") || type.contains("dgx-h200") || type == "gh200" ) {
            selectors = """
                    kubernetes.io/arch: ${arch}
                    kubernetes.io/os: linux
                    nvidia.com/gpu_type: ${gpuType}"""
        } else if (perfMode && !hasMultipleGPUs) {
        // Not using the "perf" node currently due to hardware resource constraint.
        // Use single GPU machine with "tensorrt/test_type: perf" for stable perf testing.
        // H100 / A100 single GPU machine has this unique label in TensorRT Blossom pool.
            selectors = """
                    kubernetes.io/arch: ${arch}
                    kubernetes.io/os: linux
                    nvidia.com/gpu_type: ${gpuType}
                    nvidia.com/driver_version: '${driverVersion}'"""
        }
        else
        {
            selectors = """
                    kubernetes.io/arch: ${arch}
                    kubernetes.io/os: linux
                    nvidia.com/gpu_type: ${gpuType}
                    nvidia.com/driver_version: '${driverVersion}'"""
        }

        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS}]
                    tty: true
                    resources:
                      requests:
                        cpu: ${cpuCount}
                        memory: ${memorySize}
                        nvidia.com/gpu: ${gpuCount}
                        ephemeral-storage: ${storageSize}
                      limits:
                        cpu: ${cpuCount}
                        memory: ${memorySize}
                        nvidia.com/gpu: ${gpuCount}
                        ephemeral-storage: ${storageSize}
                    imagePullPolicy: Always
                    volumeMounts:
                    - name: dshm
                      mountPath: /dev/shm
                    - name: scratch-trt-llm-data
                      mountPath: /scratch.trt_llm_data
                      readOnly: true
                    - name: sw-tensorrt-pvc
                      mountPath: "/mnt/sw-tensorrt-pvc"
                      readOnly: false
                    securityContext:
                      capabilities:
                        add:
                        - SYS_ADMIN"""
        break
    }
    def nodeLabel = trtllm_utils.appendRandomPostfix("${nodeLabelPrefix}---tensorrt-${jobName}-${buildID}")
    def pvcVolume = """
                - name: sw-tensorrt-pvc
                  persistentVolumeClaim:
                    claimName: sw-tensorrt-pvc
    """
    if (arch == "arm64") {
        // WAR: PVC mount is not setup on aarch64 platform, use nfs as a WAR
        pvcVolume = """
                - name: sw-tensorrt-pvc
                  nfs:
                    server: 10.117.145.13
                    path: /vol/scratch1/scratch.svc_tensorrt_blossom
        """
    }
    def podConfig = [
        cloud: targetCould,
        namespace: "sw-tensorrt",
        label: nodeLabel,
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                qosClass: Guaranteed
                affinity:
                    nodeAffinity:
                        requiredDuringSchedulingIgnoredDuringExecution:
                            nodeSelectorTerms:
                            - matchExpressions:
                              - key: "tensorrt/taints"
                                operator: DoesNotExist
                              - key: "tensorrt/affinity"
                                operator: NotIn
                                values:
                                - "core"
                nodeSelector: ${selectors}
                containers:
                  ${containerConfig}
                    env:
                    - name: HOST_NODE_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: spec.nodeName
                  - name: jnlp
                    image: urm.nvidia.com/docker/jenkins/inbound-agent:4.11-1-jdk11
                    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
                    resources:
                      requests:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                qosClass: Guaranteed
                volumes:
                - name: dshm
                  emptyDir:
                    medium: Memory
                - name: scratch-trt-llm-data
                  nfs:
                    server: 10.117.145.14
                    path: /vol/scratch1/scratch.michaeln_blossom
                ${pvcVolume}
        """.stripIndent(),
    ]

    return podConfig
}

def echoNodeAndGpuInfo(pipeline, stageName)
{
    String hostNodeName = sh(script: 'echo $HOST_NODE_NAME', returnStdout: true)
    String gpuUuids = pipeline.sh(script: "nvidia-smi -q | grep \"GPU UUID\" | awk '{print \$4}' | tr '\n' ',' || true", returnStdout: true)
    pipeline.echo "HOST_NODE_NAME = ${hostNodeName} ; GPU_UUIDS = ${gpuUuids} ; STAGE_NAME = ${stageName}"
}

def runLLMDocBuild(pipeline, config)
{
    // Step 1: cloning tekit source code
    sh "pwd && ls -alh"
    sh "env | sort"
    // allow to checkout from forked repo, svc_tensorrt needs to have access to the repo, otherwise clone will fail
    trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
    sh "mkdir TensorRT-LLM"
    sh "cp -r ${LLM_ROOT}/ TensorRT-LLM/src/"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "git config --global --add safe.directory \"*\"")

    def llmPath = sh (script: "realpath .", returnStdout: true).trim()
    def llmSrc = "${llmPath}/TensorRT-LLM/src"

    // Step 2: download TRT-LLM tarfile
    def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${BUILD_CONFIGS[config][TARNAME]}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
    sh "cd ${llmPath} && tar -zxf ${BUILD_CONFIGS[config][TARNAME]}"
    // install python package
    if (env.alternativeTRT) {
        sh "cd ${llmSrc} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
    }
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmSrc} && pip3 install --retries 1 -r requirements-dev.txt")
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl")

    // Step 3: build doc
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update")
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install doxygen python3-pip graphviz -y")

    def containerPATH = sh(script: "echo \${PATH}", returnStdout: true).replaceAll("\\s", "")
    if (!containerPATH.contains("/usr/local/bin:")) {
        echo "Prepend /usr/local/bin into \${PATH}"
        containerPATH = "/usr/local/bin:${containerPATH}"
    }
    containerPATH = containerPATH.replaceAll(':+$', '')
    withEnv(["PATH=${containerPATH}"]) {
        sh "env | sort"
        sh "rm -rf ${LLM_ROOT}/docs/build"
        trtllm_utils.llmExecStepWithRetry(
            pipeline,
            script: """
                cd ${LLM_ROOT}/docs && \
                pip3 install -r requirements.txt && \
                pip3 install git+https://github.com/sphinx-doc/sphinx.git@v7.4.7 && \
                doxygen Doxygen && \
                make html && \
                cd build/html && \
                touch .nojekyll
            """
        )
    }

    echo "Upload built html."
    sh "tar -czvf doc-html-preview.tar.gz  ${LLM_ROOT}/docs/build/html"
    trtllm_utils.uploadArtifacts(
        "doc-html-preview.tar.gz",
        "${UPLOAD_PATH}/test-results/"
    )
}

def launchTestListCheck(pipeline)
{
    stageName = "Test List Check"
    trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(LLM_DOCKER_IMAGE, "a10"), "trt-llm", {
        try {
            echoNodeAndGpuInfo(pipeline, stageName)
            trtllm_utils.llmExecStepWithRetry(pipeline, script: """apt-get update && apt-get install \
            libffi-dev \
            -y""")
            sh "nvidia-smi -q"
            // download TRT-LLM tarfile
            def tarName = BUILD_CONFIGS[VANILLA_CONFIG][TARNAME]
            def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pwd && wget -nv ${llmTarfile} && ls -alh")
            sh "tar -zxf ${tarName}"
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def llmSrc = "${llmPath}/TensorRT-LLM/src"
            sh "python3 ${llmSrc}/scripts/check_test_list.py --l0 --qa"
        } catch (InterruptedException e) {
            throw e
        } catch (Exception e) {
            throw e
        }
    })
}

def generateStageFailTestResultXml(stageName, subName, failureLog, resultPath) {
    String resultFiles = sh(script: "cd ${stageName} && ls -l ${resultPath} | wc -l", returnStdout: true).trim()
    echo "${resultFiles}"
    if (resultFiles != "0") {
        return null
    }
    return """<?xml version="1.0" encoding="UTF-8"?><testsuites>
        <testsuite name="${stageName}" errors="0" failures="1" skipped="0" tests="1" time="1.00">
        <testcase name="${subName}" classname="${stageName}" time="1.0">
        <failure message="${failureLog}"> ${failureLog}
        </failure></testcase></testsuite></testsuites>"""
}

def getMakoOpts(getMakoScript, makoArgs=[]) {
    // We want to save a map for the Mako opts
    def makoOpts = [:]
    def turtleOutput = ""

    // Echo the command
    // NOTE: We redirect stderr to stdout so that we can capture
    //  both stderr and stdout streams with the 'returnStdout' flag
    //  in sh command.
    def listMakoCmd = [
        "python3",
        getMakoScript,
        "--device 0"].join(" ")

    if (makoArgs) {
        def makoOptArgs = makoArgs.collect { "--mako-opt " + it }
        listMakoCmd += " " + makoOptArgs.join(" ")
    }
    // Add the withCredentials step to access gpu-chip-mapping file
    withCredentials([file(credentialsId: 'gpu-chip-mapping', variable: 'GPU_CHIP_MAPPING')]) {
        listMakoCmd = [listMakoCmd, "--chip-mapping-file ${GPU_CHIP_MAPPING}"].join(" ")
        listMakoCmd = [listMakoCmd, "2>&1"].join(" ")

        echo "Scripts to get Mako list, cmd: ${listMakoCmd}"

        // Capture the mako output, add timeout in case any hang
        timeout(time: 30, unit: 'MINUTES'){
            turtleOutput = sh(label: "Capture Mako Parameters", script: listMakoCmd, returnStdout: true)
        }
    }

    // Validate output
    assert turtleOutput: "Mako opts not found - could not construct test db test list."

    // Split each line of turtle output into a list
    def turtleOutList = turtleOutput.split("\n")

    // Extract the mako opts
    def startedMakoOpts = false
    def param = null
    def value = null
    turtleOutList.each { val ->
        if (startedMakoOpts) {
            // Handle case where value is missing
            param = null
            value = null
            try {
                (param, value) = val.split("=")
            } catch (ArrayIndexOutOfBoundsException ex) {
                param = val.split("=")[0]
                value = null
            }

            // Try to convert nulls, booleans, and floats into the correct type
            if (value != null) {
                if (value.toLowerCase() == "none") {
                    echo "Converted mako param '${param}' value '${value}' to 'null'"
                    value = null
                } else if (value.toLowerCase() in ["true", "false"]) {
                    echo "Converted mako param '${param}' value '${value}' to Boolean '${value.toBoolean()}'"
                    value = value.toBoolean()
                }
            }
            makoOpts[(param)] = value
        }
        if (val.equals("Mako options:")) {
            startedMakoOpts = true
        }
    }

    // Finally, convert the query to a json string
    def makoOptsJson = JsonOutput.toJson(makoOpts)

    // Print and return the Test DB Query as a JSON string
    echo "Test DB Mako opts: ${makoOptsJson}"

    return makoOptsJson
}

def renderTestDB(testContext, llmSrc, stageName) {
    def scriptPath = "${llmSrc}/tests/integration/defs/sysinfo/get_sysinfo.py"
    def makoArgs = []
    def isPostMerge = stageName.contains("Post-Merge")
    makoArgs += [isPostMerge ? "stage=post_merge" : "stage=pre_merge"]
    // Determine the backend type based on keywords in stageName
    if (stageName.contains("-PyTorch-")) {
        // If stageName contains "-PyTorch-", add "backend=pytorch" to makoArgs
        // At this point, only tests with backend=pytorch or unspecified backend will be run
        makoArgs += ["backend=pytorch"]
    } else if (stageName.contains("-TensorRT-")) {
        // If stageName contains "-TensorRT-", add "backend=tensorrt" to makoArgs
        // At this point, only tests with backend=tensorrt or unspecified backend will be run
        makoArgs += ["backend=tensorrt"]
    } else if (stageName.contains("-CPP-")) {
        // If stageName contains "-CPP-", add "backend=cpp" to makoArgs
        // At this point, only tests with backend=cpp or unspecified backend will be run
        makoArgs += ["backend=cpp"]
    } else {
        // If stageName does not contain "-PyTorch-", "-TensorRT-", or "-CPP-", do not add any backend
        // At this point, all tests will be run
        // For cases where backend is not specified in makoArgs, we will match all types of backends and tests without specified backend
    }
    def makoOpts = getMakoOpts(scriptPath, makoArgs)

    sh "pip3 install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-tensorrt-pypi/simple --ignore-installed trt-test-db==1.8.5+bc6df7"
    def testDBPath = "${llmSrc}/tests/integration/test_lists/test-db"
    def testList = "${llmSrc}/${testContext}.txt"
    def testDBQueryCmd = [
        "trt-test-db",
        "-d",
        testDBPath,
        "--context",
        testContext,
        "--test-names",
        "--output",
        testList,
        "--match",
        "'${makoOpts}'"
    ].join(" ")

    sh(label: "Render test list from test-db", script: testDBQueryCmd)
    sh(script: "cat ${testList}")

    return testList
}

def getSSHConnectionPorts(portConfigFile, stageName)
{
    def type = stageName.split('-')[0]
    echo "The type is: ${type}"
    def fileContent = sh(script: "cat ${portConfigFile}", returnStdout: true).trim()

    // Get available VM port list from portConfigFile based on stage name (e.g. A10: [10022, 10023])
    def portList = []
    fileContent.split('\n').each { line ->
        def matcher = (line =~ /(.+?)=\[(.+?)\]/)
        if (matcher) {
            def key = matcher[0][1].replaceAll("\\s","")
            def values = matcher[0][2].replaceAll("\\s","").split(',').collect { it.replaceAll("\\s","") }
            if (key == type) {
                portList.addAll(values)
            }
        }
    }
    echo "Port List for ${type}: ${portList}"

    // Get current port usage status
    def portUsage = ""
    withCredentials([
        usernamePassword(credentialsId: 'tensorrt_llm_infra_debug_vm_01_credentials', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD'),
        string(credentialsId: 'DEBUG_HOST_NAME', variable: 'HOST_NAME')
        ]) {
        portUsage = sh(script: "ssh -v ${USERNAME}@${HOST_NAME} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null 'netstat -tuln'",returnStdout: true)
    }
    echo "Port Usage: ${portUsage}"

    // Get an available VM port
    def userPort = 0
    while (portList.size() > 0) {
        def randomIndex = (int)(Math.random() * portList.size())
        def curPort = portList[randomIndex].toInteger()
        if (!portUsage.contains(":${curPort}")) {
            userPort = curPort
            break
        }
        portList.remove(randomIndex)
    }

    if (userPort == 0) {
        echo "There is no available port for ${type}"
        return [0, 0]
    }

    echo "The chosen port is: ${userPort}"

    // Calculate autossh monitor port by subtracting 9000 from VM port (e.g. 10022 -> 1022)
    // If monitor port is already in use, randomly assign a value between 2000-3000
    def monitorPort = userPort - 9000
    while (portUsage.contains(":${monitorPort}")) {
        monitorPort = 2000 + (int)(Math.random() * 1000)
    }

    echo "The monitor port is: ${monitorPort}"

    return [userPort, monitorPort]
}

def runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312")
{
    // Step 1: create LLM_ROOT dir
    sh "pwd && ls -alh"
    def llmRootConfig = "${LLM_ROOT}${config}"
    sh "mkdir ${llmRootConfig}"

    def llmPath = sh (script: "realpath ${llmRootConfig}",returnStdout: true).trim()
    def llmSrc = "${llmPath}/TensorRT-LLM/src"
    echoNodeAndGpuInfo(pipeline, stageName)

    if (env.alternativeTRT && cpver) {
        stage("Replace TensorRT") {
            trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
        }
    }

    // Step 2: run tests
    stage ("Setup environment")
    {
        // Random sleep to avoid resource contention
        sleep(10 * Math.random())
        sh "curl ifconfig.me || true"
        sh "nproc && free -g && hostname"
        echoNodeAndGpuInfo(pipeline, stageName)
        sh "cat ${MODEL_CACHE_DIR}/README"
        sh "nvidia-smi -q"
        sh "df -h"

        // setup HF_HOME to cache model and datasets
        // init the huggingface cache from nfs, since the nfs is read-only, and HF_HOME needs to be writable, otherwise it will fail at creating file lock
        sh "mkdir -p ${HF_HOME} && ls -alh ${HF_HOME}"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update")
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install -y rsync")
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "rsync -r ${MODEL_CACHE_DIR}/hugging-face-cache/ ${HF_HOME}/ && ls -lh ${HF_HOME}")
        sh "df -h"

        // install package
        sh "env | sort"
        sh "which python3"
        sh "python3 --version"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install -y libffi-dev")
        sh "rm -rf results-${stageName}.tar.gz ${stageName}/*"
        // download TRT-LLM tarfile
        def tarName = BUILD_CONFIGS[config][TARNAME]
        def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
        sh "cd ${llmPath} && tar -zxf ${tarName}"

        // install python package
        if (env.alternativeTRT) {
            sh "cd ${llmSrc} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
        }
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmSrc} && pip3 install --retries 1 -r requirements-dev.txt")
        if (!skipInstallWheel) {
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl")
        }
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "git config --global --add safe.directory \"*\"")
    }

    if (testFilter[(DEBUG_MODE)]) {
        stage("Interactive debug session")
        {
            testFilter[(DEBUG_MODE)] = false

            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install openssh-server -y")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install autossh -y")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install sshpass -y")

            sh """
                echo 'Port 22' >> /etc/ssh/sshd_config
                echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
                echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
                echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config
                echo 'AllowTcpForwarding yes' >> /etc/ssh/sshd_config
                echo 'GatewayPorts yes' >> /etc/ssh/sshd_config
                cat /etc/ssh/sshd_config
            """

            sh "service ssh restart"
            sh "service ssh status"

            sh "ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N '' -q"

            sh """
                chmod 700 ~/.ssh
                chmod 400 ~/.ssh/id_rsa
                touch ~/.ssh/authorized_keys
                chmod 600 ~/.ssh/authorized_keys
            """

            // The portConfig file is in the VM
            def portConfigFilePath = "/root/.ssh/ports_config.txt"

            withCredentials([
                usernamePassword(credentialsId: 'tensorrt_llm_infra_debug_vm_01_credentials', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD'),
                string(credentialsId: 'DEBUG_HOST_NAME', variable: 'HOST_NAME')
                ]) {
                sh "sshpass -p ${PASSWORD} -v ssh ${USERNAME}@${HOST_NAME} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub"
                sh "ssh -v ${USERNAME}@${HOST_NAME} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null 'echo \"\" > ~/.ssh/known_hosts && cat ~/.ssh/id_rsa.pub' >> ~/.ssh/authorized_keys"
                sh "ssh -v ${USERNAME}@${HOST_NAME} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null 'cat ~/.ssh/ports_config.txt' >> ${portConfigFilePath}"

                def (int userPort, int monitorPort) = getSSHConnectionPorts(portConfigFilePath, stageName)
                if (userPort == 0) {
                    echo "Fail to setup an interactive debug session and exit the debug mode."
                    testFilter[(DEBUG_MODE)] = false
                    return
                }

                sh "ssh -f -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -L 1111:127.0.0.1:${monitorPort} -R ${monitorPort}:127.0.0.1:1112 -NR ${userPort}:localhost:22 ${USERNAME}@${HOST_NAME}"
                sh "autossh -fNR ${userPort}:localhost:22 ${USERNAME}@${HOST_NAME}"
                sh "ps aux | grep ssh"
                try {
                    timeout(time: 2, unit: 'HOURS') {
                        input message: "Pause 2 hours for Pre-Debug. Please type 'ssh root@${HOST_NAME} -p ${userPort}' on the CLI to create the connection. Please press the button to proceed when you finish debugging."
                    }
                } catch (InterruptedException e) {
                    echo "Pre-debug session was interrupted by user or timeout"
                    currentBuild.result = 'ABORTED'
                    error("Pipeline aborted during pre-debug session")
                } catch (Exception e) {
                    echo "An error occurred during pre-debug session: ${e.message}"
                    currentBuild.result = 'FAILURE'
                    error("Error in pre-debug session: ${e.message}")
                }
            }

            testFilter[(DEBUG_MODE)] = true
        }
    }

    stage ("[${stageName}] Run Pytest")
    {
        echoNodeAndGpuInfo(pipeline, stageName)
        sh 'if [ "$(id -u)" -eq 0 ]; then dmesg -C; fi'

        def extraInternalEnv = ""
        // Move back to 3600 once TRTLLM-4000 gets resolved
        def pytestTestTimeout = "7200"

        // TRT uses half of the host logic cores for engine building which is bad for multi-GPU machines.
        extraInternalEnv = "__LUNOWUD=\"-thread_pool_size=${TESTER_CORES}\""
        // CPP test execution is timing out easily, so we always override the timeout to 7200
        extraInternalEnv += " CPP_TEST_TIMEOUT_OVERRIDDEN=7200"

        def testDBList = renderTestDB(testList, llmSrc, stageName)
        testList = "${testList}_${splitId}"
        def testCmdLine = [
            "LLM_ROOT=${llmSrc}",
            "LLM_MODELS_ROOT=${MODEL_CACHE_DIR}",
            extraInternalEnv,
            "pytest",
            "-v",
            "--apply-test-list-correction",
            "--splitting-algorithm least_duration",
            "--timeout=${pytestTestTimeout}",
            "--rootdir ${llmSrc}/tests/integration/defs",
            "--test-prefix=${stageName}",
            "--splits ${splits}",
            "--group ${splitId}",
            "--waives-file=${llmSrc}/tests/integration/test_lists/waives.txt",
            "--test-list=${testDBList}",
            "--output-dir=${WORKSPACE}/${stageName}/",
            "--csv=${WORKSPACE}/${stageName}/report.csv",
            "--junit-xml ${WORKSPACE}/${stageName}/results.xml",
            "-o junit_logging=out-err"
        ]
        if (perfMode) {
            testCmdLine += [
                "--perf",
                "--perf-log-formats csv",
                "--perf-log-formats yaml"
            ]
        }
        // Test Coverage
        def TRTLLM_WHL_PATH = sh(returnStdout: true, script: "pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2").replaceAll("\\s","")
        sh "echo ${TRTLLM_WHL_PATH}"
        def coverageConfigFile = "${llmSrc}/${stageName}/.coveragerc"
        sh "mkdir -p ${llmSrc}/${stageName} && touch ${coverageConfigFile}"
        sh """
            echo '[run]' > ${coverageConfigFile}
            echo 'branch = True' >> ${coverageConfigFile}
            echo 'data_file = ${WORKSPACE}/${stageName}/.coverage.${stageName}' >> ${coverageConfigFile}
            echo '[paths]' >> ${coverageConfigFile}
            echo 'source =\n    ${llmSrc}/tensorrt_llm/\n    ${TRTLLM_WHL_PATH}/tensorrt_llm/' >> ${coverageConfigFile}
            cat ${coverageConfigFile}
        """
        testCmdLine += [
            "--cov=${llmSrc}/examples/",
            "--cov=${llmSrc}/tensorrt_llm/",
            "--cov=${TRTLLM_WHL_PATH}/tensorrt_llm/",
            "--cov-report=",
            "--cov-config=${coverageConfigFile}"
        ]

        def containerPIP_LLM_LIB_PATH = sh(script: "pip3 show tensorrt_llm | grep \"Location\" | awk -F\":\" '{ gsub(/ /, \"\", \$2); print \$2\"/tensorrt_llm/libs\"}'", returnStdout: true).replaceAll("\\s","")
        def containerLD_LIBRARY_PATH = sh(script: "echo \${LD_LIBRARY_PATH}", returnStdout: true).replaceAll("\\s","")
        if (!containerLD_LIBRARY_PATH.contains("${containerPIP_LLM_LIB_PATH}:")) {
            echo "Prepend ${containerPIP_LLM_LIB_PATH} into \${LD_LIBRARY_PATH}"
            containerLD_LIBRARY_PATH = "${containerPIP_LLM_LIB_PATH}:${containerLD_LIBRARY_PATH}"
        }
        containerLD_LIBRARY_PATH = containerLD_LIBRARY_PATH.replaceAll(':+$', '')
        withEnv(["LD_LIBRARY_PATH=${containerLD_LIBRARY_PATH}"]) {
            withCredentials([
                usernamePassword(
                    credentialsId: 'svc_tensorrt_gitlab_read_api_token',
                    usernameVariable: 'GITLAB_API_USER',
                    passwordVariable: 'GITLAB_API_TOKEN'
                ),
                string(credentialsId: 'llm_evaltool_repo_url', variable: 'EVALTOOL_REPO_URL')
            ]) {
                sh "env | sort"
                trtllm_utils.llmExecStepWithRetry(
                    pipeline,
                    numRetries: 1,
                    script: """
                        rm -rf ${stageName}/ && \
                        cd ${llmSrc}/tests/integration/defs && \
                        ${testCmdLine.join(" ")}
                    """,
                    retryLog: "stageName = ${stageName}, HOST_NODE_NAME = ${env.HOST_NODE_NAME}"
                )
            }
        }

        if (perfMode) {
            stage("Check perf result") {
                sh """
                    python3 ${llmSrc}/tests/integration/defs/perf/sanity_perf_check.py \
                    ${stageName}/perf_script_test_results.csv \
                    ${llmSrc}/tests/integration/defs/perf/base_perf.csv
                """
            }
        }
    }
}


def runLLMTestlistOnPlatform(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312")
{
    cacheErrorAndUploadResult(stageName, {
        runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver)
    }, {
        if (testFilter[(DEBUG_MODE)]) {
            try {
                timeout(time: 2, unit: 'HOURS') {
                    input message: "Pause 2 hours for Post-Debug. Please press the button to proceed when you finish debugging."
                }
            } catch (InterruptedException e) {
                echo "Post-debug session was interrupted by user or timeout"
                currentBuild.result = 'ABORTED'
                error("Pipeline aborted during post-debug session")
            } catch (Exception e) {
                echo "An error occurred during post-debug session: ${e.message}"
                currentBuild.result = 'FAILURE'
                error("Error in post-debug session: ${e.message}")
            }
        }
        def llmPath = sh (script: "realpath .", returnStdout: true).trim()
        def llmSrc = "${llmPath}/${LLM_ROOT}${config}/TensorRT-LLM/src"
        // CPP tests will generate test result in ${llmSrc}/cpp/build_backup/, move these files to job result folder
        sh "ls -all ${llmSrc}/cpp/build_backup/ || true"
        sh "ls -all ${llmSrc}/cpp/build/ || true"
        // Sed for CPP test result
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/\" classname=\"/\" classname=\"${stageName}./g' *.xml || true"
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/testsuite name=\"[^\"]*\"/testsuite name=\"${stageName}\"/g' *.xml || true"
        // Sed for Pytest result
        sh "ls ${stageName}/ -all"
        sh "cd ${stageName} && sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' *.xml || true"
        // Copy CPP test result
        sh "cp ${llmSrc}/cpp/build_backup/*.xml ${stageName} || true"
        sh "ls ${stageName}/ -all"
    })
}


def checkPipInstall(pipeline, wheel_path)
{
    def wheelArtifactLinks = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/${wheel_path}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT}/tests/unittest && python3 test_pip_install.py --wheel_path ${wheelArtifactLinks}")
}


def runLLMBuildFromPackage(pipeline, cpu_arch, reinstall_dependencies=false, wheel_path="", cpver="cp312")
{
    def pkgUrl = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${linuxPkgName}"

    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"
    sh "ccache -sv"
    sh "cat ${CCACHE_DIR}/ccache.conf"
    sh "bash -c 'pip3 show tensorrt || true'"

    // If the image is pre-installed with cxx11-abi pytorch, using non-cxx11-abi requires reinstallation.
    if (reinstall_dependencies == true) {
        sh "#!/bin/bash \n" + "pip3 uninstall -y torch"
        sh "#!/bin/bash \n" + "yum remove -y libcudnn*"
    }
    sh "pwd && ls -alh"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv ${pkgUrl}")

    sh "env | sort"
    sh "tar -zvxf ${linuxPkgName}"

    // Check for prohibited files in the package
    sh '''
        echo "Checking prohibited files..."
        FAILED=0

        # Folders and their allowed files
        declare -A ALLOWED=(
            ["./tensorrt_llm/cpp/tensorrt_llm/kernels/internal_cutlass_kernels/src"]=""
            ["./tensorrt_llm/cpp/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/decoderXQAImplJIT/nvrtcWrapper/src"]=""
        )

        for DIR in "${!ALLOWED[@]}"; do
            [ -d "$DIR" ] || continue

            # File check
            ALLOWED_FILE="$DIR/${ALLOWED[$DIR]}"
            if [ -z "${ALLOWED[$DIR]}" ]; then
                FILES=$(find "$DIR" -type f)
            else
                FILES=$(find "$DIR" -type f ! -path "$ALLOWED_FILE")
            fi

            # Subdir check
            SUBDIRS=$(find "$DIR" -mindepth 1 -type d)

            # Error reporting
            if [ -n "$FILES$SUBDIRS" ]; then
                echo "ERROR in $DIR:"
                [ -n "$FILES" ] && echo "Prohibited files:\n$FILES"
                [ -n "$SUBDIRS" ] && echo "Prohibited subdirs:\n$SUBDIRS"
                FAILED=1
            fi

            # Verify allowed file exists
            if [ -n "${ALLOWED[$DIR]}" ] && [ ! -f "$ALLOWED_FILE" ]; then
                echo "WARNING: Missing $ALLOWED_FILE"
            fi
        done

        [ $FAILED -eq 0 ] || { echo "Build failed: Prohibited content found"; exit 1; }
        echo "No prohibited files found"
    '''

    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && pip3 install -r requirements-dev.txt")
    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
    }
    buildArgs = "--clean"
    if (cpu_arch == AARCH64_TRIPLE) {
        buildArgs = "-a '90-real;100-real;120-real'"
    } else if  (reinstall_dependencies == true) {
        buildArgs = "-a '80-real;86-real;89-real;90-real'"
    }
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && python3 scripts/build_wheel.py --use_ccache -j ${BUILD_JOBS} -D 'WARNING_IS_ERROR=ON' ${buildArgs}")
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    def wheelName = sh(returnStdout: true, script: 'cd tensorrt_llm/build && ls -1 *.whl').trim()
    echo "uploading ${wheelName} to ${cpu_arch}/${wheel_path}"
    trtllm_utils.uploadArtifacts("tensorrt_llm/build/${wheelName}",  "${UPLOAD_PATH}/${cpu_arch}/${wheel_path}")

    if (reinstall_dependencies == true) {
        // Test installation in the new environment
        def pip_keep = "-e 'pip'"
        def remove_trt = "rm -rf /usr/local/tensorrt"
        if (env.alternativeTRT) {
            pip_keep += " -e tensorrt"
            remove_trt = "echo keep /usr/local/tensorrt"
        }
        sh "#!/bin/bash \n" + "pip3 list --format=freeze | egrep -v ${pip_keep} | xargs pip3 uninstall -y"
        sh "#!/bin/bash \n" + "yum remove -y libcudnn* libnccl* libcublas* && ${remove_trt}"
    }
    // Test preview installation
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && pip3 install pytest build/tensorrt_llm-*.whl")
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    return wheelName
}


def runPackageSanityCheck(pipeline, wheel_path, reinstall_dependencies=false, cpver="cp312")
{
    def whlUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/${wheel_path}"

    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"
    sh "bash -c 'pip3 show tensorrt || true'"
    sh "cat ${MODEL_CACHE_DIR}/README"
    sh "nvidia-smi -q"

    sh "pwd && ls -alh"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv ${whlUrl}")

    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
        sh "bash -c 'pip3 show tensorrt || true'"
    }
    if (reinstall_dependencies) {
        // Test installation in the new environment
        def pip_keep = "-e 'pip'"
        def remove_trt = "rm -rf /usr/local/tensorrt"
        if (env.alternativeTRT) {
            pip_keep += " -e tensorrt"
            remove_trt = "echo keep /usr/local/tensorrt"
        }
        sh "bash -c 'pip3 list --format=freeze | egrep -v ${pip_keep} | xargs pip3 uninstall -y'"
        sh "bash -c 'yum remove -y libcudnn* libnccl* libcublas* && ${remove_trt}'"
    }
    // Test preview installation
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'pip3 install pytest tensorrt_llm-*.whl'")
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    def pkgUrl = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${linuxPkgName}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv ${pkgUrl}")
    sh "tar -zvxf ${linuxPkgName}"

    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core/gpt && python3 ../../../generate_checkpoint_config.py --architecture GPTForCausalLM --dtype float16'")
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core//gpt && trtllm-build --model_config config.json --log_level verbose'")
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core/gpt && python3 ../../../run.py --max_output_len 4 --end_id -1'")
}

def checkStageNameSet(stageNames, jobKeys, paramName) {
    echo "Validate stage names for the passed GitLab bot params [${paramName}]."
    invalidStageName = stageNames.findAll { !(it in jobKeys) }
    if (invalidStageName) {
        throw new Exception("Cannot find the stage names [${invalidStageName}] from the passed params [${paramName}].")
    }
}

def checkStageName(stageNames) {
    invalidStageName = stageNames.findAll { !(it ==~ /[-\+\w\[\]]+/) }
    if (invalidStageName) {
        throw new Exception("Invalid stage name: [${invalidStageName}], we only support chars '-+_[]0-9a-zA-Z' .")
    }
}

def runInDockerOnNode(image, label, dockerArgs)
{
    return {
        stageName, runner -> stage(stageName) {
            node(label) {
                deleteDir()
                docker.image(image).inside(dockerArgs) {
                    runner()
                }
            }
        }
    }
}

def runInKubernetes(pipeline, podSpec, containerName)
{
    return {
        stageName, runner -> stage(stageName) {
            trtllm_utils.launchKubernetesPod(pipeline, podSpec, containerName) {
                echoNodeAndGpuInfo(pipeline, stageName)
                runner()
            }
        }
    }
}

def launchTestJobs(pipeline, testFilter, dockerNode=null)
{
    def dockerArgs = "-v /mnt/scratch.trt_llm_data:/scratch.trt_llm_data:ro -v /tmp/ccache:${CCACHE_DIR}:rw -v /tmp/pipcache/http-v2:/root/.cache/pip/http-v2:rw --cap-add syslog"
    turtleConfigs = [
        "DGX_H100-4_GPUs-PyTorch-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 2, 4],
        "DGX_H100-4_GPUs-PyTorch-2": ["dgx-h100-x4", "l0_dgx_h100", 2, 2, 4],
        "DGX_H100-4_GPUs-CPP-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-TensorRT-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 2, 4],
        "DGX_H100-4_GPUs-TensorRT-2": ["dgx-h100-x4", "l0_dgx_h100", 2, 2, 4],
        "A10-PyTorch-1": ["a10", "l0_a10", 1, 1],
        "A10-CPP-1": ["a10", "l0_a10", 1, 1],
        "A10-TensorRT-1": ["a10", "l0_a10", 1, 6],
        "A10-TensorRT-2": ["a10", "l0_a10", 2, 6],
        "A10-TensorRT-3": ["a10", "l0_a10", 3, 6],
        "A10-TensorRT-4": ["a10", "l0_a10", 4, 6],
        "A10-TensorRT-5": ["a10", "l0_a10", 5, 6],
        "A10-TensorRT-6": ["a10", "l0_a10", 6, 6],
        "A30-PyTorch-1": ["a30", "l0_a30", 1, 2],
        "A30-PyTorch-2": ["a30", "l0_a30", 2, 2],
        "A30-CPP-1": ["a30", "l0_a30", 1, 2],
        "A30-CPP-2": ["a30", "l0_a30", 2, 2],
        "A30-TensorRT-1": ["a30", "l0_a30", 1, 4],
        "A30-TensorRT-2": ["a30", "l0_a30", 2, 4],
        "A30-TensorRT-3": ["a30", "l0_a30", 3, 4],
        "A30-TensorRT-4": ["a30", "l0_a30", 4, 4],
        "A100X-TensorRT-1": ["a100x", "l0_a100", 1, 4],
        "A100X-TensorRT-2": ["a100x", "l0_a100", 2, 4],
        "A100X-TensorRT-3": ["a100x", "l0_a100", 3, 4],
        "A100X-TensorRT-4": ["a100x", "l0_a100", 4, 4],
        "L40S-PyTorch-1": ["l40s", "l0_l40s", 1, 1],
        "L40S-TensorRT-1": ["l40s", "l0_l40s", 1, 3],
        "L40S-TensorRT-2": ["l40s", "l0_l40s", 2, 3],
        "L40S-TensorRT-3": ["l40s", "l0_l40s", 3, 3],
        "H100_PCIe-PyTorch-1": ["h100-cr", "l0_h100", 1, 2],
        "H100_PCIe-PyTorch-2": ["h100-cr", "l0_h100", 2, 2],
        "H100_PCIe-CPP-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-TensorRT-1": ["h100-cr", "l0_h100", 1, 5],
        "H100_PCIe-TensorRT-2": ["h100-cr", "l0_h100", 2, 5],
        "H100_PCIe-TensorRT-3": ["h100-cr", "l0_h100", 3, 5],
        "H100_PCIe-TensorRT-4": ["h100-cr", "l0_h100", 4, 5],
        "H100_PCIe-TensorRT-5": ["h100-cr", "l0_h100", 5, 5],
        "B200_PCIe-PyTorch-1": ["b100-ts2", "l0_b200", 1, 2],
        "B200_PCIe-PyTorch-2": ["b100-ts2", "l0_b200", 2, 2],
        "B200_PCIe-TensorRT-1": ["b100-ts2", "l0_b200", 1, 2],
        "B200_PCIe-TensorRT-2": ["b100-ts2", "l0_b200", 2, 2],
        // Currently post-merge test stages only run tests with "stage: post_merge" mako
        // in the test-db. This behavior may change in the future.
        "A10-TensorRT-[Post-Merge]-1": ["a10", "l0_a10", 1, 2],
        "A10-TensorRT-[Post-Merge]-2": ["a10", "l0_a10", 2, 2],
        "A30-TensorRT-[Post-Merge]-1": ["a30", "l0_a30", 1, 2],
        "A30-TensorRT-[Post-Merge]-2": ["a30", "l0_a30", 2, 2],
        "A100X-TensorRT-[Post-Merge]-1": ["a100x", "l0_a100", 1, 2],
        "A100X-TensorRT-[Post-Merge]-2": ["a100x", "l0_a100", 2, 2],
        "L40S-TensorRT-[Post-Merge]-1": ["l40s", "l0_l40s", 1, 2],
        "L40S-TensorRT-[Post-Merge]-2": ["l40s", "l0_l40s", 2, 2],
        "H100_PCIe-PyTorch-[Post-Merge]-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-CPP-[Post-Merge]-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-TensorRT-[Post-Merge]-1": ["h100-cr", "l0_h100", 1, 2],
        "H100_PCIe-TensorRT-[Post-Merge]-2": ["h100-cr", "l0_h100", 2, 2],
        "DGX_H100-4_GPUs-PyTorch-[Post-Merge]": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-TensorRT-[Post-Merge]": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "A100_80GB_PCIE-TensorRT-Perf": ["a100-80gb-pcie", "l0_perf", 1, 1],
        "H100_PCIe-TensorRT-Perf": ["h100-cr", "l0_perf", 1, 1],
        "DGX_H200-8_GPUs-PyTorch-[Post-Merge]": ["dgx-h200-x8", "l0_dgx_h200", 1, 1, 8],
    ]

    parallelJobs = turtleConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "amd64", values[4] ?: 1, key.contains("Perf")), {
        def config = VANILLA_CONFIG
        if (key.contains("single-device")) {
            config = SINGLE_DEVICE_CONFIG
        }
        if (key.contains("llvm")) {
            config = LLVM_CONFIG
        }
        runLLMTestlistOnPlatform(pipeline, values[0], values[1], config, key.contains("Perf"), key, values[2], values[3])
    }]]}

    fullSet = parallelJobs.keySet()

    // Try to match what are being tested on x86 H100_PCIe.
    // The total machine time is scaled proportionally according to the number of each GPU.
    aarch64Configs = [
        "GH200-1": ["gh200", "l0_gh200", 1, 2],
        "GH200-2": ["gh200", "l0_gh200", 2, 2],
        "GH200-[Post-Merge]": ["gh200", "l0_gh200", 1, 1],
    ]

    fullSet += aarch64Configs.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        parallelJobs = aarch64Configs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "arm64"), {
            runLLMTestlistOnPlatform(pipeline, values[0], values[1], LINUX_AARCH64_CONFIG, false, key, values[2], values[3])
        }]]}
    }


    docBuildSpec = createKubernetesPodConfig(LLM_DOCKER_IMAGE, "a10")
    docBuildConfigs = [
        "A10-Build_TRT-LLM_Doc": [docBuildSpec, {
            sh "rm -rf **/*.xml *.tar.gz"
            runLLMDocBuild(pipeline, config=VANILLA_CONFIG)
        }],
    ]

    fullSet += docBuildConfigs.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        docBuildConfigs = [:]
    }

    docBuildJobs = docBuildConfigs.collectEntries{key, values -> [key, [values[0], {
        stage("[${key}] Run") {
            cacheErrorAndUploadResult("${key}", values[1], {}, true)
        }
    }]]}

    sanityCheckConfigs = [
        "DLFW": [
            LLM_DOCKER_IMAGE,
            "B200_PCIe",
            X86_64_TRIPLE,
            false,
            "cxx11/",
            DLFW_IMAGE,
        ],
        "manylinux-py310": [
            LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE,
            "A10",
            X86_64_TRIPLE,
            true,
            "",
            UBUNTU_22_04_IMAGE,
        ],
        "manylinux-py312": [
            LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE,
            "A10",
            X86_64_TRIPLE,
            true,
            "",
            UBUNTU_24_04_IMAGE,
        ],
    ]

    def toStageName = { gpuType, key -> "${gpuType}-PackageSanityCheck-${key}".toString() }

    fullSet += sanityCheckConfigs.collectEntries{ key, values -> [toStageName(values[1], key), null] }.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        sanityCheckConfigs = [
            "DLFW": [
                LLM_DOCKER_IMAGE,
                "GH200",
                AARCH64_TRIPLE,
                false,
                "",
                // TODO: Change to UBUNTU_24_04_IMAGE after https://nvbugs/5161461 is fixed
                DLFW_IMAGE,
            ],
        ]
    }

    fullSet += [toStageName("GH200", "DLFW")]

    sanityCheckJobs = sanityCheckConfigs.collectEntries {key, values -> [toStageName(values[1], key), {
        cacheErrorAndUploadResult(toStageName(values[1], key), {
            def cpu_arch = values[2]
            def gpu_type = values[1].toLowerCase()
            if (values[1] == "B200_PCIe") {
                gpu_type = "b100-ts2"
            }

            def k8s_arch = "amd64"
            if (cpu_arch == AARCH64_TRIPLE) {
                k8s_arch = "arm64"
            }

            def buildSpec = createKubernetesPodConfig(values[0], "build", k8s_arch)
            def buildRunner = runInKubernetes(pipeline, buildSpec, "trt-llm")
            def sanityRunner = null

            if (dockerNode) {
                sanityRunner = runInDockerOnNode(values[0], dockerNode, dockerArgs)
            } else {
                def sanitySpec = createKubernetesPodConfig(values[0], gpu_type, k8s_arch)
                sanityRunner = runInKubernetes(pipeline, sanitySpec, "trt-llm")
            }

            def wheelPath = "${values[4]}"
            def wheelName = ""
            def cpver = "cp312"
            def pyver = "3.12"
            if (key.contains("py310")) {
                cpver = "cp310"
                pyver = "3.10"
            }

            buildRunner("[${toStageName(values[1], key)}] Build") {
                def env = []
                if (key.contains("manylinux")) {
                    env = ["LD_LIBRARY_PATH+=:/usr/local/cuda/compat"]
                }
                withEnv(env) {
                    wheelName = runLLMBuildFromPackage(pipeline, cpu_arch, values[3], wheelPath, cpver)
                }
            }

            def fullWheelPath = "${cpu_arch}/${wheelPath}${wheelName}"

            sanityRunner("Sanity check") {
                runPackageSanityCheck(pipeline, fullWheelPath, values[3], cpver)
            }

            def checkPipStage = false
            if (cpu_arch == X86_64_TRIPLE) {
                checkPipStage = true
            } else if (cpu_arch == AARCH64_TRIPLE) {
                checkPipStage = true
            }

            if (checkPipStage) {
                stage("Run LLMAPI tests") {
                    pipInstallSanitySpec = createKubernetesPodConfig(values[5], gpu_type, k8s_arch)
                    trtllm_utils.launchKubernetesPod(pipeline, pipInstallSanitySpec, "trt-llm", {
                        echo "###### Prerequisites Start ######"
                        // Clean up the pip constraint file from the base NGC PyTorch image.
                        if (values[5] == DLFW_IMAGE) {
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt || true")
                        }
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get -y install python3-pip git rsync curl")
                        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 config set global.break-system-packages true")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install requests")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 uninstall -y tensorrt")
                        if ((values[5] != DLFW_IMAGE) && (cpu_arch == AARCH64_TRIPLE)) {
                            echo "###### Extra prerequisites on aarch64 Start ######"
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
                        }
                        def libEnv = []
                        if (env.alternativeTRT) {
                            stage("Replace TensorRT") {
                                trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
                            }
                            libEnv += ["LD_LIBRARY_PATH+tensorrt=/usr/local/tensorrt/lib"]
                            libEnv += ["LD_LIBRARY_PATH+nvrtc=/usr/local/lib/python${pyver}/dist-packages/nvidia/cuda_nvrtc/lib"]
                        }
                        echo "###### Check pip install Start ######"
                        withEnv(libEnv) {
                            sh "env | sort"
                            checkPipInstall(pipeline, "${cpu_arch}/${wheelPath}")
                        }
                        echo "###### Run LLMAPI tests Start ######"
                        def config = VANILLA_CONFIG
                        if (cpu_arch == AARCH64_TRIPLE) {
                            config = LINUX_AARCH64_CONFIG
                        }
                        withEnv(libEnv) {
                            sh "env | sort"
                            runLLMTestlistOnPlatform(pipeline, gpu_type, "l0_sanity_check", config, false, "${values[1]}-${key}-sanity-check" , 1, 1, true, null)
                        }
                    })
                }
            }
        }, {}, true)
    }]}

    multiGpuJobs = parallelJobs.findAll{(it.key.contains("4_GPUs") || it.key.contains("8_GPUs")) && !it.key.contains("Post-Merge")}
    println multiGpuJobs.keySet()

    parallelJobs += docBuildJobs
    parallelJobs += sanityCheckJobs

    postMergeJobs = parallelJobs.findAll {it.key.contains("Post-Merge")}

    // Start as a normal pre-merge job
    parallelJobsFiltered = parallelJobs - multiGpuJobs - postMergeJobs

    // Check if the multi GPU related file has changed or not. If changed, add multi GPU test stages.
    if (testFilter[(MULTI_GPU_FILE_CHANGED)]) {
        parallelJobsFiltered += multiGpuJobs
    }

    // Check --post-merge, post-merge or TRT dependency testing pipelines.
    // If true, add post-merge only test stages and multi-GPU test stages.
    if (env.alternativeTRT || testFilter[(IS_POST_MERGE)]) {
        parallelJobsFiltered += multiGpuJobs
        parallelJobsFiltered += postMergeJobs
    }

    // Check --skip-test, only run doc build and sanity check stages.
    if (testFilter[(ENABLE_SKIP_TEST)]) {
        echo "All test stages are skipped."
        parallelJobsFiltered = docBuildJobs + sanityCheckJobs
    }

    // Check --add-multi-gpu-test, if true, add multi-GPU test stages back.
    if (testFilter[(ADD_MULTI_GPU_TEST)]) {
        parallelJobsFiltered += multiGpuJobs
    }

    // Check --only-multi-gpu-test, if true, only run multi-GPU test stages.
    if (testFilter[(ONLY_MULTI_GPU_TEST)]) {
        parallelJobsFiltered = multiGpuJobs
    }

    // Check --disable-multi-gpu-test, if true, remove multi-GPU test stages.
    if (testFilter[(DISABLE_MULTI_GPU_TEST)]) {
        parallelJobsFiltered -= multiGpuJobs
    }

    // Check --gpu-type, filter test stages.
    if (testFilter[(GPU_TYPE_LIST)] != null) {
        echo "Use GPU_TYPE_LIST for filtering."
        parallelJobsFiltered = parallelJobsFiltered.findAll {it.key.tokenize('-')[0] in testFilter[(GPU_TYPE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    if (testFilter[(ONLY_PYTORCH_FILE_CHANGED)]) {
        echo "ONLY_PYTORCH_FILE_CHANGED mode is true."
        parallelJobsFiltered = parallelJobsFiltered.findAll { !it.key.contains("-CPP-") && !it.key.contains("-TensorRT-") }
        println parallelJobsFiltered.keySet()
    }

    // Check --stage-list, only run the stages in stage-list.
    if (testFilter[TEST_STAGE_LIST] != null) {
        echo "Use TEST_STAGE_LIST for filtering."
        parallelJobsFiltered = parallelJobs.findAll {it.key in testFilter[(TEST_STAGE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    // Check --extra-stage, add the stages in extra-stage.
    if (testFilter[EXTRA_STAGE_LIST] != null) {
        echo "Use EXTRA_STAGE_LIST for filtering."
        parallelJobsFiltered += parallelJobs.findAll {it.key in testFilter[(EXTRA_STAGE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    checkStageName(fullSet)

    if (testFilter[(TEST_STAGE_LIST)] != null) {
        checkStageNameSet(testFilter[(TEST_STAGE_LIST)], fullSet, TEST_STAGE_LIST)
    }
    if (testFilter[(EXTRA_STAGE_LIST)] != null) {
        checkStageNameSet(testFilter[(EXTRA_STAGE_LIST)], fullSet, EXTRA_STAGE_LIST)
    }

    echo "Check the passed GitLab bot testFilter parameters."
    def keysStr = parallelJobsFiltered.keySet().join(",\n")
    pipeline.echo "Now we will run stages: [\n${keysStr}\n]"

    parallelJobsFiltered = parallelJobsFiltered.collectEntries { key, values -> [key, {
        stage(key) {
            if (key in testFilter[REUSE_STAGE_LIST]) {
                stage("Skip - reused") {
                    echo "Skip - Passed in the last pipeline."
                }
            } else if (values instanceof List && dockerNode == null) {
                trtllm_utils.launchKubernetesPod(pipeline, values[0], "trt-llm", {
                    values[1]()
                })
            } else if (values instanceof List && dockerNode != null) {
                node(dockerNode) {
                    deleteDir()
                    docker.image(LLM_DOCKER_IMAGE).inside(dockerArgs) {
                        values[1]()
                    }
                }
            } else {
                values()
            }
        }
    }]}

    return parallelJobsFiltered
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig("", "agent")
    }
    options {
        // Check the valid options at: https://www.jenkins.io/doc/book/pipeline/syntax/
        // some step like results analysis stage, does not need to check out source code
        skipDefaultCheckout()
        // to better analyze the time for each step/test
        timestamps()
        timeout(time: 24, unit: 'HOURS')
    }
    environment {
        //Workspace normally is: /home/jenkins/agent/workspace/LLM/L0_MergeRequest@tmp/
        HF_HOME="${env.WORKSPACE_TMP}/.cache/huggingface"
        CCACHE_DIR="${CCACHE_DIR}"
        PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
        // force datasets to be offline mode, to prevent CI jobs are downloading HF dataset causing test failures
        HF_DATASETS_OFFLINE=1
    }
    stages {
        stage("Setup environment")
        {
            steps
            {
                script {
                    echo "enableFailFast is: ${params.enableFailFast}"
                    echo "env.testFilter is: ${env.testFilter}"
                    if (env.testFilter)
                    {
                        def mp = readJSON text: env.testFilter, returnPojo: true
                        mp.each {
                            if (testFilter.containsKey(it.key)) {
                                echo "setting ${it.key} = ${it.value}"
                                testFilter[it.key] = it.value
                            }
                        }
                    }
                    println testFilter
                }
            }
        }
        stage("Check Test Lists")
        {
            when {
                expression {
                    env.targetArch == X86_64_TRIPLE  // Only execute the check if running on x86
                }
            }
            steps
            {
                script {
                    launchTestListCheck(this)
                }
            }
        }
        stage("Test") {
            steps {
                script {
                    parallelJobs = launchTestJobs(this, testFilter)

                    singleGpuJobs = parallelJobs
                    dgxJobs = [:]

                    def testPhase2StageName = env.testPhase2StageName
                    if (testPhase2StageName) {
                        def dgxSigns = ["DGX_H100", "DGX_H200"]
                        singleGpuJobs = parallelJobs.findAll{!dgxSigns.any{sign -> it.key.contains(sign)}}
                        dgxJobs = parallelJobs.findAll{dgxSigns.any{sign -> it.key.contains(sign)}}
                    }

                    if (singleGpuJobs.size() > 0) {
                        singleGpuJobs.failFast = params.enableFailFast
                        parallel singleGpuJobs
                    } else {
                        echo "Skip single-GPU testing. No test to run."
                    }

                    if (dgxJobs.size() > 0) {
                        stage(testPhase2StageName) {
                            dgxJobs.failFast = params.enableFailFast
                            parallel dgxJobs
                        }
                    }
                }
            }
        } // Test stage
    } // stages
} // pipeline
