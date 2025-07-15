@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.InterruptedException
import groovy.transform.Field
import groovy.json.JsonSlurper
import groovy.json.JsonOutput
import com.nvidia.bloom.KubernetesManager
import com.nvidia.bloom.Constants
import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.KubernetesManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.SlurmCluster
import com.nvidia.bloom.SlurmPartition
import com.nvidia.bloom.Utils
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
LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE = env.wheelDockerImagePy310
LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE = env.wheelDockerImagePy312

// DLFW torch image
DLFW_IMAGE = "nvcr.io/nvidia/pytorch:25.05-py3"

//Ubuntu base image
UBUNTU_22_04_IMAGE = "urm.nvidia.com/docker/ubuntu:22.04"
UBUNTU_24_04_IMAGE = "urm.nvidia.com/docker/ubuntu:24.04"

POD_TIMEOUT_SECONDS = env.podTimeoutSeconds ? env.podTimeoutSeconds : "21600"
POD_TIMEOUT_SECONDS_TMP = env.podTimeoutSeconds ? env.podTimeoutSeconds : "43200"

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

SLURM_CORES_REQUEST = "1"
SLURM_CORES_LIMIT = "1"
SLURM_MEMORY_REQUEST = "8Gi"
SLURM_MEMORY_LIMIT = "12Gi"

TESTER_CORES = "12"
TESTER_MEMORY = "96Gi"

CCACHE_DIR="/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"
MODEL_CACHE_DIR="/scratch.trt_llm_data/llm-models"

def uploadResults(def pipeline, SlurmCluster cluster, String nodeName, String stageName){
    withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        def remote = [
            ip           : cluster.ip,
            host         : cluster.host,
            user         : "${pipeline.USERNAME}",
            passwd       : "${pipeline.PASSWORD}",
            allowAnyHosts: true,
        ]

        Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")
        pipeline.stage('Submit Test Results') {
            sh "mkdir -p ${stageName}"
            def resultsFilePath = "/home/svc_tensorrt/bloom/scripts/${nodeName}/results/results.xml"
            def downloadResultCmd = "sshpass -p '${remote.passwd}' scp -r -p -oStrictHostKeyChecking=no ${remote.user}@${remote.host}:${resultsFilePath} ${stageName}/"
            def downloadSucceed = sh(script: downloadResultCmd, returnStatus: true) == 0
            if (downloadSucceed) {
                sh "ls ${stageName}"
                echo "Upload test results."
                sh "tar -czvf results-${stageName}.tar.gz ${stageName}/"
                trtllm_utils.uploadArtifacts(
                    "results-${stageName}.tar.gz",
                    "${UPLOAD_PATH}/test-results/"
                )
                junit(testResults: "${stageName}/results*.xml")
            } else {
                println("No results xml to submit")
            }
        }
    }
}

//TODO: consolidate slurm related code for both multi nodes and single nodes
def cleanUpNodeResourcesMultiNodes(def pipeline, SlurmCluster cluster, String jobUID){
    withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        def remote = [
            ip           : cluster.ip,
            host         : cluster.host,
            user         : "${pipeline.USERNAME}",
            passwd       : "${pipeline.PASSWORD}",
            allowAnyHosts: true,
        ]

        Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")
        pipeline.stage('Clean up SLURM Agent Resources') {
            Utils.exec(
                pipeline,
                timeout: false,
                script: Utils.sshUserCmd(
                    remote,
                    "rm -rf /home/svc_tensorrt/bloom/scripts/${jobUID}"
                )
            )
        }
    }
}

def cleanUpNodeResources(def pipeline, SlurmCluster cluster, String nodeName){
    withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        def remote = [
            ip           : cluster.ip,
            host         : cluster.host,
            user         : "${pipeline.USERNAME}",
            passwd       : "${pipeline.PASSWORD}",
            allowAnyHosts: true,
        ]

        Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")
        pipeline.stage('Clean up SLURM Agent Resources') {
            Utils.exec(
                pipeline,
                timeout: false,
                script: Utils.sshUserCmd(
                    remote,
                    "rm -rf /home/svc_tensorrt/bloom/scripts/agent-${nodeName}.jar /home/svc_tensorrt/bloom/scripts/${nodeName}-slurm_jenkins_agent_setup.sh"
                )
            )
            Utils.exec(pipeline, script: "echo done")
        }
    }
}

def executeLLMTestOnSlurm(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312", runner)
{
    runner {
        // TODO: refactor the finallyRunner to reuse within slurm or nonslurm job.
        cacheErrorAndUploadResult(stageName, {
            runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver)
        }, {
            // If the execution test list is null, remove the test result xml
            sh """
                ls -all ${stageName}/
                if ! grep -q '<testcase' ${stageName}/results.xml; then
                    rm ${stageName}/results.xml || true
                fi
            """
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def llmSrc = "${llmPath}/${LLM_ROOT}${config}/TensorRT-LLM/src"
            // CPP tests will generate test result in ${llmSrc}/cpp/build_backup/, move these files to job result folder
            sh "ls -all ${llmSrc}/cpp/build_backup/ || true"
            sh "ls -all ${llmSrc}/cpp/build/ || true"
            // Sed for CPP test result
            sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/\" classname=\"/\" classname=\"${stageName}./g' *.xml || true"
            sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/testsuite name=\"[^\"]*\"/testsuite name=\"${stageName}\"/g' *.xml || true"
            // Sed for Pytest result
            sh "cd ${stageName} && sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' *.xml || true"
            // Copy CPP test result
            sh "cp ${llmSrc}/cpp/build_backup/*.xml ${stageName} || true"
            sh "ls ${stageName}/ -all"
        })
    }
}

def runLLMTestlistOnSlurm(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, skipInstallWheel=false, cpver="cp312")
{
    SlurmPartition partition = SlurmConfig.partitionConfig[platform] as SlurmPartition
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]

    def nodeName = "${cluster.host}-test-${UUID.randomUUID().toString()}"
    def nodeSecret = CloudManager.createNode(nodeName)

    try {
        // Run ssh command to start node in desired cluster via SLURM
        withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
            def remote = [
                    ip           : cluster.ip,
                    host         : cluster.host,
                    user         : "${pipeline.USERNAME}",
                    passwd       : "${pipeline.PASSWORD}",
                    allowAnyHosts: true,
            ]

            Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")
            stage('Request Node via SLURM') {
                println("Selected Cluster: ${cluster.name}")

                def jenkinsSetupPath = Utils.copyLibraryResource(pipeline, "slurm_jenkins_agent_setup.sh")

                Utils.exec(pipeline, script: "chmod +x ${jenkinsSetupPath}", returnStdout: true)

                Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' scp -r -p -oStrictHostKeyChecking=no ${jenkinsSetupPath} ${remote.user}@${remote.host}:~/bloom/scripts/${nodeName}-slurm_jenkins_agent_setup.sh",)

                Utils.exec(
                    pipeline,
                    timeout: false,
                    script: Utils.sshUserCmd(
                            remote,
                            """${SlurmConfig.generateCommand(cluster, partition, nodeSecret, nodeName, Jenkins.instance.rootUrl)}"""
                    )
                )
                Utils.exec(pipeline, script: "echo Sleeping to allow agent initialization; sleep 30")
            }
        }

        stage('Checking if the Node is Online') {
            def counter = 0
            while (!CloudManager.isNodeOnline(nodeName) && counter < 12) {
                sleep(time: 10, unit: 'MINUTES')  // Wait 10 minutes to check status of the node again
                counter++
            }

            if (CloudManager.isNodeOnline(nodeName)) {
                def dockerArgs = "--gpus ${gpuCount} --cap-add=SYS_ADMIN --ipc=host --security-opt seccomp=unconfined  -u root:root -v /home/scratch.trt_llm_data:/scratch.trt_llm_data:ro -v /tmp/ccache:${CCACHE_DIR}:rw -v /tmp/pipcache/http-v2:/root/.cache/pip/http-v2:rw --cap-add syslog"
                slurmRunner = runInDockerOnNodeMultiStage(LLM_DOCKER_IMAGE, nodeName, dockerArgs, false)
                executeLLMTestOnSlurm(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver, slurmRunner)
            } else {
                echo "The node does not come online in 2 hours, terminating the job"
            }
        }
    } finally {
        cleanUpNodeResources(pipeline, cluster, nodeName)
        CloudManager.destroyNode(nodeName)
    }
}

def getNodeArgs(int nodeCount, int gpuCount) {
    int gpusPerNode = ((gpuCount / nodeCount) as BigDecimal).setScale(0, BigDecimal.ROUND_CEILING).intValue()
    return [
        "--nodes=${nodeCount}",
        "--ntasks=${gpuCount}",
        "--ntasks-per-node=${gpusPerNode}",
        "--gpus-per-node=${gpusPerNode}",
    ].join(" ")
}

def runLLMTestlistOnSlurm_MultiNodes(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, nodeCount=2, skipInstallWheel=false, cpver="cp312")
{
    SlurmPartition partition = SlurmConfig.partitionConfig[platform] as SlurmPartition
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]

    def jobUID = "${cluster.host}-multi_node_test-${UUID.randomUUID().toString()}"

    try {
        // Run ssh command to start node in desired cluster via SLURM
        withCredentials([
            usernamePassword(
                credentialsId: 'svc_tensorrt',
                usernameVariable: 'USERNAME',
                passwordVariable: 'PASSWORD'
            )
        ]) {
            def remote = [
                    ip           : cluster.ip,
                    host         : cluster.host,
                    user         : "${pipeline.USERNAME}",
                    passwd       : "${pipeline.PASSWORD}",
                    allowAnyHosts: true,
            ]
            Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")
            def tarName = BUILD_CONFIGS[config][TARNAME]
            def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def jobWorkspace = "/home/svc_tensorrt/bloom/scripts/${jobUID}"
            def resourcePathNode = "/tmp"
            def llmSrcNode = "${resourcePathNode}/TensorRT-LLM/src"
            def llmSrcLocal = "${llmPath}/TensorRT-LLM/src"
            def scriptRunNode = "${jobWorkspace}/slurm_run.sh"
            def testListPathNode = "${jobWorkspace}/${testList}.txt"
            def isAarch64 = config.contains("aarch64")
            def pytestTestTimeout = "7200"

            stage('Prepare Testing') {
                // Create Job Workspace folder in Frontend Node
                Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' ssh -oStrictHostKeyChecking=no ${remote.user}@${remote.host} 'mkdir ${jobWorkspace}'",)

                // Download and Unzip Tar File
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
                sh "cd ${llmPath} && tar -zxf ${BUILD_CONFIGS[config][TARNAME]}"

                // Upload slurm_run_sh to Frontend node
                def scriptRunLocalPath = "${llmSrcLocal}/jenkins/scripts/slurm_run.sh"
                Utils.exec(pipeline, script: "chmod +x ${scriptRunLocalPath}", returnStdout: true)
                Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' scp -r -p -oStrictHostKeyChecking=no ${scriptRunLocalPath} ${remote.user}@${remote.host}:${scriptRunNode}",)

                // Generate Test List and Upload to Frontend Node
                def makoArgs = getMakoArgsFromStageName(stageName, true)
                // TODO: currently the options will only be processed if the first
                // line is "Mako options:", maybe we can make it more generic, which
                // if the line cannot be split by "=", just ignore that line.
                def makoOptsJson = transformMakoArgsToJson(["Mako options:"] + makoArgs)
                def testListPath = renderTestDB(testList, llmSrcLocal, stageName, makoOptsJson)
                Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' scp -r -p -oStrictHostKeyChecking=no ${testListPath} ${remote.user}@${remote.host}:${testListPathNode}",)

                // Generate Multi Node Job Launch Script
                def container = LLM_DOCKER_IMAGE.replace("urm.nvidia.com/", "urm.nvidia.com#")
                def mounts = "/home/scratch.trt_llm_data:/scratch.trt_llm_data:ro,/home/svc_tensorrt/bloom/scripts:/home/svc_tensorrt/bloom/scripts"
                String taskArgs = getNodeArgs(nodeCount, gpuCount)

                if (taskArgs == null) {
                    error "Invalid multinode task stage name is set"
                }

                taskArgs =  [
                    taskArgs,
                    "--exclusive",
                    "--container-image=${container}",
                    "--container-workdir=/home/svc_tensorrt/bloom/scripts",
                    "--container-mounts=${mounts}",
                ].join(" ")

                def scriptLaunch = "/home/svc_tensorrt/bloom/scripts/${jobUID}/slurm_launch.sh"
                def srunCmd = SlurmConfig.generateMultiNodeCommand(partition, taskArgs, scriptRunNode)
                scriptLaunchDestPath = Utils.createTempLocation(pipeline, "./slurm_launch.sh")
                def scriptContent = """#!/bin/bash
                    export jobWorkspace=$jobWorkspace
                    export tarName=$tarName
                    export llmTarfile=$llmTarfile
                    export llmSrcNode=$llmSrcNode
                    export stageName=$stageName
                    export testList=$testList
                    export testListPathNode=$testListPathNode
                    export pytestTestTimeout=$pytestTestTimeout
                    export splits=$splits
                    export splitId=$splitId
                    export perfMode=$perfMode
                    export resourcePathNode=$resourcePathNode
                    export MODEL_CACHE_DIR=$MODEL_CACHE_DIR
                    chmod +x ${scriptRunNode}
                    ${srunCmd}
                """.stripIndent()
                pipeline.writeFile(file: scriptLaunchDestPath, text: scriptContent)
                Utils.exec(pipeline, script: "chmod +x ${scriptLaunchDestPath}", returnStdout: true)
                Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' scp -r -p -oStrictHostKeyChecking=no ${scriptLaunchDestPath} ${remote.user}@${remote.host}:${scriptLaunch}",)
            }
            stage('Run Test') {
                def scriptLaunch = "${jobWorkspace}/slurm_launch.sh"
                Utils.exec(
                    pipeline,
                    timeout: false,
                    script: Utils.sshUserCmd(
                        remote,
                        """bash ${scriptLaunch}"""
                    )
                )
            }
        }
    } finally {
        uploadResults(pipeline, cluster, jobUID, stageName)
        cleanUpNodeResourcesMultiNodes(pipeline, cluster, jobUID)
    }
}

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
def TEST_BACKEND = "test_backend"
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
def AUTO_TRIGGER_TAG_LIST = "auto_trigger_tag_list"
@Field
def DEBUG_MODE = "debug"
@Field
def DETAILED_LOG = "detailed_log"
@Field
def ONLY_DOCS_FILE_CHANGED = "only_docs_file_changed"
@Field
def testFilter = [
    (REUSE_STAGE_LIST): null,
    (ENABLE_SKIP_TEST): false,
    (TEST_STAGE_LIST): null,
    (GPU_TYPE_LIST): null,
    (TEST_BACKEND): null,
    (IS_POST_MERGE): false,
    (ADD_MULTI_GPU_TEST): false,
    (ONLY_MULTI_GPU_TEST): false,
    (DISABLE_MULTI_GPU_TEST): false,
    (EXTRA_STAGE_LIST): null,
    (MULTI_GPU_FILE_CHANGED): false,
    (ONLY_PYTORCH_FILE_CHANGED): false,
    (DEBUG_MODE): false,
    (AUTO_TRIGGER_TAG_LIST): [],
    (DETAILED_LOG): false,
    (ONLY_DOCS_FILE_CHANGED): false,
]

@Field
def GITHUB_PR_API_URL = "github_pr_api_url"
@Field
def CACHED_CHANGED_FILE_LIST = "cached_changed_file_list"
@Field
def ACTION_INFO = "action_info"
def globalVars = [
    (GITHUB_PR_API_URL): null,
    (CACHED_CHANGED_FILE_LIST): null,
    (ACTION_INFO): null,
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

    def archSuffix = arch == "arm64" ? "arm" : "amd"
    def jnlpImage = "urm.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_${archSuffix}_linux:jdk17"

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
    case "slurm":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS}]
                    tty: true
                    resources:
                      requests:
                        cpu: ${SLURM_CORES_REQUEST}
                        memory: ${SLURM_MEMORY_REQUEST}
                        ephemeral-storage: 100Gi
                      limits:
                        cpu: ${SLURM_CORES_LIMIT}
                        memory: ${SLURM_MEMORY_LIMIT}
                        ephemeral-storage: 100Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    case "build":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS_TMP}]
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
        if (type.contains("dgx-h100") || type.contains("dgx-h200") || type in ["b100-ts2", "gh200", "rtx-5080", "rtx-5090"]) {
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
        // PVC mount isn't supported on aarch64 platform. Use NFS as a WAR.
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
                    image: ${jnlpImage}
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
            sh "NVIDIA_TRITON_SERVER_VERSION=25.04 LLM_ROOT=${llmSrc} LLM_BACKEND_ROOT=${llmSrc}/triton_backend python3 ${llmSrc}/scripts/check_test_list.py --l0 --qa --waive"
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

def transformMakoArgsToJson(optList) {
    def makoOpts = [:]
    def startedMakoOpts = false
    def param = null
    def value = null
    optList.each { val ->
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

    def makoOptsJson = JsonOutput.toJson(makoOpts)

    // Print and return the Test DB Query as a JSON string
    echo "Test DB Mako opts: ${makoOptsJson}"
    return makoOptsJson
}

def getMakoOpts(getMakoScript, makoArgs=[]) {
    // We want to save a map for the Mako opts
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

    def makoOptsJson = transformMakoArgsToJson(turtleOutList)

    return makoOptsJson
}

def parseMultiNodeTaskConfigFromStageName(String stageName) {
    def taskConfig = null
    def matcher = (stageName =~ /([^-]+)-(\d+)_GPUs-(\d+)_Nodes/)
    if (matcher.find()) {
        taskConfig = [
            gpu: "${matcher.group(1)}",
            system_gpu_count: "${matcher.group(2)}",
            node_count: "${matcher.group(3)}" // "node_count" might not be used currently
        ]
    }
    return taskConfig
}

def getMakoArgsFromStageName(stageName, parseSysinfo=false) {
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
    } else if (stageName.contains("-Triton-")) {
        // If stageName contains "-Triton-", add "backend=triton" to makoArgs
        // At this point, only tests with backend=triton or unspecified backend will be run
        makoArgs += ["backend=triton"]
    } else {
        // If stageName does not contain "-PyTorch-", "-TensorRT-", "-CPP-", or "-Triton-", do not add any backend
        // At this point, all tests will be run
        // For cases where backend is not specified in makoArgs, we will match all types of backends and tests without specified backend
    }
    if (stageName.contains("-DeepSeek-")) {
        makoArgs += ["auto_trigger=deepseek"]
    } else {
        makoArgs += ["auto_trigger=others"]
    }

    if (parseSysinfo) {
        def taskConfig = parseMultiNodeTaskConfigFromStageName(stageName)
        if (taskConfig) {
            makoArgs += [
                "gpu=${taskConfig.gpu}",
                "system_gpu_count=${taskConfig.system_gpu_count}"
            ]
        }
    }

    return makoArgs
}

def renderTestDB(testContext, llmSrc, stageName, preDefinedMakoOpts=null) {
    def makoOpts = preDefinedMakoOpts

    if (!makoOpts) {
        def scriptPath = "${llmSrc}/tests/integration/defs/sysinfo/get_sysinfo.py"
        def makoArgs = getMakoArgsFromStageName(stageName)
        makoOpts = getMakoOpts(scriptPath, makoArgs)
    }

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

def rerunFailedTests(stageName, llmSrc, testCmdLine) {
    if (!fileExists("${WORKSPACE}/${stageName}/results.xml")) {
        error "There is not results.xml file, skip the rerun step"
    }

    // Generate rerun test lists
    def failSignaturesList = trtllm_utils.getFailSignaturesList().join(",")
    sh """
        python3 ${llmSrc}/tests/integration/defs/test_rerun.py \
        generate_rerun_tests_list \
        --output-dir=${WORKSPACE}/${stageName}/ \
        --input-file=${WORKSPACE}/${stageName}/results.xml \
        --fail-signatures='${failSignaturesList}'
    """

    // If there are some failed tests that cannot be rerun (e.g. test duration > 10 min and no known failure signatures),
    // fail the stage immediately without attempting any reruns
    rerunTestList = "${WORKSPACE}/${stageName}/rerun_0.txt"
    if (fileExists(rerunTestList)) {
        sh "cat ${rerunTestList}"
        error "There are some failed tests that cannot be rerun, skip the rerun step."
    }

    // If the stage has more than 5 failed tests, skip the rerun step
    def validLineCount = 0
    for (times in [1, 2]) {
        rerunTestList = "${WORKSPACE}/${stageName}/rerun_${times}.txt"
        if (fileExists(rerunTestList)) {
            count = sh(
                script: "grep -v '^[[:space:]]*\$' ${rerunTestList} | wc -l",
                returnStdout: true
            ).trim().toInteger()
            echo "Found ${count} tests to rerun ${times} time(s)"
            validLineCount += count
        }
    }
    if (validLineCount > 5) {
        error "There are more than 5 failed tests, skip the rerun step."
    } else if (validLineCount == 0) {
        error "No failed tests need to be rerun, skip the rerun step."
    }

    // Rerun tests
    isRerunFailed = false
    for (times in [1, 2]) {
        rerunTestList = "${WORKSPACE}/${stageName}/rerun_${times}.txt"
        if (!fileExists(rerunTestList)) {
            echo "No failed tests need to be rerun ${times} time(s)"
            continue
        }
        sh "cat ${rerunTestList}"
        xmlFile = "${WORKSPACE}/${stageName}/rerun_results_${times}.xml"
        // change the testCmdLine for rerun
        noNeedLine = ["--splitting-algorithm", "--splits", "--group", "--waives-file", "--cov"]
        needToChangeLine = ["--test-list", "--csv", "--junit-xml"]
        testCmdLine = testCmdLine.findAll { cmd ->
            !noNeedLine.any { line -> cmd.contains(line) } && !needToChangeLine.any { line -> cmd.contains(line) }
        }
        testCmdLine += [
            "--test-list=${rerunTestList}",
            "--csv=${WORKSPACE}/${stageName}/rerun_report_${times}.csv",
            "--junit-xml ${xmlFile}",
            "--reruns ${times - 1}"
        ]
        try {
            sh """
                cd ${llmSrc}/tests/integration/defs && \
                ${testCmdLine.join(" ")}
            """
        } catch(InterruptedException e) {
            throw e
        } catch (Exception e) {
            if (!fileExists(xmlFile)) {
                echo "The tests crashed when rerun attempt."
                throw e
            }
            echo "The tests still failed after rerun attempt."
            isRerunFailed = true
        }
    }

    // generate rerun report
    inputFiles = ["${WORKSPACE}/${stageName}/results.xml",
                  "${WORKSPACE}/${stageName}/rerun_results_1.xml",
                  "${WORKSPACE}/${stageName}/rerun_results_2.xml"]
    sh """
        python3 ${llmSrc}/tests/integration/defs/test_rerun.py \
        generate_rerun_report \
        --output-file=${WORKSPACE}/${stageName}/rerun_results.xml \
        --input-files=${inputFiles.join(",")}
    """

    // Update original results xml file with rerun results xml files for junit
    sh """
        python3 ${llmSrc}/tests/integration/defs/test_rerun.py \
        merge_junit_xmls \
        --output-file=${WORKSPACE}/${stageName}/results.xml \
        --input-files=${inputFiles.join(",")} \
        --deduplicate
    """

    trtllm_utils.uploadArtifacts(
        "${WORKSPACE}/${stageName}/rerun_results.html",
        "${UPLOAD_PATH}/rerun_reports/${stageName}_rerun_results.html"
    )

    echo "Test rerun report: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/rerun_reports/${stageName}_rerun_results.html"
    echo "isRerunFailed: ${isRerunFailed}"
    return isRerunFailed
}

def runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312")
{
    // Step 1: create LLM_ROOT dir
    sh "pwd && ls -alh"
    // TODO: proper way to clean workspace, maybe save in a folder named with BUILD_ID.
    // So that it can work with multiple job running in same node
    sh "rm -rf ./*"
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

        trtllm_utils.llmExecStepWithRetry(pipeline, script: "mkdir -p /opt/tritonserver/backends/tensorrtllm")
        def isAarch64 = config.contains("aarch64")
        if (!isAarch64) {
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && cp TensorRT-LLM/triton_backend/inflight_batcher_llm/libtriton_tensorrtllm.so /opt/tritonserver/backends/tensorrtllm/")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && cp TensorRT-LLM/triton_backend/inflight_batcher_llm/trtllmExecutorWorker /opt/tritonserver/backends/tensorrtllm/")
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
        def pytestTestTimeout = "3600"

        // TRT uses half of the host logic cores for engine building which is bad for multi-GPU machines.
        extraInternalEnv = "__LUNOWUD=\"-thread_pool_size=${TESTER_CORES}\""
        // CPP test execution is timing out easily, so we always override its internal timeout to the same value as pytest
        extraInternalEnv += " CPP_TEST_TIMEOUT_OVERRIDDEN=${pytestTestTimeout}"

        def testDBList = renderTestDB(testList, llmSrc, stageName)
        testList = "${testList}_${splitId}"
        def testCmdLine = [
            "LLM_ROOT=${llmSrc}",
            "LLM_BACKEND_ROOT=${llmSrc}/triton_backend",
            "LLM_MODELS_ROOT=${MODEL_CACHE_DIR}",
            "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}",
            extraInternalEnv,
            "pytest",
            "-v",
            testFilter[(DETAILED_LOG)] ? "-s" : "",
            "--timeout-method=thread",
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
                try {
                    sh """
                        rm -rf ${stageName}/ && \
                        cd ${llmSrc}/tests/integration/defs && \
                        ${testCmdLine.join(" ")}
                    """
                } catch (InterruptedException e) {
                    throw e
                } catch (Exception e) {
                    isRerunFailed = rerunFailedTests(stageName, llmSrc, testCmdLine)
                    if (isRerunFailed) {
                        error "The tests still failed after rerun attempt."
                    }
                }
            }
        }

        if (perfMode) {
            basePerfFilename = stageName.contains("PyTorch") ? "base_perf_pytorch.csv" : "base_perf.csv"
            basePerfPath = "${llmSrc}/tests/integration/defs/perf/${basePerfFilename}"
            stage("Check perf result") {
                sh """
                    python3 ${llmSrc}/tests/integration/defs/perf/sanity_perf_check.py \
                    ${stageName}/perf_script_test_results.csv \
                    ${basePerfPath}
                """
            }
            stage("Create perf report") {
                sh """
                    python3 ${llmSrc}/tests/integration/defs/perf/create_perf_comparison_report.py \
                    --output_path ${stageName}/report.pdf \
                    --files ${stageName}/perf_script_test_results.csv \
                    ${basePerfPath}
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
        // If the execution test list is null, remove the test result xml
        sh """
            ls -all ${stageName}/
            if ! grep -q '<testcase' ${stageName}/results.xml; then
                rm ${stageName}/results.xml || true
            fi
        """
        def llmPath = sh (script: "realpath .", returnStdout: true).trim()
        def llmSrc = "${llmPath}/${LLM_ROOT}${config}/TensorRT-LLM/src"
        // CPP tests will generate test result in ${llmSrc}/cpp/build_backup/, move these files to job result folder
        sh "ls -all ${llmSrc}/cpp/build_backup/ || true"
        sh "ls -all ${llmSrc}/cpp/build/ || true"
        // Sed for CPP test result
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/\" classname=\"/\" classname=\"${stageName}./g' *.xml || true"
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/testsuite name=\"[^\"]*\"/testsuite name=\"${stageName}\"/g' *.xml || true"
        // Sed for Pytest result
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


def runLLMBuild(pipeline, cpu_arch, reinstall_dependencies=false, wheel_path="", cpver="cp312")
{
    sh "pwd && ls -alh"
    sh "env | sort"
    sh "ccache -sv"

    trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, "tensorrt_llm", true, true)
    if (env.alternativeTRT) {
        sh "cd ${LLM_ROOT} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
    }

    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"
    sh "cat ${CCACHE_DIR}/ccache.conf"
    sh "bash -c 'pip3 show tensorrt || true'"
    if (reinstall_dependencies == true) {
        sh "#!/bin/bash \n" + "pip3 uninstall -y torch"
        sh "#!/bin/bash \n" + "yum remove -y libcudnn*"
    }

    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && pip3 install -r requirements-dev.txt")
    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
    }
    buildArgs = "--clean"
    if (cpu_arch == AARCH64_TRIPLE) {
        buildArgs += " -a '90-real;100-real;120-real'"
    }

    withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'CONAN_LOGIN_USERNAME', passwordVariable: 'CONAN_PASSWORD')]) {
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && python3 scripts/build_wheel.py --use_ccache -G Ninja -j ${BUILD_JOBS} -D 'WARNING_IS_ERROR=ON' ${buildArgs}")
    }
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
        def sortedJobKeys = jobKeys.sort()
        throw new Exception("Cannot find the stage names [${invalidStageName}] from the passed params [${paramName}]. Available stage names (${sortedJobKeys.size()} total):\n${sortedJobKeys.collect { "    ${it}" }.join('\n')}")
    }
}

def checkStageName(stageNames) {
    invalidStageName = stageNames.findAll { !(it ==~ /[-\+\w\[\]]+/) }
    if (invalidStageName) {
        throw new Exception("Invalid stage name: [${invalidStageName}], we only support chars '-+_[]0-9a-zA-Z' .")
    }
}

// TODO: Update existing functions to use runInDockerOnNodeMultiStage and get rid of runInDockerOnNode
def runInDockerOnNodeMultiStage(image, label, dockerArgs, needToDeleteDir=true)
{
    return {
        runner -> node(label) {
            if (needToDeleteDir) {
                deleteDir()
            }
            stage('Pull Docker Image') {
                docker.image(image).pull()
            }
            docker.image(image).inside(dockerArgs) {
                runner()
            }
        }
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
    x86TestConfigs = [
        "DGX_H100-4_GPUs-PyTorch-DeepSeek-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 2, 4],
        "DGX_H100-4_GPUs-PyTorch-DeepSeek-2": ["dgx-h100-x4", "l0_dgx_h100", 2, 2, 4],
        "DGX_H100-4_GPUs-PyTorch-Others-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-Triton-Post-Merge-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-CPP-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "A10-PyTorch-1": ["a10", "l0_a10", 1, 1],
        "A10-CPP-1": ["a10", "l0_a10", 1, 1],
        "A10-TensorRT-1": ["a10", "l0_a10", 1, 6],
        "A10-TensorRT-2": ["a10", "l0_a10", 2, 6],
        "A10-TensorRT-3": ["a10", "l0_a10", 3, 6],
        "A10-TensorRT-4": ["a10", "l0_a10", 4, 6],
        "A10-TensorRT-5": ["a10", "l0_a10", 5, 6],
        "A10-TensorRT-6": ["a10", "l0_a10", 6, 6],
        "A30-Triton-1": ["a30", "l0_a30", 1, 1],
        "A30-PyTorch-1": ["a30", "l0_a30", 1, 2],
        "A30-PyTorch-2": ["a30", "l0_a30", 2, 2],
        "A30-CPP-1": ["a30", "l0_a30", 1, 3],
        "A30-CPP-2": ["a30", "l0_a30", 2, 3],
        "A30-CPP-3": ["a30", "l0_a30", 3, 3],
        "A100X-PyTorch-1": ["a100x", "l0_a100", 1, 1],
        "L40S-PyTorch-1": ["l40s", "l0_l40s", 1, 2],
        "L40S-PyTorch-2": ["l40s", "l0_l40s", 2, 2],
        "H100_PCIe-PyTorch-1": ["h100-cr", "l0_h100", 1, 3],
        "H100_PCIe-PyTorch-2": ["h100-cr", "l0_h100", 2, 3],
        "H100_PCIe-PyTorch-3": ["h100-cr", "l0_h100", 3, 3],
        "H100_PCIe-CPP-1": ["h100-cr", "l0_h100", 1, 2],
        "H100_PCIe-CPP-2": ["h100-cr", "l0_h100", 2, 2],
        "H100_PCIe-TensorRT-1": ["h100-cr", "l0_h100", 1, 2],
        "H100_PCIe-TensorRT-2": ["h100-cr", "l0_h100", 2, 2],
        "B200_PCIe-PyTorch-1": ["b100-ts2", "l0_b200", 1, 3],
        "B200_PCIe-PyTorch-2": ["b100-ts2", "l0_b200", 2, 3],
        "B200_PCIe-PyTorch-3": ["b100-ts2", "l0_b200", 3, 3],
        "B200_PCIe-TensorRT-1": ["b100-ts2", "l0_b200", 1, 2],
        "B200_PCIe-TensorRT-2": ["b100-ts2", "l0_b200", 2, 2],
        "RTX5090-PyTorch-1": ["rtx-5090", "l0_gb202", 1, 1],
        "RTX5080-TensorRT-1": ["rtx-5080", "l0_gb203", 1, 2],
        "RTX5080-TensorRT-2": ["rtx-5080", "l0_gb203", 2, 2],
        // Currently post-merge test stages only run tests with "stage: post_merge" mako
        // in the test-db. This behavior may change in the future.
        "A10-PyTorch-Post-Merge-1": ["a10", "l0_a10", 1, 1],
        "A10-TensorRT-Post-Merge-1": ["a10", "l0_a10", 1, 2],
        "A10-TensorRT-Post-Merge-2": ["a10", "l0_a10", 2, 2],
        "A30-TensorRT-Post-Merge-1": ["a30", "l0_a30", 1, 6],
        "A30-TensorRT-Post-Merge-2": ["a30", "l0_a30", 2, 6],
        "A30-TensorRT-Post-Merge-3": ["a30", "l0_a30", 3, 6],
        "A30-TensorRT-Post-Merge-4": ["a30", "l0_a30", 4, 6],
        "A30-TensorRT-Post-Merge-5": ["a30", "l0_a30", 5, 6],
        "A30-TensorRT-Post-Merge-6": ["a30", "l0_a30", 6, 6],
        "A30-CPP-Post-Merge-1": ["a30", "l0_a30", 1, 1],
        "A30-Triton-Post-Merge-1": ["a30", "l0_a30", 1, 2],
        "A30-Triton-Post-Merge-2": ["a30", "l0_a30", 2, 2],
        "A100X-TensorRT-Post-Merge-1": ["a100x", "l0_a100", 1, 6],
        "A100X-TensorRT-Post-Merge-2": ["a100x", "l0_a100", 2, 6],
        "A100X-TensorRT-Post-Merge-3": ["a100x", "l0_a100", 3, 6],
        "A100X-TensorRT-Post-Merge-4": ["a100x", "l0_a100", 4, 6],
        "A100X-TensorRT-Post-Merge-5": ["a100x", "l0_a100", 5, 6],
        "A100X-TensorRT-Post-Merge-6": ["a100x", "l0_a100", 6, 6],
        "A100X-Triton-Post-Merge-1": ["a100x", "l0_a100", 1, 2],
        "A100X-Triton-Post-Merge-2": ["a100x", "l0_a100", 2, 2],
        "L40S-TensorRT-Post-Merge-1": ["l40s", "l0_l40s", 1, 5],
        "L40S-TensorRT-Post-Merge-2": ["l40s", "l0_l40s", 2, 5],
        "L40S-TensorRT-Post-Merge-3": ["l40s", "l0_l40s", 3, 5],
        "L40S-TensorRT-Post-Merge-4": ["l40s", "l0_l40s", 4, 5],
        "L40S-TensorRT-Post-Merge-5": ["l40s", "l0_l40s", 5, 5],
        "H100_PCIe-PyTorch-Post-Merge-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-CPP-Post-Merge-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-TensorRT-Post-Merge-1": ["h100-cr", "l0_h100", 1, 5],
        "H100_PCIe-TensorRT-Post-Merge-2": ["h100-cr", "l0_h100", 2, 5],
        "H100_PCIe-TensorRT-Post-Merge-3": ["h100-cr", "l0_h100", 3, 5],
        "H100_PCIe-TensorRT-Post-Merge-4": ["h100-cr", "l0_h100", 4, 5],
        "H100_PCIe-TensorRT-Post-Merge-5": ["h100-cr", "l0_h100", 5, 5],
        "B200_PCIe-Triton-Post-Merge-1": ["b100-ts2", "l0_b200", 1, 1],
        "H100_PCIe-TensorRT-Perf-1": ["h100-cr", "l0_perf", 1, 1],
        "H100_PCIe-PyTorch-Perf-1": ["h100-cr", "l0_perf", 1, 1],
        "DGX_H200-8_GPUs-PyTorch-Post-Merge-1": ["dgx-h200-x8", "l0_dgx_h200", 1, 1, 8],
        "DGX_H200-4_GPUs-PyTorch-Post-Merge-1": ["dgx-h200-x4", "l0_dgx_h200", 1, 1, 4],
        "DGX_H200-4_GPUs-TensorRT-Post-Merge-1": ["dgx-h200-x4", "l0_dgx_h200", 1, 3, 4],
        "DGX_H200-4_GPUs-TensorRT-Post-Merge-2": ["dgx-h200-x4", "l0_dgx_h200", 2, 3, 4],
        "DGX_H200-4_GPUs-TensorRT-Post-Merge-3": ["dgx-h200-x4", "l0_dgx_h200", 3, 3, 4],
    ]

    parallelJobs = x86TestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "amd64", values[4] ?: 1, key.contains("Perf")), {
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

    x86SlurmTestConfigs = [
        "RTXPro6000-PyTorch-Post-Merge-1": ["rtx-pro-6000", "l0_rtx_pro_6000", 1, 1],
        "DGX_B200-4_GPUs-PyTorch-Post-Merge-1": ["b200-4-gpus", "l0_dgx_b200", 1, 1, 4],
    ]
    fullSet += x86SlurmTestConfigs.keySet()

    parallelSlurmJobs = x86SlurmTestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, "slurm", "amd64"), {
        def config = VANILLA_CONFIG
        if (key.contains("single-device")) {
            config = SINGLE_DEVICE_CONFIG
        }
        if (key.contains("llvm")) {
            config = LLVM_CONFIG
        }
        runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("Perf"), key, values[2], values[3], values[4] ?: 1)
    }]]}

    parallelJobs += parallelSlurmJobs

    // Try to match what are being tested on x86 H100_PCIe.
    // The total machine time is scaled proportionally according to the number of each GPU.
    SBSATestConfigs = [
        "GH200-TensorRT-Post-Merge-1": ["gh200", "l0_gh200", 1, 2],
        "GH200-TensorRT-Post-Merge-2": ["gh200", "l0_gh200", 2, 2],
    ]
    fullSet += SBSATestConfigs.keySet()

    SBSASlurmTestConfigs = [
        "GB200-4_GPUs-PyTorch-1": ["gb200-4-gpus", "l0_gb200", 1, 1, 4],
        "GB200-4_GPUs-PyTorch-Post-Merge-1": ["gb200-4-gpus", "l0_gb200", 1, 1, 4],
    ]
    fullSet += SBSASlurmTestConfigs.keySet()

    multiNodesSBSAConfigs = [
        // Each stage test 1 testcase with 8 GPUs and 2 nodes.
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-1": ["gb200-multi-node", "l0_gb200_multi_nodes", 1, 4, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-2": ["gb200-multi-node", "l0_gb200_multi_nodes", 2, 4, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-3": ["gb200-multi-node", "l0_gb200_multi_nodes", 3, 4, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-4": ["gb200-multi-node", "l0_gb200_multi_nodes", 4, 4, 8, 2],
    ]
    fullSet += multiNodesSBSAConfigs.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        parallelJobs = SBSATestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "arm64"), {
            runLLMTestlistOnPlatform(pipeline, values[0], values[1], LINUX_AARCH64_CONFIG, false, key, values[2], values[3])
        }]]}

        // Add SBSA Slurm jobs
        parallelSlurmJobs = SBSASlurmTestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, "slurm", "arm64"), {
            def config = LINUX_AARCH64_CONFIG
            if (key.contains("single-device")) {
                config = SINGLE_DEVICE_CONFIG
            }
            if (key.contains("llvm")) {
                config = LLVM_CONFIG
            }
            runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("Perf"), key, values[2], values[3], values[4] ?: 1)
        }]]}
        parallelJobs += parallelSlurmJobs

        // Add SBSA multi node Slurm jobs
        parallelMultiNodesSBSAJobs = multiNodesSBSAConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, "slurm", "arm64"), {
            def config = LINUX_AARCH64_CONFIG
            if (key.contains("single-device")) {
                config = SINGLE_DEVICE_CONFIG
            }
            if (key.contains("llvm")) {
                config = LLVM_CONFIG
            }
            runLLMTestlistOnSlurm_MultiNodes(pipeline, values[0], values[1], config, key.contains("Perf"), key, values[2], values[3], values[4] ?: 1, values[5] ?: 2)
        }]]}

        parallelJobs += parallelMultiNodesSBSAJobs
    }

    docBuildSpec = createKubernetesPodConfig(LLM_DOCKER_IMAGE, "a10")
    docBuildConfigs = [
        "A10-Build_Docs": [docBuildSpec, {
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

    // Python version and OS for sanity check
    x86SanityCheckConfigs = [
        "PY312-DLFW": [
            LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE,
            "B200_PCIe",
            X86_64_TRIPLE,
            true,
            "dlfw/",
            DLFW_IMAGE,
            false,
        ],
        "PY310-UB2204": [
            LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE,
            "A10",
            X86_64_TRIPLE,
            true,
            "",
            UBUNTU_22_04_IMAGE,
            false,
        ],
        "PY312-UB2404": [
            LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE,
            "RTX5090",
            X86_64_TRIPLE,
            true,
            "",
            UBUNTU_24_04_IMAGE,
            true, // Extra PyTorch CUDA 12.8 install
        ],
    ]

    aarch64SanityCheckConfigs = [
        "PY312-UB2404": [
            LLM_DOCKER_IMAGE,
            "GH200",
            AARCH64_TRIPLE,
            false,
            "",
            UBUNTU_24_04_IMAGE,
            true, // Extra PyTorch CUDA 12.8 install
        ],
        "PY312-DLFW": [
            LLM_DOCKER_IMAGE,
            "GH200",
            AARCH64_TRIPLE,
            false,
            "dlfw/",
            DLFW_IMAGE,
            false,
        ],
    ]

    def toStageName = { gpuType, key -> "${gpuType}-PackageSanityCheck-${key}".toString() }
    fullSet += x86SanityCheckConfigs.collectEntries{ key, values -> [toStageName(values[1], key), null] }.keySet()
    fullSet += aarch64SanityCheckConfigs.collectEntries{ key, values -> [toStageName(values[1], key), null] }.keySet()

    sanityCheckConfigs = x86SanityCheckConfigs
    if (env.targetArch == AARCH64_TRIPLE) {
        sanityCheckConfigs = aarch64SanityCheckConfigs
    }

    sanityCheckJobs = sanityCheckConfigs.collectEntries {key, values -> [toStageName(values[1], key), {
        cacheErrorAndUploadResult(toStageName(values[1], key), {
            def cpu_arch = values[2]
            def gpu_type = values[1].toLowerCase()
            if (values[1] == "B200_PCIe") {
                gpu_type = "b100-ts2"
            }
            if (values[1] == "RTX5090") {
                gpu_type = "rtx-5090"
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
            if (key.contains("PY310")) {
                cpver = "cp310"
                pyver = "3.10"
            }

            buildRunner("[${toStageName(values[1], key)}] Build") {
                def env = []
                if (key.contains("manylinux")) {
                    env = ["LD_LIBRARY_PATH+=:/usr/local/cuda/compat"]
                }
                withEnv(env) {
                    wheelName = runLLMBuild(pipeline, cpu_arch, values[3], wheelPath, cpver)
                }
            }

            def fullWheelPath = "${cpu_arch}/${wheelPath}${wheelName}"

            // TODO: Re-enable the sanity check after updating GPU testers' driver version.
            // sanityRunner("Sanity check") {
            //     runPackageSanityCheck(pipeline, fullWheelPath, values[3], cpver)
            // }

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
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get -y install python3-pip git rsync curl wget")
                        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 config set global.break-system-packages true")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install requests")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 uninstall -y tensorrt")
                        if (values[5] != DLFW_IMAGE) {
                            def ubuntu_version = key.contains("UB2404") ? "ubuntu2404" : "ubuntu2204"
                            def platform = values[2] == X86_64_TRIPLE ? "x86_64" : "sbsa"
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_version}/${platform}/cuda-keyring_1.1-1_all.deb")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "dpkg -i cuda-keyring_1.1-1_all.deb")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get -y install cuda-toolkit-12-9")
                        }

                        // Extra PyTorch CUDA 12.8 install
                        if (values[6]) {
                            echo "###### Extra PyTorch CUDA 12.8 install Start ######"
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
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
    multiGpuJobsPostMerge = parallelJobs.findAll{(it.key.contains("4_GPUs") || it.key.contains("8_GPUs")) && it.key.contains("Post-Merge")}

    parallelJobs += docBuildJobs
    parallelJobs += sanityCheckJobs

    postMergeJobs = parallelJobs.findAll {it.key.contains("Post-Merge")}

    // Start as a normal pre-merge job
    parallelJobsFiltered = parallelJobs - multiGpuJobs - postMergeJobs

    // Check if the multi GPU related file has changed or not. If changed, add multi GPU test stages.
    if (testFilter[(MULTI_GPU_FILE_CHANGED)]) {
        parallelJobsFiltered += multiGpuJobs
    }

    if (testFilter[(AUTO_TRIGGER_TAG_LIST)] != null) {
        echo "AUTO_TRIGGER_TAG_LIST mode is true. Auto trigger tags: ${testFilter[(AUTO_TRIGGER_TAG_LIST)].join(', ')}."
        def autoTriggerTagStages = [:]
        for (tag in testFilter[(AUTO_TRIGGER_TAG_LIST)]) {
            autoTriggerTagStages += parallelJobs.findAll { it.key.contains(tag) }
        }
        parallelJobsFiltered += autoTriggerTagStages
        if (autoTriggerTagStages.size() > 0) {
            echo "Auto trigger will force run stages: ${autoTriggerTagStages.keySet().join(', ')}."
        }
        println parallelJobsFiltered.keySet()
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
        if (testFilter[(IS_POST_MERGE)]) {
            parallelJobsFiltered = multiGpuJobsPostMerge
        } else {
            parallelJobsFiltered = multiGpuJobs
        }
    }

    // Check --disable-multi-gpu-test, if true, remove multi-GPU test stages.
    if (testFilter[(DISABLE_MULTI_GPU_TEST)]) {
        parallelJobsFiltered -= multiGpuJobs
    }

    // Check --gpu-type, filter test stages.
    if (testFilter[(GPU_TYPE_LIST)] != null) {
        echo "Use GPU_TYPE_LIST for filtering. GPU types: ${testFilter[(GPU_TYPE_LIST)]}."
        parallelJobsFiltered = parallelJobsFiltered.findAll {it.key.tokenize('-')[0] in testFilter[(GPU_TYPE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    // Check --backend-mode, filter test stages.
    if (testFilter[(TEST_BACKEND)] != null) {
        echo "Use TEST_BACKEND for filtering. Backend mode: ${testFilter[(TEST_BACKEND)]}."
        def backendMode = testFilter[(TEST_BACKEND)].collect { it.toLowerCase() }
        def changeMap = [
            "pytorch": "-PyTorch-",
            "tensorrt": "-TensorRT-",
            "cpp": "-CPP-",
        ]
        def backendModeList = backendMode.collect { changeMap.get(it) }.flatten()
        def parallelJobsNoBackend = parallelJobsFiltered.findAll { key, _ ->
            !changeMap.values().any { backend -> key.contains(backend) }
        }
        def parallelJobsBackendMode = parallelJobsFiltered.findAll { key, _ ->
            backendModeList.any { backend -> key.contains(backend) }
        }
        parallelJobsFiltered = parallelJobsNoBackend + parallelJobsBackendMode
        echo "parallelJobsBackendMode: ${parallelJobsBackendMode.keySet()}"
        println parallelJobsFiltered.keySet()
    }

    if (testFilter[(ONLY_PYTORCH_FILE_CHANGED)]) {
        if (testFilter[(TEST_BACKEND)] != null) {
            echo "Force disable ONLY_PYTORCH_FILE_CHANGED mode. Backend mode set by flag: ${testFilter[(TEST_BACKEND)]}."
        } else {
            echo "ONLY_PYTORCH_FILE_CHANGED mode is true."
            parallelJobsFiltered = parallelJobsFiltered.findAll { !it.key.contains("-CPP-") && !it.key.contains("-TensorRT-") }
            println parallelJobsFiltered.keySet()
        }
    }

    if (testFilter[(ONLY_DOCS_FILE_CHANGED)]) {
        echo "Only docs files are changed, run doc build stage only."
        parallelJobsFiltered = docBuildJobs
        println parallelJobsFiltered.keySet()
    }

    // Check --stage-list, only run the stages in stage-list.
    if (testFilter[TEST_STAGE_LIST] != null) {
        echo "Use TEST_STAGE_LIST for filtering. Stages: ${testFilter[(TEST_STAGE_LIST)]}."
        parallelJobsFiltered = parallelJobs.findAll {it.key in testFilter[(TEST_STAGE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    // Check --extra-stage, add the stages in extra-stage.
    if (testFilter[EXTRA_STAGE_LIST] != null) {
        echo "Use EXTRA_STAGE_LIST for filtering. Stages: ${testFilter[(EXTRA_STAGE_LIST)]}."
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
        GITHUB_MIRROR="https://urm.nvidia.com/artifactory/github-go-remote"
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
                    testFilter = trtllm_utils.updateMapWithJson(this, testFilter, env.testFilter, "testFilter")
                    println testFilter
                    echo "env.globalVars is: ${env.globalVars}"
                    globalVars = trtllm_utils.updateMapWithJson(this, globalVars, env.globalVars, "globalVars")
                    globalVars[ACTION_INFO] = trtllm_utils.setupPipelineDescription(this, globalVars[ACTION_INFO])
                }
            }
        }
        stage("Check Test Lists")
        {
            when {
                expression {
                    // Only run the test list validation when necessary
                    env.targetArch == X86_64_TRIPLE && testFilter[ONLY_DOCS_FILE_CHANGED] == false
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
                        def dgxSigns = ["DGX_H100", "DGX_H200", "GB200", "DGX_B200"]
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
