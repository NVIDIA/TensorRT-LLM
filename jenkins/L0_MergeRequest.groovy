@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.InterruptedException
import groovy.transform.Field
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

// Container configuration
// available tags can be found in: https://urm.nvidia.com/artifactory/sw-tensorrt-docker/tensorrt-llm/
// [base_image_name]-[arch]-[os](-[python_version])-[trt_version]-[torch_install_type]-[stage]-[date]-[mr_id]
LLM_DOCKER_IMAGE = "urm-rn.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.01-py3-x86_64-ubuntu24.04-trt10.8.0.43-skip-devel-202503131720-8877"
LLM_SBSA_DOCKER_IMAGE = "urm-rn.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.01-py3-aarch64-ubuntu24.04-trt10.8.0.43-skip-devel-202503131720-8877"
LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE = "urm-rn.nvidia.com/sw-tensorrt-docker/tensorrt-llm:cuda-12.8.0-devel-rocky8-x86_64-rocky8-py310-trt10.8.0.43-skip-devel-202503131720-8877"
LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE = "urm-rn.nvidia.com/sw-tensorrt-docker/tensorrt-llm:cuda-12.8.0-devel-rocky8-x86_64-rocky8-py312-trt10.8.0.43-skip-devel-202503131720-8877"

LLM_ROCKYLINUX8_DOCKER_IMAGE = LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE

// TODO: Move common variables to an unified location
BUILD_CORES_REQUEST = "8"
BUILD_CORES_LIMIT = "8"
BUILD_MEMORY_REQUEST = "48Gi"
BUILD_MEMORY_LIMIT = "48Gi"

// Stage choices
STAGE_CHOICE_NORMAL = "normal"
STAGE_CHOICE_SKIP = "skip"
STAGE_CHOICE_IGNORE = "ignore"

RELESE_CHECK_CHOICE = env.releaseCheckChoice ? env.releaseCheckChoice : STAGE_CHOICE_NORMAL
X86_TEST_CHOICE = env.x86TestChoice ? env.x86TestChoice : STAGE_CHOICE_NORMAL
SBSA_TEST_CHOICE = env.SBSATestChoice ? env.SBSATestChoice : STAGE_CHOICE_NORMAL

def gitlabParamsFromBot = [:]

if (env.gitlabTriggerPhrase)
{
    gitlabParamsFromBot = readJSON text: env.gitlabTriggerPhrase, returnPojo: true
}

// "Fail Fast" feature is enabled by default for the pre-merge pipeline.
// "Fail Fast" feature is always disabled for the post-merge pipeline.
boolean enableFailFast = !(env.JOB_NAME ==~ /.*PostMerge.*/ || env.JOB_NAME ==~ /.*Dependency_Testing_TRT.*/) && !gitlabParamsFromBot.get("disable_fail_fast", false)

boolean isReleaseCheckMode = (gitlabParamsFromBot.get("run_mode", "full") == "release_check")

BUILD_STATUS_NAME = isReleaseCheckMode ? "Jenkins Release Check" : "Jenkins Full Build"

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
def ENABLE_MULTI_GPU_TEST = "multi_gpu_test"
@Field
def ONLY_MULTI_GPU_TEST = "only_multi_gpu_test"
@Field
def DISABLE_MULTI_GPU_TEST = "disable_multi_gpu_test"
@Field
def EXTRA_STAGE_LIST = "extra_stage"
@Field
def MULTI_GPU_FILE_CHANGED = "multi_gpu_file_changed"
def testFilter = [
    (REUSE_STAGE_LIST): trimForStageList(gitlabParamsFromBot.get(REUSE_STAGE_LIST, null)?.tokenize(',')),
    (ENABLE_SKIP_TEST): gitlabParamsFromBot.get((ENABLE_SKIP_TEST), false),
    (TEST_STAGE_LIST): trimForStageList(gitlabParamsFromBot.get((TEST_STAGE_LIST), null)?.tokenize(',')),
    (GPU_TYPE_LIST): trimForStageList(gitlabParamsFromBot.get((GPU_TYPE_LIST), null)?.tokenize(',')),
    (IS_POST_MERGE): (env.JOB_NAME ==~ /.*PostMerge.*/) || gitlabParamsFromBot.get((IS_POST_MERGE), false),
    (ADD_MULTI_GPU_TEST): gitlabParamsFromBot.get((ADD_MULTI_GPU_TEST), false),
    (ONLY_MULTI_GPU_TEST): gitlabParamsFromBot.get((ONLY_MULTI_GPU_TEST), false) || gitlabParamsFromBot.get((ENABLE_MULTI_GPU_TEST), false),
    (DISABLE_MULTI_GPU_TEST): gitlabParamsFromBot.get((DISABLE_MULTI_GPU_TEST), false),
    (EXTRA_STAGE_LIST): trimForStageList(gitlabParamsFromBot.get((EXTRA_STAGE_LIST), null)?.tokenize(',')),
    (MULTI_GPU_FILE_CHANGED): false,
]

String reuseBuild = gitlabParamsFromBot.get('reuse_build', null)

// If not running all test stages in the L0 pre-merge, we will not update the GitLab status at the end.
boolean enableUpdateGitlabStatus =
    !testFilter[ENABLE_SKIP_TEST] &&
    !testFilter[ONLY_MULTI_GPU_TEST] &&
    testFilter[GPU_TYPE_LIST] == null &&
    testFilter[TEST_STAGE_LIST] == null

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

def createKubernetesPodConfig(image, type)
{
    def targetCould = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
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
                    image: urm-rn.nvidia.com/docker/alpine:latest
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
                    command: ['cat']
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
    case "package":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
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
    }
    def nodeLabel = trtllm_utils.appendRandomPostfix("${nodeLabelPrefix}---tensorrt-${jobName}-${buildID}")
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
                    image: urm-rn.nvidia.com/docker/jenkins/inbound-agent:4.11-1-jdk11
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
                - name: sw-tensorrt-pvc
                  persistentVolumeClaim:
                    claimName: sw-tensorrt-pvc

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

def setupPipelineEnvironment(pipeline, testFilter)
{
    setupPipelineSpec = createKubernetesPodConfig(LLM_DOCKER_IMAGE, "build")
    trtllm_utils.launchKubernetesPod(pipeline, setupPipelineSpec, "trt-llm", {
        sh "env | sort"
        updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: 'running'
        echo "Using GitLab repo: ${LLM_REPO}."
        sh "git config --global --add safe.directory \"*\""
        if (env.gitlabMergeRequestLastCommit) {
            env.gitlabCommit = env.gitlabMergeRequestLastCommit
        } else {
            branch = env.gitlabBranch ? env.gitlabBranch : "main"
            trtllm_utils.checkoutSource(LLM_REPO, branch, LLM_ROOT, true, true)
            checkoutCommit = sh (script: "cd ${LLM_ROOT} && git rev-parse HEAD",returnStdout: true).trim()
            env.gitlabCommit = checkoutCommit
        }
        echo "Env.gitlabMergeRequestLastCommit: ${env.gitlabMergeRequestLastCommit}."
        echo "Freeze GitLab commit. Branch: ${env.gitlabBranch}. Commit: ${env.gitlabCommit}."
        testFilter[(MULTI_GPU_FILE_CHANGED)] = getMultiGpuFileChanged(pipeline, testFilter)
    })
}

def launchReleaseCheck(pipeline)
{
    stages = {
        trtllm_utils.llmExecStepWithRetry(pipeline, script: """apt-get update && apt-get install \
            python3-pip \
            -y""")
        sh "pip3 config set global.break-system-packages true"
        sh "git config --global --add safe.directory \"*\""
        // Step 1: cloning tekit source code
        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
        sh "cd ${LLM_ROOT} && git config --unset-all core.hooksPath"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT} && pip3 install `grep pre-commit requirements-dev.txt`")
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT} && pip3 install `grep bandit requirements-dev.txt`")
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT} && pre-commit install")
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT} && pre-commit run -a --show-diff-on-failure || (git restore . && false)")
        sh "cd ${LLM_ROOT} && bandit --configfile scripts/bandit.yaml -r tensorrt_llm | tee /tmp/bandit.log"
        sh "cat /tmp/bandit.log | grep -q 'Total lines skipped (#nosec): 0' && exit 0 || exit 1"
        sh "cat /tmp/bandit.log | grep -q 'Issue:' && exit 1 || exit 0"

        // Step 2: build tools
        withEnv(['GONOSUMDB=*.nvidia.com']) {
            withCredentials([
                gitUsernamePassword(
                    credentialsId: 'svc_tensorrt_gitlab_read_api_token',
                    gitToolName: 'git-tool'
                ),
                string(
                    credentialsId: 'default-git-url',
                    variable: 'DEFAULT_GIT_URL'
                )
            ]) {
                sh "go install ${DEFAULT_GIT_URL}/TensorRT/Infrastructure/licensechecker/cmd/license_checker@v0.3.0"
            }
        }
        // Step 3: do some check in container
        sh "cd ${LLM_ROOT}/cpp && /go/bin/license_checker -config ../jenkins/license_cpp.json include tensorrt_llm"
    }

    def image = "urm-rn.nvidia.com/docker/golang:1.22"
    stageName = "Release Check"
    trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(image, "build"), "trt-llm", {
        stage("[${stageName}] Run") {
            if (RELESE_CHECK_CHOICE == STAGE_CHOICE_SKIP) {
                echo "Release Check job is skipped due to Jenkins configuration"
                return
            }
            try {
                echoNodeAndGpuInfo(pipeline, stageName)
                stages()
            } catch (InterruptedException e) {
                throw e
            } catch (Exception e) {
                if (RELESE_CHECK_CHOICE == STAGE_CHOICE_IGNORE) {
                    catchError(
                        buildResult: 'SUCCESS',
                        stageResult: 'FAILURE') {
                        error "Release Check failed but ignored due to Jenkins configuration"
                    }
                } else {
                    throw e
                }
            }
        }
    })
}

def getMergeRequestChangedFileList(pipeline) {
    def changedFileList = []
    def pageId = 0
    withCredentials([
        usernamePassword(
            credentialsId: 'svc_tensorrt_gitlab_read_api_token',
            usernameVariable: 'GITLAB_API_USER',
            passwordVariable: 'GITLAB_API_TOKEN'
        ),
        string(credentialsId: 'default-git-url', variable: 'DEFAULT_GIT_URL')
    ]) {
        while(true) {
            pageId += 1
            def rawDataJson = pipeline.sh(
                script: "curl --header \"PRIVATE-TOKEN: $GITLAB_API_TOKEN\" --url \"https://${DEFAULT_GIT_URL}/api/v4/projects/${env.gitlabMergeRequestTargetProjectId}/merge_requests/${env.gitlabMergeRequestIid}/diffs?page=${pageId}&per_page=20\"",
                returnStdout: true
            )
            def rawDataList = readJSON text: rawDataJson, returnPojo: true
            rawDataList.each { rawData ->
                changedFileList += [rawData.get("old_path"), rawData.get("new_path")]
            }
            if (!rawDataList) { break }
        }
    }
    def changedFileListStr = changedFileList.join(",\n")
    pipeline.echo("The changeset of this MR is: ${changedFileListStr}.")
    return changedFileList
}

def getMultiGpuFileChanged(pipeline, testFilter)
{
    if (testFilter[(DISABLE_MULTI_GPU_TEST)]) {
        pipeline.echo("Force not run multi-GPU testing.")
        return false
    }
    if (env.alternativeTRT || testFilter[(ADD_MULTI_GPU_TEST)] || testFilter[(ONLY_MULTI_GPU_TEST)] || testFilter[(IS_POST_MERGE)]) {
        pipeline.echo("Force run multi-GPU testing.")
        return true
    }

    def relatedFileList = [
        "cpp/include/tensorrt_llm/runtime/gptJsonConfig.h",
        "cpp/include/tensorrt_llm/runtime/worldConfig.h",
        "cpp/include/tensorrt_llm/runtime/utils/mpiUtils.h",
        "cpp/include/tensorrt_llm/runtime/utils/multiDeviceUtils.h",
        "cpp/tensorrt_llm/runtime/utils/mpiUtils.cpp",
        "cpp/tests/runtime/mpiUtilsTest.cpp",
        "cpp/tensorrt_llm/batch_manager/trtGptModelFactory.h",
        "cpp/tensorrt_llm/runtime/worldConfig.cpp",
        "cpp/tensorrt_llm/runtime/ncclCommunicator.cpp",
        "cpp/tensorrt_llm/runtime/workerPool.h",
        "cpp/tensorrt_llm/executor_worker/executorWorker.cpp",
        "cpp/tensorrt_llm/runtime/ipcUtils.cpp",
        "cpp/tensorrt_llm/executor/executor.cpp",
        "cpp/tensorrt_llm/executor/executorImpl.cpp",
        "cpp/tensorrt_llm/executor/executorImpl.h",
        "cpp/tensorrt_llm/runtime/ncclCommunicator.cpp",
        "cpp/tensorrt_llm/kernels/allReduceFusionKernels.h",
        "cpp/tensorrt_llm/kernels/allReduceFusionKernels.cu",
        "cpp/tensorrt_llm/kernels/customAllReduceKernels.h",
        "cpp/tensorrt_llm/kernels/customAllReduceKernels.cu",
        "cpp/tensorrt_llm/kernels/gptKernels.h",
        "cpp/tensorrt_llm/kernels/gptKernels.cu",
        "cpp/tensorrt_llm/kernels/unfusedAttentionKernels.h",
        "cpp/tensorrt_llm/kernels/unfusedAttentionKernels.cu",
        "cpp/tensorrt_llm/kernels/userbuffers/",
        "cpp/tensorrt_llm/pybind/",
        "cpp/tests/kernels/allReduce/",
        "cpp/tensorrt_llm/plugins/cpSplitPlugin/cpSplitPlugin.h",
        "cpp/tensorrt_llm/plugins/cpSplitPlugin/cpSplitPlugin.cpp",
        "cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h",
        "cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.cpp",
        "cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h",
        "cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp",
        "cpp/tests/runtime/mpiUtilsTest.cpp",
        "cpp/tensorrt_llm/plugins/ncclPlugin/",
        "tensorrt_llm/functional.py",
        "tensorrt_llm/mapping.py",
        "tensorrt_llm/llmapi/",
        "tensorrt_llm/executor.py",
        "tensorrt_llm/_ipc_utils.py",
        "tensorrt_llm/parameter.py",
        "tensorrt_llm/models/llama/",
        "tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py",
        "tensorrt_llm/_torch/compilation/patterns/ub_allreduce.py",
        "tensorrt_llm/_torch/custom_ops/userbuffers_custom_ops.py",
        "tests/integration/test_lists/test-db/l0_dgx_h100.yml",
        "tests/unittest/_torch/multi_gpu/",
        "tests/unittest/_torch/multi_gpu_modeling/",
        "jenkins/L0_Test.groovy",
    ]

    def changedFileList = ","
    def relatedFileChanged = false
    try {
        changedFileList = getMergeRequestChangedFileList(pipeline).join(", ")
        relatedFileChanged = relatedFileList.any { it ->
            if (changedFileList.contains(it)) {
                return true
            }
        }
    }
    catch (InterruptedException e)
    {
        throw e
    }
    catch (Exception e)
    {
        pipeline.echo("getMultiGpuFileChanged failed execution.")
    }
    if (relatedFileChanged) {
        pipeline.echo("Detect multi-GPU related files changed.")
    }
    return relatedFileChanged
}

def collectTestResults(pipeline, testFilter)
{
    collectResultPodSpec = createKubernetesPodConfig("", "agent")
    trtllm_utils.launchKubernetesPod(pipeline, collectResultPodSpec, "alpine", {
        stage ("Collect test result") {
            sh "rm -rf **/*.xml *.tar.gz"

            testResultLink = "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}/test-results"

            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add --no-cache curl")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget ${testResultLink}/", allowStepFailed: true)
            sh "cat index.html | grep \"tar.gz\" | cut -d \"\\\"\" -f 2 > result_file_names.txt"
            sh "cat result_file_names.txt"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cat result_file_names.txt | xargs -n1 -I {} wget -c -nv ${testResultLink}/{}", allowStepFailed: true)
            sh "ls -l | grep \"tar.gz\" || true"
            resultFileNumber = sh(script: "cat result_file_names.txt | wc -l", returnStdout: true)
            resultFileDownloadedNumber = sh(script: "ls -l | grep \"tar.gz\" | wc -l", returnStdout: true)
            echo "Result File Number: ${resultFileNumber}, Downloaded: ${resultFileDownloadedNumber}"

            sh "find . -name results-\\*.tar.gz -type f -exec tar -zxvf {} \\; || true"

            junit(testResults: '**/results*.xml', allowEmptyResults : true)
        } // Collect test result stage
        try {
            stage("Test coverage") {
                sh "ls"
                def CUR_PATH = sh(returnStdout: true, script: 'pwd').replaceAll("\\s","")
                sh "echo ${CUR_PATH}"
                sh "rm -rf cov && mkdir -p cov"
                sh "find . -type f -wholename '*/.coverage.*' -exec mv {} cov/ \\; || true"
                sh "cd cov && find . -type f"
                def fileCount = sh(returnStdout: true, script: 'find cov -type f | wc -l').replaceAll("\\s","").toInteger()
                if (fileCount == 0) {
                    echo "Test coverage is skipped because there is no test data file."
                    return
                }
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add python3")
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add py3-pip")
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 config set global.break-system-packages true")
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install coverage")
                sh "coverage --version"

                trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
                sh "cp llm/examples/openai_triton/manual_plugin/fmha_triton.py llm/examples/openai_triton/plugin_autogen/"
                def coverageConfigFile = "cov/.coveragerc"
                sh """
                    echo '[paths]' > ${coverageConfigFile}
                    echo 'source1=\n    ${CUR_PATH}/llm/examples/\n    */TensorRT-LLM/src/examples/' >> ${coverageConfigFile}
                    echo 'source2=\n    ${CUR_PATH}/llm/tensorrt_llm/\n    */tensorrt_llm/' >> ${coverageConfigFile}
                    cat ${coverageConfigFile}
                """

                sh "cd cov && coverage combine"
                sh "cd cov && find . -type f"
                sh "cd cov && coverage report"
                sh "cd cov && coverage html -d test_coverage_html"
                trtllm_utils.uploadArtifacts("cov/test_coverage_html/*", "${UPLOAD_PATH}/test-results/coverage-report/")
                echo "Test coverage report: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/test-results/coverage-report/index.html"
            } // Test coverage
        }
        catch (InterruptedException e)
        {
            throw e
        }
        catch (Exception e)
        {
            pipeline.echo("Test coverage failed execution.")
        }
    })
}

def getCommonParameters()
{
    return [
        'gitlabSourceRepoHttpUrl': LLM_REPO,
        'gitlabCommit': env.gitlabCommit,
        'artifactPath': UPLOAD_PATH,
        'uploadPath': UPLOAD_PATH,
    ]
}

def triggerJob(jobName, parameters, jenkinsUrl = "", credentials = "")
{
    if (jenkinsUrl == "" && env.localJobCredentials) {
        jenkinsUrl = env.JENKINS_URL
        credentials = env.localJobCredentials
    }
    def status = ""
    if (jenkinsUrl != "") {
        def jobPath = trtllm_utils.resolveFullJobName(jobName).replace('/', '/job/').substring(1)
        def handle = triggerRemoteJob(
            job: "${jenkinsUrl}${jobPath}/",
            auth: CredentialsAuth(credentials: credentials),
            parameters: trtllm_utils.toRemoteBuildParameters(parameters),
            pollInterval: 60,
            abortTriggeredJob: true,
        )
        status = handle.getBuildResult().toString()
    } else {
        def handle = build(
            job: jobName,
            parameters: trtllm_utils.toBuildParameters(parameters),
            propagate: false,
        )
        echo "Triggered job: ${handle.absoluteUrl}"
        status = handle.result
    }
    return status
}

def launchStages(pipeline, reuseBuild, testFilter, enableFailFast)
{
    stages = [
        "Release Check": {
            script {
                launchReleaseCheck(this)
            }
        },
        "x86_64-linux": {
            script {
                stage("Build") {
                    def parameters = getCommonParameters()
                    parameters += [
                        'enableFailFast': enableFailFast,
                        'dockerImage': LLM_DOCKER_IMAGE,
                        'wheelDockerImage': LLM_ROCKYLINUX8_DOCKER_IMAGE,
                    ]

                    if (env.alternativeTRT) {
                        parameters += [
                            'alternativeTRT': env.alternativeTRT,
                        ]
                    }

                    if (reuseBuild) {
                        parameters['reuseArtifactPath'] = "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${reuseBuild}"
                    }

                    echo "trigger x86_64 build job, params: ${parameters}"

                    def status = triggerJob("/LLM/helpers/Build-x86_64", parameters)
                    if (status != "SUCCESS") {
                        error "Downstream job did not succeed"
                    }

                }
                def testStageName = "[Test-x86_64] Run"
                if (env.localJobCredentials) {
                    testStageName = "[Test-x86_64] Remote Run"
                }
                stage(testStageName) {
                    if (X86_TEST_CHOICE == STAGE_CHOICE_SKIP) {
                        echo "x86_64 test job is skipped due to Jenkins configuration"
                        return
                    }
                    try {
                        parameters = getCommonParameters()

                        String testFilterJson = writeJSON returnText: true, json: testFilter
                        parameters += [
                            'enableFailFast': enableFailFast,
                            'testFilter': testFilterJson,
                            'dockerImage': LLM_DOCKER_IMAGE,
                        ]

                        if (env.alternativeTRT) {
                            parameters += [
                                'alternativeTRT': env.alternativeTRT,
                            ]
                        }

                        if (env.testPhase2StageName) {
                            parameters += [
                                'testPhase2StageName': env.testPhase2StageName,
                            ]
                        }

                        echo "trigger x86_64 test job, params: ${parameters}"

                        def status = triggerJob(
                            "L0_Test-x86_64",
                            parameters,
                        )

                        if (status != "SUCCESS") {
                            error "Downstream job did not succeed"
                        }
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception e) {
                        if (X86_TEST_CHOICE == STAGE_CHOICE_IGNORE) {
                            catchError(
                                buildResult: 'SUCCESS',
                                stageResult: 'FAILURE') {
                                error "x86_64 test failed but ignored due to Jenkins configuration"
                            }
                        } else {
                            throw e
                        }
                    }
                }
            }
        },
        "SBSA-linux": {
            script {
                def jenkinsUrl = ""
                def credentials = ""
                def testStageName = "[Test-SBSA] Run"
                if (env.localJobCredentials) {
                    testStageName = "[Test-SBSA] Remote Run"
                }

                def stageName = "Build"
                stage(stageName) {
                    def parameters = getCommonParameters()
                    parameters += [
                        'enableFailFast': enableFailFast,
                        "dockerImage": LLM_SBSA_DOCKER_IMAGE,
                    ]

                    if (env.alternativeTrtSBSA) {
                        parameters += [
                            "alternativeTRT": env.alternativeTrtSBSA,
                        ]
                    }

                    if (reuseBuild) {
                        parameters['reuseArtifactPath'] = "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${reuseBuild}"
                    }

                    echo "trigger SBSA build job, params: ${parameters}"

                    def status = triggerJob(
                        "/LLM/helpers/Build-SBSA",
                        parameters,
                        jenkinsUrl,
                        credentials,
                    )

                    if (status != "SUCCESS") {
                        error "Downstream job did not succeed"
                    }
                }
                stage(testStageName) {
                    if (SBSA_TEST_CHOICE == STAGE_CHOICE_SKIP) {
                        echo "SBSA test job is skipped due to Jenkins configuration"
                        return
                    }
                    try {
                        def parameters = getCommonParameters()

                        String testFilterJson = writeJSON returnText: true, json: testFilter
                        parameters += [
                            'enableFailFast': enableFailFast,
                            'testFilter': testFilterJson,
                            "dockerImage": LLM_SBSA_DOCKER_IMAGE,
                        ]

                        if (env.alternativeTrtSBSA) {
                            parameters += [
                                "alternativeTRT": env.alternativeTrtSBSA,
                            ]
                        }

                        echo "trigger SBSA test job, params: ${parameters}"

                        def status = triggerJob(
                            "L0_Test-SBSA",
                            parameters,
                            jenkinsUrl,
                            credentials,
                        )

                        if (status != "SUCCESS") {
                            error "Downstream job did not succeed"
                        }
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception e) {
                        if (SBSA_TEST_CHOICE == STAGE_CHOICE_IGNORE) {
                            catchError(
                                buildResult: 'SUCCESS',
                                stageResult: 'FAILURE') {
                                error "SBSA test failed but ignored due to Jenkins configuration"
                            }
                        } else {
                            throw e
                        }
                    }
                }
            }
        },
    ]

    parallelJobs = stages.collectEntries{key, value -> [key, {
        script {
            stage(key) {
                value()
            }
        }
    }]}

    parallelJobs.failFast = enableFailFast
    pipeline.parallel parallelJobs
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
    post {
        unsuccessful {
            updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: "failed"
        }
        success {
            script {
                if (enableUpdateGitlabStatus) {
                    updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: "success"
                } else {
                    updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: "canceled"
                    updateGitlabCommitStatus name: "Custom Jenkins build", state: "success"
                }
            }
        }
        aborted {
            updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: 'canceled'
        }
        always {
            script {
                if (!isReleaseCheckMode) {
                    collectTestResults(this, testFilter)
                }
            }
        }
    }
    stages {
        stage("Setup environment")
        {
            steps
            {
                script {
                    setupPipelineEnvironment(this, testFilter)
                    echo "enableFailFast is: ${enableFailFast}"
                    echo "env.gitlabTriggerPhrase is: ${env.gitlabTriggerPhrase}"
                    println testFilter
                    echo "Check the passed GitLab bot testFilter parameters."
                }
            }
        }
        stage("Build and Test") {
            steps {
                script {
                    if (isReleaseCheckMode) {
                        stage("Release Check") {
                            script {
                                launchReleaseCheck(this)
                            }
                        }
                    } else {
                        launchStages(this, reuseBuild, testFilter, enableFailFast)
                    }
                }
            }
        }
    } // stages
} // pipeline
