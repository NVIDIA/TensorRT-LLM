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

// LLM repository configuration
withCredentials([string(credentialsId: 'default-scan-repo', variable: 'DEFAULT_SCAN_REPO')]) {
    SCAN_REPO = "${DEFAULT_SCAN_REPO}"
}
SCAN_COMMIT = "main"
SCAN_ROOT = "scan"

ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
UPLOAD_PATH = env.uploadPath ? env.uploadPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"

// Container configuration
def getContainerURIs()
{
    // available tags can be found in: https://urm.nvidia.com/artifactory/sw-tensorrt-docker/tensorrt-llm/
    // [base_image_name]-[arch]-[os](-[python_version])-[trt_version]-[torch_install_type]-[stage]-[date]-[mr_id]
    tagProps = readProperties file: "${LLM_ROOT}/jenkins/current_image_tags.properties", interpolate: true
    uris = [:]
    keys = [
        "LLM_DOCKER_IMAGE",
        "LLM_SBSA_DOCKER_IMAGE",
        "LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE",
        "LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE"
    ]
    for (key in keys) {
        uris[key] = tagProps[key]
    }
    return uris
}

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
def TEST_BACKEND = "test_backend"
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
@Field
def ONLY_ONE_GROUP_CHANGED = "only_one_group_changed"
@Field
def AUTO_TRIGGER_TAG_LIST = "auto_trigger_tag_list"
@Field
def DEBUG_MODE = "debug"
@Field
def DETAILED_LOG = "detailed_log"

def testFilter = [
    (REUSE_STAGE_LIST): trimForStageList(gitlabParamsFromBot.get(REUSE_STAGE_LIST, null)?.tokenize(',')),
    (ENABLE_SKIP_TEST): gitlabParamsFromBot.get((ENABLE_SKIP_TEST), false),
    (TEST_STAGE_LIST): trimForStageList(gitlabParamsFromBot.get((TEST_STAGE_LIST), null)?.tokenize(',')),
    (GPU_TYPE_LIST): trimForStageList(gitlabParamsFromBot.get((GPU_TYPE_LIST), null)?.tokenize(',')),
    (TEST_BACKEND): trimForStageList(gitlabParamsFromBot.get((TEST_BACKEND), null)?.tokenize(',')),
    (IS_POST_MERGE): (env.JOB_NAME ==~ /.*PostMerge.*/) || gitlabParamsFromBot.get((IS_POST_MERGE), false),
    (ADD_MULTI_GPU_TEST): gitlabParamsFromBot.get((ADD_MULTI_GPU_TEST), false),
    (ONLY_MULTI_GPU_TEST): gitlabParamsFromBot.get((ONLY_MULTI_GPU_TEST), false) || gitlabParamsFromBot.get((ENABLE_MULTI_GPU_TEST), false),
    (DISABLE_MULTI_GPU_TEST): gitlabParamsFromBot.get((DISABLE_MULTI_GPU_TEST), false),
    (EXTRA_STAGE_LIST): trimForStageList(gitlabParamsFromBot.get((EXTRA_STAGE_LIST), null)?.tokenize(',')),
    (MULTI_GPU_FILE_CHANGED): false,
    (ONLY_ONE_GROUP_CHANGED): "",
    (DEBUG_MODE): gitlabParamsFromBot.get(DEBUG_MODE, false),
    (AUTO_TRIGGER_TAG_LIST): [],
    (DETAILED_LOG): gitlabParamsFromBot.get(DETAILED_LOG, false),
]

String reuseBuild = gitlabParamsFromBot.get('reuse_build', null)

@Field
def GITHUB_PR_API_URL = "github_pr_api_url"
@Field
def CACHED_CHANGED_FILE_LIST = "cached_changed_file_list"
@Field
def ACTION_INFO = "action_info"
@Field
def IMAGE_KEY_TO_TAG = "image_key_to_tag"
@Field
def TARGET_BRANCH = "target_branch"
def globalVars = [
    (GITHUB_PR_API_URL): gitlabParamsFromBot.get('github_pr_api_url', null),
    (CACHED_CHANGED_FILE_LIST): null,
    (ACTION_INFO): gitlabParamsFromBot.get('action_info', null),
    (IMAGE_KEY_TO_TAG): [:],
    (TARGET_BRANCH): gitlabParamsFromBot.get('target_branch', null),
]

// If not running all test stages in the L0 pre-merge, we will not update the GitLab status at the end.
boolean enableUpdateGitlabStatus =
    !testFilter[ENABLE_SKIP_TEST] &&
    !testFilter[ONLY_MULTI_GPU_TEST] &&
    testFilter[GPU_TYPE_LIST] == null &&
    testFilter[TEST_STAGE_LIST] == null &&
    testFilter[TEST_BACKEND] == null

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

def createKubernetesPodConfig(image, type, arch = "amd64")
{
    def targetCould = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
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
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
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
                    image: ${jnlpImage}
                    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
                    resources:
                      requests:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 5Gi
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

def setupPipelineEnvironment(pipeline, testFilter, globalVars)
{
    sh "env | sort"
    updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: 'running'
    echo "Using GitLab repo: ${LLM_REPO}."
    sh "git config --global --add safe.directory \"*\""
    // NB: getContainerURIs reads files in ${LLM_ROOT}/jenkins/
    if (env.gitlabMergeRequestLastCommit) {
        env.gitlabCommit = env.gitlabMergeRequestLastCommit
        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
    } else {
        branch = env.gitlabBranch ? env.gitlabBranch : "main"
        trtllm_utils.checkoutSource(LLM_REPO, branch, LLM_ROOT, true, true)
        checkoutCommit = sh (script: "cd ${LLM_ROOT} && git rev-parse HEAD",returnStdout: true).trim()
        env.gitlabCommit = checkoutCommit
    }
    echo "Env.gitlabMergeRequestLastCommit: ${env.gitlabMergeRequestLastCommit}."
    echo "Freeze GitLab commit. Branch: ${env.gitlabBranch}. Commit: ${env.gitlabCommit}."
    testFilter[(MULTI_GPU_FILE_CHANGED)] = getMultiGpuFileChanged(pipeline, testFilter, globalVars)
    testFilter[(ONLY_ONE_GROUP_CHANGED)] = getOnlyOneGroupChanged(pipeline, testFilter, globalVars)
    testFilter[(AUTO_TRIGGER_TAG_LIST)] = getAutoTriggerTagList(pipeline, testFilter, globalVars)
    getContainerURIs().each { k, v ->
        globalVars[k] = v
    }
}

def mergeWaiveList(pipeline, globalVars)
{
    // Get current waive list
    sh "git config --global --add safe.directory \"*\""
    sh "cp ${LLM_ROOT}/tests/integration/test_lists/waives.txt ./waives_CUR_${env.gitlabCommit}.txt"
    sh "cp ${LLM_ROOT}/jenkins/scripts/mergeWaiveList.py ./"

    // Get TOT waive list
    LLM_TOT_ROOT = "llm-tot"
    targetBranch = env.gitlabTargetBranch ? env.gitlabTargetBranch : globalVars[TARGET_BRANCH]
    echo "Target branch: ${targetBranch}"
    withCredentials([string(credentialsId: 'default-sync-llm-repo', variable: 'DEFAULT_SYNC_LLM_REPO')]) {
        trtllm_utils.checkoutSource(DEFAULT_SYNC_LLM_REPO, targetBranch, LLM_TOT_ROOT, false, false)
    }
    targetBranchTOTCommit = sh (script: "cd ${LLM_TOT_ROOT} && git rev-parse HEAD", returnStdout: true).trim()
    echo "Target branch TOT commit: ${targetBranchTOTCommit}"
    sh "cp ${LLM_TOT_ROOT}/tests/integration/test_lists/waives.txt ./waives_TOT_${targetBranchTOTCommit}.txt"

    // Get waive list diff in current MR
    def diff = getMergeRequestOneFileChanges(pipeline, globalVars, "tests/integration/test_lists/waives.txt")

    // Write diff to a temporary file to avoid shell escaping issues
    writeFile file: 'diff_content.txt', text: diff

    // Merge waive lists
    sh """
        python3 mergeWaiveList.py \
        --cur-waive-list=waives_CUR_${env.gitlabCommit}.txt \
        --latest-waive-list=waives_TOT_${targetBranchTOTCommit}.txt \
        --diff-file=diff_content.txt \
        --output-file=waives.txt
    """
    trtllm_utils.uploadArtifacts("waives*.txt", "${UPLOAD_PATH}/waive_list/")
    echo "New merged test waive list: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/waive_list/waives.txt"
}

def preparation(pipeline, testFilter, globalVars)
{
    image = "urm.nvidia.com/docker/golang:1.22"
    setupPipelineSpec = createKubernetesPodConfig(image, "package")
    trtllm_utils.launchKubernetesPod(pipeline, setupPipelineSpec, "trt-llm", {
        stage("Setup Environment") {
            setupPipelineEnvironment(pipeline, testFilter, globalVars)
        }
        stage("Merge Test Waive List") {
            mergeWaiveList(pipeline, globalVars)
        }
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
        // Step 1: Clone TRT-LLM source codes
        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
        sh "cd ${LLM_ROOT} && git config --unset-all core.hooksPath"

        // Step 2: Run guardwords scan
        def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
        if (env.alternativeTRT || isOfficialPostMergeJob) {
            trtllm_utils.checkoutSource(SCAN_REPO, SCAN_COMMIT, SCAN_ROOT, true, true)
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${SCAN_ROOT} && pip3 install -e .")
            try {
                ignoreList = [
                    "*/.git/*",
                    "*/3rdparty/*",
                    "*/cpp/tensorrt_llm/deep_ep/nvshmem_src_*.txz",
                    "*/examples/scaffolding/contrib/mcp/weather/weather.py",
                    "*/tensorrt_llm_internal_cutlass_kernels_static.tar.xz"
                ]
                sh "cd ${LLM_ROOT} && confidentiality-scan \$(find . -type f ${ignoreList.collect { "-not -path \"${it}\"" }.join(' ')}) 2>&1 | tee scan.log"
                def lastLine = sh(script: "tail -n 1 ${LLM_ROOT}/scan.log", returnStdout: true).trim()
                if (lastLine.toLowerCase().contains("error")) {
                    error "Guardwords Scan Failed."
                }
            } catch (Exception e) {
                throw e
            } finally {
                trtllm_utils.uploadArtifacts("${LLM_ROOT}/scan.log", "${UPLOAD_PATH}/guardwords-scan-results/")
                echo "Guardwords Scan Results: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/guardwords-scan-results/scan.log"
            }
        }

        // Step 3: Run pre-commit checks
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT} && python3 -u scripts/release_check.py || (git restore . && false)")

        // Step 4: Run license check
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
        sh "cd ${LLM_ROOT}/cpp && /go/bin/license_checker -config ../jenkins/license_cpp.json include tensorrt_llm"
    }

    def image = "urm.nvidia.com/docker/golang:1.22"
    stageName = "Release Check"
    trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(image, "package"), "trt-llm", {
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

def getGitlabMRChangedFile(pipeline, function, filePath="") {
    def result = null
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
                script: """
                    curl --header "PRIVATE-TOKEN: $GITLAB_API_TOKEN" \
                         --url "https://${DEFAULT_GIT_URL}/api/v4/projects/${env.gitlabMergeRequestTargetProjectId}/merge_requests/${env.gitlabMergeRequestIid}/diffs?page=${pageId}&per_page=20"
                """,
                returnStdout: true
            )
            def rawDataList = readJSON text: rawDataJson, returnPojo: true
            if (function == "getOneFileChanges") {
                if (result == null) {
                    result = ""
                }
                rawDataList.find { rawData ->
                    if (rawData.get("new_path") == filePath || rawData.get("old_path") == filePath) {
                        result = rawData.get("diff")
                        return true
                    }
                    return false
                }
                if (result != "") { break }
            } else if (function == "getChangedFileList") {
                if (result == null) {
                    result = []
                }
                rawDataList.each { rawData ->
                    result += [rawData.get("old_path"), rawData.get("new_path")]
                }
            }
            if (!rawDataList) { break }
        }
    }
    return result
}

def getGithubMRChangedFile(pipeline, githubPrApiUrl, function, filePath="") {
    def result = null
    def pageId = 0
    withCredentials([
        string(
            credentialsId: 'github-token-trtllm-ci',
            variable: 'GITHUB_API_TOKEN'
        ),
    ]) {
        while(true) {
            pageId += 1
            def rawDataJson = pipeline.sh(
                script: """
                    curl --header "Authorization: Bearer $GITHUB_API_TOKEN" \
                         --url "${githubPrApiUrl}/files?page=${pageId}&per_page=20"
                """,
                returnStdout: true
            )
            echo "rawDataJson: ${rawDataJson}"
            def rawDataList = readJSON text: rawDataJson, returnPojo: true
            if (function == "getOneFileChanges") {
                if (result == null) {
                    result = ""
                }
                rawDataList.find { rawData ->
                    if (rawData.get("filename") == filePath || rawData.get("previous_filename") == filePath) {
                        result = rawData.get("patch")
                        return true
                    }
                    return false
                }
                if (result != "") { break }
            } else if (function == "getChangedFileList") {
                if (result == null) {
                    result = []
                }
                rawDataList.each { rawData ->
                    result += [rawData.get("filename"), rawData.get("previous_filename")].findAll { it }
                }
            }
            if (!rawDataList) { break }
        }
    }
    return result
}

def getMergeRequestChangedFileList(pipeline, globalVars) {
    def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
    if (env.alternativeTRT || isOfficialPostMergeJob) {
        pipeline.echo("Force set changed file list to empty list.")
        return []
    }

    def githubPrApiUrl = globalVars[GITHUB_PR_API_URL]

    if (globalVars[CACHED_CHANGED_FILE_LIST] != null) {
        return globalVars[CACHED_CHANGED_FILE_LIST]
    }
    try {
        def changedFileList = []
        if (githubPrApiUrl != null) {
            changedFileList = getGithubMRChangedFile(pipeline, githubPrApiUrl, "getChangedFileList")
        } else {
            changedFileList = getGitlabMRChangedFile(pipeline, "getChangedFileList")
        }
        def changedFileListStr = changedFileList.join(",\n")
        pipeline.echo("The changeset of this MR is: ${changedFileListStr}.")
        globalVars[CACHED_CHANGED_FILE_LIST] = changedFileList
        return globalVars[CACHED_CHANGED_FILE_LIST]
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        pipeline.echo("Get merge request changed file list failed. Error: ${e.toString()}")
        globalVars[CACHED_CHANGED_FILE_LIST] = []
        return globalVars[CACHED_CHANGED_FILE_LIST]
    }
}

def getMergeRequestOneFileChanges(pipeline, globalVars, filePath) {
    def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
    if (env.alternativeTRT || isOfficialPostMergeJob) {
        pipeline.echo("Force set changed file diff to empty string.")
        return ""
    }

    def githubPrApiUrl = globalVars[GITHUB_PR_API_URL]
    def diff = ""

    try {
        if (githubPrApiUrl != null) {
            diff = getGithubMRChangedFile(pipeline, githubPrApiUrl, "getOneFileChanges", filePath)
        } else {
            diff = getGitlabMRChangedFile(pipeline, "getOneFileChanges", filePath)
        }
        pipeline.echo("The change of ${filePath} is: ${diff}")
        return diff
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        pipeline.echo("Get merge request one changed file diff failed. Error: ${e.toString()}")
        return ""
    }
}

def getAutoTriggerTagList(pipeline, testFilter, globalVars) {
    def autoTriggerTagList = []
    def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
    if (env.alternativeTRT || isOfficialPostMergeJob) {
        pipeline.echo("Force set auto trigger tags to empty list.")
        return autoTriggerTagList
    }
    def changedFileList = getMergeRequestChangedFileList(pipeline, globalVars)
    if (!changedFileList || changedFileList.isEmpty()) {
        return autoTriggerTagList
    }
    def specialFileToTagMap = [
        "tensorrt_llm/_torch/models/modeling_deepseekv3.py": ["-DeepSeek-"],
        "cpp/kernels/fmha_v2/": ["-FMHA-"],
        "tensorrt_llm/_torch/models/modeling_gpt_oss.py": ["-GptOss-"],
    ]
    for (file in changedFileList) {
        for (String key : specialFileToTagMap.keySet()) {
            if (file.startsWith(key)) {
                autoTriggerTagList += specialFileToTagMap[key]
            }
        }
    }
    autoTriggerTagList = autoTriggerTagList.unique()
    if (!autoTriggerTagList.isEmpty()) {
        pipeline.echo("Auto trigger tags detected: ${autoTriggerTagList.join(', ')}")
    }
    return autoTriggerTagList
}

def getMultiGpuFileChanged(pipeline, testFilter, globalVars)
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
        "cpp/include/tensorrt_llm/batch_manager/",
        "cpp/include/tensorrt_llm/executor/",
        "cpp/include/tensorrt_llm/runtime/gptJsonConfig.h",
        "cpp/include/tensorrt_llm/runtime/utils/mpiUtils.h",
        "cpp/include/tensorrt_llm/runtime/utils/multiDeviceUtils.h",
        "cpp/include/tensorrt_llm/runtime/worldConfig.h",
        "cpp/tensorrt_llm/batch_manager/",
        "cpp/tensorrt_llm/executor/",
        "cpp/tensorrt_llm/executor_worker/",
        "cpp/tensorrt_llm/kernels/communicationKernels/",
        "cpp/tensorrt_llm/kernels/customAllReduceKernels.cu",
        "cpp/tensorrt_llm/kernels/customAllReduceKernels.h",
        "cpp/tensorrt_llm/kernels/gptKernels.cu",
        "cpp/tensorrt_llm/kernels/gptKernels.h",
        "cpp/tensorrt_llm/kernels/moe",
        "cpp/tensorrt_llm/kernels/unfusedAttentionKernels.cu",
        "cpp/tensorrt_llm/kernels/unfusedAttentionKernels.h",
        "cpp/tensorrt_llm/kernels/userbuffers/",
        "cpp/tensorrt_llm/plugins/cpSplitPlugin/cpSplitPlugin.cpp",
        "cpp/tensorrt_llm/plugins/cpSplitPlugin/cpSplitPlugin.h",
        "cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.cpp",
        "cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h",
        "cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp",
        "cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h",
        "cpp/tensorrt_llm/plugins/ncclPlugin/",
        "cpp/tensorrt_llm/pybind/",
        "cpp/tensorrt_llm/runtime/ipcUtils.cpp",
        "cpp/tensorrt_llm/runtime/ncclCommunicator.cpp",
        "cpp/tensorrt_llm/runtime/utils/mpiUtils.cpp",
        "cpp/tensorrt_llm/runtime/workerPool.h",
        "cpp/tensorrt_llm/runtime/worldConfig.cpp",
        "cpp/tensorrt_llm/thop/allgatherOp.cpp",
        "cpp/tensorrt_llm/thop/allreduceOp.cpp",
        "cpp/tensorrt_llm/thop/reducescatterOp.cpp",
        "cpp/tests/e2e_tests/batch_manager/",
        "cpp/tests/e2e_tests/executor/",
        "cpp/tests/unit_tests/multi_gpu/",
        "jenkins/L0_Test.groovy",
        "tensorrt_llm/_ipc_utils.py",
        "tensorrt_llm/_torch/compilation/patterns/ar_residual_norm.py",
        "tensorrt_llm/_torch/compilation/patterns/ub_allreduce.py",
        "tensorrt_llm/_torch/custom_ops/userbuffers_custom_ops.py",
        "tensorrt_llm/_torch/models/modeling_llama.py",
        "tensorrt_llm/_torch/modules/fused_moe/",
        "tensorrt_llm/_torch/pyexecutor/_util.py",
        "tensorrt_llm/_torch/pyexecutor/model_engine.py",
        "tensorrt_llm/_torch/pyexecutor/py_executor.py",
        "tensorrt_llm/evaluate/json_mode_eval.py",
        "tensorrt_llm/evaluate/mmlu.py",
        "tensorrt_llm/executor/",
        "tensorrt_llm/functional.py",
        "tensorrt_llm/llmapi/",
        "tensorrt_llm/mapping.py",
        "tensorrt_llm/models/llama/",
        "tensorrt_llm/parameter.py",
        "tensorrt_llm/serve/",
        "tests/integration/defs/cpp/test_multi_gpu.py",
        "tests/integration/test_lists/test-db/l0_dgx_h100.yml",
        "tests/integration/test_lists/test-db/l0_dgx_h200.yml",
        "tests/unittest/_torch/auto_deploy/unit/multigpu",
        "tests/unittest/_torch/multi_gpu/",
        "tests/unittest/_torch/multi_gpu_modeling/",
        "tests/unittest/disaggregated/",
        "tests/unittest/llmapi/test_llm_multi_gpu.py",
        "tests/unittest/llmapi/test_llm_multi_gpu_pytorch.py",
    ]

    def changedFileList = getMergeRequestChangedFileList(pipeline, globalVars)
    if (!changedFileList || changedFileList.isEmpty()) {
        return false
    }

    def changedFileListStr = ","
    def relatedFileChanged = false
    try {
        changedFileListStr = changedFileList.join(", ")
        relatedFileChanged = relatedFileList.any { it ->
            if (changedFileListStr.contains(it)) {
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
        pipeline.echo("getMultiGpuFileChanged failed execution. Error: ${e.toString()}")
    }
    if (relatedFileChanged) {
        pipeline.echo("Detect multi-GPU related files changed.")
    }
    return relatedFileChanged
}

def getOnlyOneGroupChanged(pipeline, testFilter, globalVars) {
    def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
    if (env.alternativeTRT || isOfficialPostMergeJob) {
        pipeline.echo("Force set ONLY_ONE_GROUP_CHANGED \"\".")
        return ""
    }
    def groupFileMap = [
        "Docs": [ // TODO: Add more docs path to the list, e.g. *.md files in other directories
            "docs/",
        ],
        "PyTorch": [
            "tensorrt_llm/_torch/",
            "tensorrt_llm/scaffolding/",
            "tests/unittest/_torch/",
            "tests/unittest/scaffolding/",
            "tests/unittest/llmapi/test_llm_pytorch.py",
            "tests/unittest/llmapi/test_llm_multi_gpu_pytorch.py",
            "tests/integration/defs/accuracy/test_llm_api_pytorch.py",
            "tests/integration/defs/disaggregated/",
            "examples/auto_deploy",
            "examples/disaggregated",
            "examples/pytorch/",
            "examples/scaffolding/",
            "docs/",
        ],
        "Triton": [
            "tests/integration/defs/triton_server/",
            "triton_backend/",
        ],
    ]

    def changedFileList = getMergeRequestChangedFileList(pipeline, globalVars)
    if (!changedFileList || changedFileList.isEmpty()) {
        return ""
    }

    for (group in groupFileMap.keySet()) {
        def groupPrefixes = groupFileMap[group]
        def allFilesInGroup = changedFileList.every { file ->
            groupPrefixes.any { prefix -> file.startsWith(prefix) }
        }

        if (allFilesInGroup) {
            pipeline.echo("Only ${group} files changed.")
            return group
        } else {
            def nonGroupFile = changedFileList.find { file ->
                !groupPrefixes.any { prefix -> file.startsWith(prefix) }
            }
            if (nonGroupFile != null) {
                pipeline.echo("Found non-${group} file: ${nonGroupFile}")
            }
        }
    }

    return ""
}

def collectTestResults(pipeline, testFilter)
{
    collectResultPodSpec = createKubernetesPodConfig("", "agent")
    trtllm_utils.launchKubernetesPod(pipeline, collectResultPodSpec, "alpine", {
        stage ("Collect test result") {
            sh "rm -rf **/*.xml *.tar.gz"

            testResultLink = "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}/test-results"

            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add --no-cache curl")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add python3")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget ${testResultLink}/", allowStepFailed: true)
            sh "cat index.html | grep \"tar.gz\" | cut -d \"\\\"\" -f 2 > result_file_names.txt"
            sh "cat result_file_names.txt"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cat result_file_names.txt | xargs -n1 -I {} wget -c -nv ${testResultLink}/{}", allowStepFailed: true)
            sh "ls -l | grep \"tar.gz\" || true"
            resultFileNumber = sh(script: "cat result_file_names.txt | wc -l", returnStdout: true)
            resultFileDownloadedNumber = sh(script: "ls -l | grep \"tar.gz\" | wc -l", returnStdout: true)
            echo "Result File Number: ${resultFileNumber}, Downloaded: ${resultFileDownloadedNumber}"

            sh "find . -name results-\\*.tar.gz -type f -exec tar -zxvf {} \\; || true"
            trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
            if (testFilter[(IS_POST_MERGE)]) {
                try {
                    sh "python3 llm/scripts/generate_duration.py --duration-file=new_test_duration.json"
                    trtllm_utils.uploadArtifacts("new_test_duration.json", "${UPLOAD_PATH}/test-results/")
                } catch (Exception e) {
                    // No need to fail the stage if the duration file generation fails
                    echo "An error occurred while generating or uploading the duration file: ${e.toString()}"
                }
            }

            junit(testResults: '**/results*.xml', allowEmptyResults : true)
        } // Collect test result stage
        stage("Rerun report") {
            sh "rm -rf rerun && mkdir -p rerun"
            sh "find . -type f -wholename '*/rerun_results.xml' -exec sh -c 'mv \"{}\" \"rerun/\$(basename \$(dirname \"{}\"))_rerun_results.xml\"' \\; || true"
            sh "find rerun -type f"
            def rerunFileCount = sh(returnStdout: true, script: 'find rerun -type f | wc -l').replaceAll("\\s","").toInteger()
            if (rerunFileCount == 0) {
                echo "Rerun report is skipped because there is no rerun test data file."
                return
            }
            def xmlFiles = findFiles(glob: 'rerun/**/*.xml')
            def xmlFileList = xmlFiles.collect { it.path }
            def inputfiles = xmlFileList.join(',')
            echo "inputfiles: ${inputfiles}"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add python3")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add py3-pip")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 config set global.break-system-packages true")
            sh """
                python3 llm/jenkins/scripts/test_rerun.py \
                generate_rerun_report \
                --output-file=rerun/rerun_report.xml \
                --input-files=${inputfiles}
            """
            trtllm_utils.uploadArtifacts("rerun/rerun_report.html", "${UPLOAD_PATH}/test-results/")
            echo "Rerun report: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/test-results/rerun_report.html"
            def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
            if (env.alternativeTRT || isOfficialPostMergeJob) {
                catchError(
                    buildResult: 'FAILURE',
                    stageResult: 'FAILURE') {
                    error "Some failed tests were reruned, please check the rerun report."
                }
            } else {
                catchError(
                    buildResult: 'SUCCESS',
                    stageResult: 'UNSTABLE') {
                    error "Some failed tests were reruned, please check the rerun report."
                }
            }
        } // Rerun report stage
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
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "apk add py3-pip")
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 config set global.break-system-packages true")
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install coverage")
                sh "coverage --version"

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
                sh "cd cov && coverage report -i"   // -i: ignore errors. Ignore the error that the source code file cannot be found.
                sh "cd cov && coverage html -d test_coverage_html -i"
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

def launchJob(jobName, reuseBuild, enableFailFast, globalVars, platform="x86_64", additionalParameters = [:]) {
    def parameters = getCommonParameters()
    String globalVarsJson = writeJSON returnText: true, json: globalVars
    parameters += [
        'enableFailFast': enableFailFast,
        'globalVars': globalVarsJson,
    ] + additionalParameters

    if (env.alternativeTRT && platform == "x86_64") {
        parameters += [
            'alternativeTRT': env.alternativeTRT,
        ]
    }

    if (env.alternativeTrtSBSA && platform == "SBSA") {
        parameters += [
            'alternativeTRT': env.alternativeTrtSBSA,
        ]
    }

    if (env.testPhase2StageName) {
        parameters += [
            'testPhase2StageName': env.testPhase2StageName,
        ]
    }

    if (reuseBuild) {
        parameters['reuseArtifactPath'] = "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${reuseBuild}"
    }

    echo "Trigger ${jobName} job, params: ${parameters}"

    def status = triggerJob(jobName, parameters)
    if (status != "SUCCESS") {
        error "Downstream job did not succeed"
    }
    return status
}

def launchStages(pipeline, reuseBuild, testFilter, enableFailFast, globalVars)
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
                    def additionalParameters = [
                        'dockerImage': globalVars["LLM_DOCKER_IMAGE"],
                        'wheelDockerImagePy310': globalVars["LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE"],
                        'wheelDockerImagePy312': globalVars["LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE"],
                    ]
                    launchJob("/LLM/helpers/Build-x86_64", reuseBuild, enableFailFast, globalVars, "x86_64", additionalParameters)
                }
                def testStageName = "[Test-x86_64-Single-GPU] ${env.localJobCredentials ? "Remote Run" : "Run"}"
                def singleGpuTestFailed = false
                stage(testStageName) {
                    if (X86_TEST_CHOICE == STAGE_CHOICE_SKIP) {
                        echo "x86_64 test job is skipped due to Jenkins configuration"
                        return
                    }
                    try {
                        String testFilterJson = writeJSON returnText: true, json: testFilter
                        def additionalParameters = [
                            'testFilter': testFilterJson,
                            'dockerImage': globalVars["LLM_DOCKER_IMAGE"],
                            'wheelDockerImagePy310': globalVars["LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE"],
                            'wheelDockerImagePy312': globalVars["LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE"],
                        ]

                        launchJob("L0_Test-x86_64-Single-GPU", false, enableFailFast, globalVars, "x86_64", additionalParameters)
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
                            catchError(
                                buildResult: 'FAILURE',
                                stageResult: 'FAILURE') {
                                error "x86_64 single-GPU test failed"
                            }
                            singleGpuTestFailed = true
                        }
                    }
                }

                def requireMultiGpuTesting = currentBuild.description?.contains("Require x86_64 Multi-GPU Testing") ?: false
                echo "requireMultiGpuTesting: ${requireMultiGpuTesting}"
                if (!requireMultiGpuTesting) {
                    if (singleGpuTestFailed) {
                        error "x86_64 single-GPU test failed"
                    }
                    return
                }

                if (singleGpuTestFailed) {
                    if (env.JOB_NAME ==~ /.*PostMerge.*/) {
                        echo "In the official post-merge pipeline, x86_64 single-GPU test failed, whereas multi-GPU test is still kept running."
                    } else {
                        stage("[Test-x86_64-Multi-GPU] Blocked") {
                            error "This pipeline requires running multi-GPU test, but x86_64 single-GPU test has failed."
                        }
                        return
                    }
                }

                testStageName = "[Test-x86_64-Multi-GPU] ${env.localJobCredentials ? "Remote Run" : "Run"}"
                stage(testStageName) {
                    if (X86_TEST_CHOICE == STAGE_CHOICE_SKIP) {
                        echo "x86_64 test job is skipped due to Jenkins configuration"
                        return
                    }
                    try {
                        def testFilterJson = writeJSON returnText: true, json: testFilter
                        def additionalParameters = [
                            'testFilter': testFilterJson,
                            'dockerImage': globalVars["LLM_DOCKER_IMAGE"],
                            'wheelDockerImagePy310': globalVars["LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE"],
                            'wheelDockerImagePy312': globalVars["LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE"],
                        ]

                        launchJob("L0_Test-x86_64-Multi-GPU", false, enableFailFast, globalVars, "x86_64", additionalParameters)

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
                def testStageName = "[Test-SBSA-Single-GPU] ${env.localJobCredentials ? "Remote Run" : "Run"}"
                def singleGpuTestFailed = false

                if (testFilter[(ONLY_ONE_GROUP_CHANGED)] == "Docs") {
                    echo "SBSA build job is skipped due to Jenkins configuration or conditional pipeline run"
                    return
                }

                stage("Build") {
                    def additionalParameters = [
                        "dockerImage": globalVars["LLM_SBSA_DOCKER_IMAGE"],
                    ]
                    launchJob("/LLM/helpers/Build-SBSA", reuseBuild, enableFailFast, globalVars, "SBSA", additionalParameters)
                }
                stage(testStageName) {
                    if (SBSA_TEST_CHOICE == STAGE_CHOICE_SKIP) {
                        echo "SBSA test job is skipped due to Jenkins configuration"
                        return
                    }
                    try {
                        String testFilterJson = writeJSON returnText: true, json: testFilter
                        def additionalParameters = [
                            'testFilter': testFilterJson,
                            "dockerImage": globalVars["LLM_SBSA_DOCKER_IMAGE"],
                        ]

                        launchJob("L0_Test-SBSA-Single-GPU", false, enableFailFast, globalVars, "SBSA", additionalParameters)
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
                            catchError(
                                buildResult: 'FAILURE',
                                stageResult: 'FAILURE') {
                                error "SBSA single-GPU test failed"
                            }
                            singleGpuTestFailed = true
                        }
                    }
                }

                def requireMultiGpuTesting = currentBuild.description?.contains("Require SBSA Multi-GPU Testing") ?: false
                echo "requireMultiGpuTesting: ${requireMultiGpuTesting}"
                if (!requireMultiGpuTesting) {
                    if (singleGpuTestFailed) {
                        error "SBSA single-GPU test failed"
                    }
                    return
                }

                if (singleGpuTestFailed) {
                    if (env.JOB_NAME ==~ /.*PostMerge.*/) {
                        echo "In the official post-merge pipeline, SBSA single-GPU test failed, whereas multi-GPU test is still kept running."
                    } else {
                        stage("[Test-SBSA-Multi-GPU] Blocked") {
                            error "This pipeline requires running SBSA multi-GPU test, but SBSA single-GPU test has failed."
                        }
                        return
                    }
                }

                testStageName = "[Test-SBSA-Multi-GPU] ${env.localJobCredentials ? "Remote Run" : "Run"}"
                stage(testStageName) {
                    if (SBSA_TEST_CHOICE == STAGE_CHOICE_SKIP) {
                        echo "SBSA test job is skipped due to Jenkins configuration"
                        return
                    }
                    try {
                        def testFilterJson = writeJSON returnText: true, json: testFilter
                        def additionalParameters = [
                            'testFilter': testFilterJson,
                            "dockerImage": globalVars["LLM_SBSA_DOCKER_IMAGE"],
                        ]

                        launchJob("L0_Test-SBSA-Multi-GPU", false, enableFailFast, globalVars, "SBSA", additionalParameters)

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
    def dockerBuildJob = [
        "Build-Docker-Images": {
            script {
                stage("[Build-Docker-Images] Remote Run") {
                    def branch = env.gitlabBranch ? env.gitlabBranch : "main"
                    if (globalVars[GITHUB_PR_API_URL]) {
                        branch = "github-pr-" + globalVars[GITHUB_PR_API_URL].split('/').last()
                    }

                    def additionalParameters = [
                        'branch': branch,
                        'action': "push",
                        'triggerType': env.JOB_NAME ==~ /.*PostMerge.*/ ? "post-merge" : "pre-merge",
                        'runSanityCheck': env.JOB_NAME ==~ /.*PostMerge.*/ ? true : false,
                    ]

                    launchJob("/LLM/helpers/BuildDockerImages", false, enableFailFast, globalVars, "x86_64", additionalParameters)
                }
            }
        }
    ]

    if (env.JOB_NAME ==~ /.*PostMerge.*/) {
        stages += dockerBuildJob
    }
    if (testFilter[(TEST_STAGE_LIST)]?.contains("Build-Docker-Images") || testFilter[(EXTRA_STAGE_LIST)]?.contains("Build-Docker-Images")) {
        stages += dockerBuildJob
        testFilter[(TEST_STAGE_LIST)]?.remove("Build-Docker-Images")
        testFilter[(EXTRA_STAGE_LIST)]?.remove("Build-Docker-Images")
        echo "Will run Build-Docker-Images job"
        stages.remove("x86_64-linux")
        stages.remove("SBSA-linux")
        echo "Build-Docker-Images job is set explicitly. Both x86_64-linux and SBSA-linux sub-pipelines will be disabled."
    }

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
        stage("Preparation")
        {
            steps
            {
                script {
                    preparation(this, testFilter, globalVars)
                    println globalVars
                    globalVars[ACTION_INFO] = trtllm_utils.setupPipelineDescription(this, globalVars[ACTION_INFO])
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
                        // globalVars[CACHED_CHANGED_FILE_LIST] is only used in setupPipelineEnvironment
                        // Reset it to null to workaround the "Argument list too long" error
                        globalVars[CACHED_CHANGED_FILE_LIST] = null
                        launchStages(this, reuseBuild, testFilter, enableFailFast, globalVars)
                    }
                }
            }
        }
    } // stages
} // pipeline
