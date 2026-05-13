@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.InterruptedException
import groovy.transform.Field
import groovy.json.JsonOutput
import groovy.json.JsonSlurper
import com.nvidia.bloom.KubernetesManager
import com.nvidia.bloom.Constants
import com.nvidia.bloom.Logger
import com.nvidia.bloom.JobBuilder
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

RELEASE_CHECK_CHOICE = env.releaseCheckChoice ? env.releaseCheckChoice : STAGE_CHOICE_NORMAL
BUILD_CHECK_CHOICE = env.buildCheckChoice ? env.buildCheckChoice : STAGE_CHOICE_NORMAL
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

GEN_POST_MERGE_BUILDS_ONLY = (env.JOB_NAME?.contains("GenPostMergeBuilds") ?: false)

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
def REUSE_TEST = "reuse_test"   // Determine if the pipeline should reuse test results in a stage from the previous pipelines.
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
@Field
def CBTS_RESULT = "cbts_result"

def testFilter = [
    (REUSE_TEST): gitlabParamsFromBot.get(REUSE_TEST, null),
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
    (CBTS_RESULT): null,
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
// GenPostMergeBuilds pipelines do not update GitLab status.
boolean enableUpdateGitlabStatus =
    !GEN_POST_MERGE_BUILDS_ONLY &&
    !testFilter[ENABLE_SKIP_TEST] &&
    !testFilter[ONLY_MULTI_GPU_TEST] &&
    !testFilter[DISABLE_MULTI_GPU_TEST] &&
    !testFilter[DEBUG_MODE] &&
    testFilter[GPU_TYPE_LIST] == null &&
    testFilter[TEST_STAGE_LIST] == null &&
    testFilter[TEST_BACKEND] == null


def createKubernetesPodConfig(image, type, arch = "amd64")
{
    def targetCould = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux"""
    def containerConfig = ""
    def nodeLabelPrefix = ""

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
                        memory: 20Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 20Gi
                        ephemeral-storage: 25Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    }
    def nodeLabel = trtllm_utils.generateNodeLabel(nodeLabelPrefix)
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
                                - "qa_only"
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
    if (!GEN_POST_MERGE_BUILDS_ONLY) {
        updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: 'running'
    }
    echo "Using GitLab repo: ${LLM_REPO}."
    sh "git config --global --add safe.directory \"*\""
    // NB: getContainerURIs reads files in ${LLM_ROOT}/jenkins/
    if (env.gitlabMergeRequestLastCommit) {
        env.gitlabCommit = env.gitlabMergeRequestLastCommit
        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, false, true)
    } else {
        branch = env.gitlabBranch ? env.gitlabBranch : "main"
        trtllm_utils.checkoutSource(LLM_REPO, branch, LLM_ROOT, false, true)
        checkoutCommit = sh (script: "cd ${LLM_ROOT} && git rev-parse HEAD",returnStdout: true).trim()
        env.gitlabCommit = checkoutCommit
    }
    echo "Env.gitlabMergeRequestLastCommit: ${env.gitlabMergeRequestLastCommit}."
    echo "Freeze GitLab commit. Branch: ${env.gitlabBranch}. Commit: ${env.gitlabCommit}."
    testFilter[(MULTI_GPU_FILE_CHANGED)] = getMultiGpuFileChanged(pipeline, testFilter, globalVars)
    testFilter[(ONLY_ONE_GROUP_CHANGED)] = getOnlyOneGroupChanged(pipeline, testFilter, globalVars)
    testFilter[(AUTO_TRIGGER_TAG_LIST)] = getAutoTriggerTagList(pipeline, testFilter, globalVars)
    testFilter[(CBTS_RESULT)] = getCbtsResult(pipeline, testFilter, globalVars)
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

    def targetBranchTOTCommit = ""
    def isGetTOTWaiveList = false
    try {
        withCredentials([usernamePassword(credentialsId: 'svc_tensorrt_gitlab_api_token', usernameVariable: 'GITHUB_USER', passwordVariable: 'GITHUB_PASSWORD')]) {
            def apiUrl = "https://api.github.com/repos/NVIDIA/TensorRT-LLM/commits?sha=${targetBranch}&per_page=1"
            def connection = new URL(apiUrl).openConnection()
            connection.setRequestProperty("Authorization", "Basic " + "${GITHUB_USER}:${GITHUB_PASSWORD}".bytes.encodeBase64().toString())
            connection.setRequestMethod("GET")
            def response = connection.inputStream.text
            def json = new JsonSlurper().parseText(response)
            targetBranchTOTCommit = json[0].sha
        }
        echo "Target branch TOT commit: ${targetBranchTOTCommit}"
        sh "wget https://urm.nvidia.com/artifactory/vcs-remote/NVIDIA/TensorRT-LLM/raw/${targetBranchTOTCommit}/tests/integration/test_lists/waives.txt -O waives_TOT_${targetBranchTOTCommit}.txt"
        isGetTOTWaiveList = true
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        echo "Failed to checkout TOT waive list from public GitHub repository. Error: ${e.toString()}"
    }

    if (!isGetTOTWaiveList) {
        try {
            withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
                trtllm_utils.checkoutFile(DEFAULT_LLM_REPO, targetBranch, "tests/integration/test_lists/waives.txt", ".")
            }
            sh "mv waives.txt waives_TOT_.txt"
            isGetTOTWaiveList = true
        } catch (InterruptedException e) {
            throw e
        } catch (Exception e) {
            echo "Failed to checkout TOT waive list from internal GitLab repository. Error: ${e.toString()}"
        }
    }

    if (!isGetTOTWaiveList) {
        catchError(
            buildResult: 'SUCCESS',
            stageResult: 'UNSTABLE') {
            error "Failed to get TOT waive list. Fallback to use the default test waive list from the PR."
        }
        return
    }

    try {
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
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        catchError(
            buildResult: 'SUCCESS',
            stageResult: 'UNSTABLE') {
            error "Merge test waive list failed. Fallback to use the default test waive list from the PR. Error: ${e.toString()}"
        }
    }
}

def preparation(pipeline, testFilter, globalVars)
{
    image = "urm.nvidia.com/docker/buildpack-deps:trixie-scm"
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

def launchReleaseCheck(pipeline, globalVars)
{
    stages = {
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y python3-pip")
        sh "pip3 config set global.break-system-packages true"
        sh "git config --global --add safe.directory \"*\""
        // Step 1: Clone TRT-LLM source codes
        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, false, true)
        sh "cd ${LLM_ROOT} && git config --unset-all core.hooksPath"

        // Step 2: Run guardwords scan
        def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
        if (env.alternativeTRT || isOfficialPostMergeJob) {
            trtllm_utils.checkoutSource(SCAN_REPO, SCAN_COMMIT, SCAN_ROOT, false, true)
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${SCAN_ROOT} && pip3 install -e .")
            try {
                ignoreList = [
                    "*/.git/*",
                    "*/3rdparty/*",
                    "*/cpp/tensorrt_llm/deep_ep/nvshmem_src_*.txz",
                    "*/examples/scaffolding/contrib/mcp/weather/weather.py",
                    "*/tensorrt_llm_internal_cutlass_kernels_static.tar.xz",
                    "*/triton_kernels/*.py"
                ]
                sh "cd ${LLM_ROOT} && confidentiality-scan \$(find . -type f ${ignoreList.collect { "-not -path \"${it}\"" }.join(' ')}) 2>&1 | tee scan.log"
                def lastLine = sh(script: "tail -n 1 ${LLM_ROOT}/scan.log", returnStdout: true).trim()
                if (lastLine.toLowerCase().contains("error")) {
                    error "GUARDWORDS_WARN: Guardwords Scan Failed."
                }
            } catch (InterruptedException e) {
                throw e
            } catch (Exception e) {
                catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') {
                    error "Release Check failed (warn-only): ${e.getMessage()}"
                }
            } finally {
                trtllm_utils.uploadArtifacts("${LLM_ROOT}/scan.log", "${UPLOAD_PATH}/guardwords-scan-results/")
                echo "Guardwords Scan Results: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/guardwords-scan-results/scan.log"
            }
        }

        // Step 3: Run pre-commit checks
        // Post-merge CI runs on all files; pre-merge CI runs only on changed files.
        def precommitArgs = "-a"
        if (!(env.JOB_NAME ==~ /.*PostMerge.*/ || env.alternativeTRT)) {
            // Use GitLab/GitHub API to get the exact list of changed files in this MR.
            // This avoids git history depth issues with shallow clones.
            def changedFileList = getMergeRequestChangedFileList(pipeline, globalVars)
            if (changedFileList && !changedFileList.isEmpty()) {
                def changedFilesPath = "${LLM_ROOT}/changed_files.txt"
                writeFile file: changedFilesPath, text: changedFileList.unique().join("\n")
                // Script runs after "cd ${LLM_ROOT}", so use relative path
                precommitArgs = "--files-from changed_files.txt"
                echo "Pre-commit will check ${changedFileList.unique().size()} changed file(s)"
            } else {
                echo "Could not determine changed files, falling back to all files"
            }
        }
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT} && python3 -u scripts/release_check.py ${precommitArgs} || (git restore . && false)")

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
    stageName = "Release-Check"
    trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(image, "package"), "trt-llm", {
        stage("[${stageName}] Run") {
            if (RELEASE_CHECK_CHOICE == STAGE_CHOICE_SKIP) {
                echo "Release Check job is skipped due to Jenkins configuration"
                return
            }
            try {
                echoNodeAndGpuInfo(pipeline, stageName)
                stages()
            } catch (InterruptedException e) {
                throw e
            } catch (Exception e) {
                if (RELEASE_CHECK_CHOICE == STAGE_CHOICE_IGNORE) {
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
        usernamePassword(
            credentialsId: 'github-cred-trtllm-ci',
            usernameVariable: 'NOT_USED_YET',
            passwordVariable: 'GITHUB_API_TOKEN'
        ),
    ]) {
        while(true) {
            pageId += 1
            def rawDataJson = pipeline.sh(
                script: """
                    curl --header "Authorization: Bearer \${GITHUB_API_TOKEN}" \
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
    // Note: This function intentionally propagates exceptions to the caller.
    // If there is an error to get the changed file diff, skip merging the waive list.
    def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
    if (env.alternativeTRT || isOfficialPostMergeJob) {
        pipeline.echo("Force set changed file diff to empty string.")
        return ""
    }

    def githubPrApiUrl = globalVars[GITHUB_PR_API_URL]
    def diff = ""

    if (githubPrApiUrl != null) {
        diff = getGithubMRChangedFile(pipeline, githubPrApiUrl, "getOneFileChanges", filePath)
    } else {
        diff = getGitlabMRChangedFile(pipeline, "getOneFileChanges", filePath)
    }
    pipeline.echo("The change of ${filePath} is: ${diff}")
    return diff
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
        "tests/integration/defs/triton_server/": ["-Triton-"],
        "triton_backend/": ["-Triton-"],
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

// ============================================================================
// CBTS (Change-Based Testing Selection)
//
// Calls jenkins/scripts/cbts/main.py with PR changed_files + diffs and returns
// a result map (or null = defer to existing filter chain). Result keys:
// scope, affected_stages, reasons, test_db_dir_override, affected_stage_test_counts.
// CBTS narrows test cases only — Build always runs. See cbts/README.md.
// ============================================================================

def getCbtsResult(pipeline, testFilter, globalVars)
{
    def isOfficialPostMergeJob = (env.JOB_NAME ==~ /.*PostMerge.*/)
    if (env.alternativeTRT || isOfficialPostMergeJob) {
        pipeline.echo("CBTS: deferring — post-merge job or alternativeTRT set")
        return null
    }

    // CBTS only activates on bare `/bot run`. If the user specified any
    // stage-selection flag, defer entirely to their explicit choice.
    def triggeredFlags = _cbtsTriggeredUserFlags(testFilter)
    if (!triggeredFlags.isEmpty()) {
        pipeline.echo("CBTS: deferring — user-specified /bot run flag(s): ${triggeredFlags.join(', ')}")
        return null
    }

    def changedFiles = getMergeRequestChangedFileList(pipeline, globalVars).unique()
    if (!changedFiles) {
        pipeline.echo("CBTS: deferring — no changed files detected")
        return null
    }

    try {
        // pyyaml is needed by main.py's blocks.py to parse test-db YAMLs.
        sh "apt-get update -qq && apt-get install -y -qq python3-yaml"

        // Ask Python which file patterns need diffs, fetch them.
        def patternsOut = sh(
            script: "cd ${LLM_ROOT} && python3 jenkins/scripts/cbts/main.py --list-needed-diffs",
            returnStdout: true,
        ).trim()
        def needsDiffFor = patternsOut ? patternsOut.readLines().collect { it.trim() }.findAll { it } : []
        def diffs = [:]
        for (f in changedFiles) {
            if (_cbtsMatchesAnyPattern(f, needsDiffFor)) {
                diffs[f] = getMergeRequestOneFileChanges(pipeline, globalVars, f)
            }
        }

        // Write INPUT_JSON; Python reads stages/yaml itself.
        def inputJson = groovy.json.JsonOutput.toJson([
            changed_files: changedFiles,
            diffs: diffs,
            post_merge: testFilter[(IS_POST_MERGE)] ?: false,
        ])
        def inputPath = "${LLM_ROOT}/cbts_input.json"
        writeFile file: inputPath, text: inputJson

        def output = sh(
            script: "cd ${LLM_ROOT} && python3 jenkins/scripts/cbts/main.py cbts_input.json",
            returnStdout: true,
        )

        def result = _cbtsParseSelectionResult(output)
        if (result.scope == null) {
            pipeline.echo("CBTS: deferring — Python returned scope=null. " +
                          "Reasons: ${result.reasons.join('; ')}")
            return null
        }
        // Piggyback input JSON on testFilter so each L0_Test stage agent can
        // re-run main.py and regenerate cbts_test_db/ locally. Capped at
        // 256 KB; oversize → drop piggyback, Layer 3 falls back to source.
        final int CBTS_INPUT_PIGGYBACK_MAX_BYTES = 256000
        def inputJsonSize = inputJson.length()
        if (inputJsonSize <= CBTS_INPUT_PIGGYBACK_MAX_BYTES) {
            result.cbts_input_json = inputJson
            pipeline.echo("CBTS Layer 3: cbts_input_json piggyback enabled (${inputJsonSize} bytes)")
        } else {
            pipeline.echo("CBTS Layer 3: cbts_input_json is ${inputJsonSize} bytes, " +
                          "exceeds ${CBTS_INPUT_PIGGYBACK_MAX_BYTES}-byte piggyback limit; " +
                          "downstream stages will fall back to source test-db " +
                          "(Layer 2 stage filtering still applies)")
        }
        pipeline.echo("CBTS: scope=${result.scope}, " +
                      "stages=${result.affected_stages.size()}")
        return result
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        pipeline.echo("CBTS failed, falling back to full run: ${e}")
        return null
    }
}

// Translate an Ant-style glob to a regex:
//   **/  zero or more path segments
//   **   any chars (including /)
//   *    any chars except /
//   ?    single char except /
def _cbtsGlobToRegex(String glob)
{
    def escaped = glob.collect { c ->
        (c == '*' || c == '?') ? c
            : ('.+()[]{}|^$\\'.contains(c) ? '\\' + c : c)
    }.join('')
    return '^' + escaped
        .replace('**/', '__CBTSDOUBLESLASH__')
        .replace('**',  '__CBTSDOUBLESTAR__')
        .replace('*',   '[^/]*')
        .replace('?',   '[^/]')
        .replace('__CBTSDOUBLESLASH__', '(?:.*/)?')
        .replace('__CBTSDOUBLESTAR__',  '.*') + '$'
}

def _cbtsMatchesAnyPattern(String filePath, List patterns)
{
    return patterns.any { filePath ==~ _cbtsGlobToRegex(it) }
}

// Returns user-set stage-selection flags that should force CBTS to defer.
// IS_POST_MERGE and orthogonal flags (REUSE_*, DEBUG_MODE, DETAILED_LOG, ...)
// are intentionally absent.
def _cbtsTriggeredUserFlags(testFilter)
{
    def deferFlags = [
        ENABLE_SKIP_TEST, TEST_STAGE_LIST, EXTRA_STAGE_LIST, GPU_TYPE_LIST,
        TEST_BACKEND, ADD_MULTI_GPU_TEST, ONLY_MULTI_GPU_TEST, DISABLE_MULTI_GPU_TEST,
    ]
    return deferFlags.findAll { testFilter[it] }
                     .collect { "${it}=${testFilter[it]}" }
}

// Parse CBTS JSON stdout into a map. `scope == null` → no decision; caller
// logs reasons and defers.
def _cbtsParseSelectionResult(String text)
{
    def data = new groovy.json.JsonSlurper().parseText(text)
    return [
        scope: data.scope,
        affected_stages: data.affected_stages ?: [],
        reasons: data.reasons ?: [],
        test_db_dir_override: data.test_db_dir_override,
        affected_stage_test_counts: data.affected_stage_test_counts ?: [:],
        // Explicit null check preserves `false`; default True is safe.
        sanity_required: data.sanity_required != null ? data.sanity_required : true,
        perfsanity_required: data.perfsanity_required != null ? data.perfsanity_required : true,
    ]
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
        "cpp/tensorrt_llm/kernels/fmhaDispatcher.cpp",
        "cpp/tensorrt_llm/kernels/fmhaDispatcher.h",
        "cpp/tensorrt_llm/kernels/gptKernels.cu",
        "cpp/tensorrt_llm/kernels/gptKernels.h",
        "cpp/tensorrt_llm/kernels/moe",
        "cpp/tensorrt_llm/kernels/trtllmGenKernels/fmha/",
        "cpp/tensorrt_llm/kernels/unfusedAttentionKernels.cu",
        "cpp/tensorrt_llm/kernels/unfusedAttentionKernels.h",
        "cpp/tensorrt_llm/kernels/userbuffers/",
        "cpp/tensorrt_llm/kernels/xqaDispatcher.cpp",
        "cpp/tensorrt_llm/kernels/xqaDispatcher.h",
        "cpp/tensorrt_llm/plugins/cpSplitPlugin/cpSplitPlugin.cpp",
        "cpp/tensorrt_llm/plugins/cpSplitPlugin/cpSplitPlugin.h",
        "cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.cpp",
        "cpp/tensorrt_llm/plugins/gptAttentionCommon/gptAttentionCommon.h",
        "cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.cpp",
        "cpp/tensorrt_llm/plugins/gptAttentionPlugin/gptAttentionPlugin.h",
        "cpp/tensorrt_llm/plugins/ncclPlugin/",
        "cpp/tensorrt_llm/nanobind/",
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
        "tensorrt_llm/_torch/custom_ops/torch_custom_ops.py",
        "tensorrt_llm/_torch/custom_ops/userbuffers_custom_ops.py",
        "tensorrt_llm/_torch/distributed/",
        "tensorrt_llm/_torch/models/modeling_llama.py",
        "tensorrt_llm/_torch/models/modeling_qwen3_next.py",
        "tensorrt_llm/_torch/modules/fused_moe/",
        "tensorrt_llm/_torch/pyexecutor/_util.py",
        "tensorrt_llm/_torch/pyexecutor/model_engine.py",
        "tensorrt_llm/_torch/pyexecutor/py_executor.py",
        "tensorrt_llm/_torch/auto_deploy/transform/library/sharding.py",
        "tensorrt_llm/_torch/visual_gen/attention_backend/parallel.py",
        "tensorrt_llm/_torch/visual_gen/modules/vae/",
        "tensorrt_llm/_torch/visual_gen/modules/attention.py",
        "tensorrt_llm/_torch/visual_gen/executor.py",
        "tensorrt_llm/_torch/visual_gen/mapping.py",
        "tensorrt_llm/_torch/visual_gen/pipeline.py",
        "tensorrt_llm/_torch/visual_gen/pipeline_loader.py",
        "tensorrt_llm/_torch/visual_gen/models/wan/parallel_vae.py",
        "tensorrt_llm/visual_gen/",
        "tensorrt_llm/bench/benchmark/visual_gen.py",
        "tensorrt_llm/evaluate/json_mode_eval.py",
        "tensorrt_llm/evaluate/mmlu.py",
        "tensorrt_llm/executor/",
        "tensorrt_llm/functional.py",
        "tensorrt_llm/llmapi/disagg_utils.py",
        "tensorrt_llm/llmapi/mgmn_leader_node.py",
        "tensorrt_llm/llmapi/mgmn_worker_node.py",
        "tensorrt_llm/llmapi/mpi_session.py",
        "tensorrt_llm/llmapi/trtllm-llmapi-launch",
        "tensorrt_llm/mapping.py",
        "tensorrt_llm/models/llama/",
        "tensorrt_llm/parameter.py",
        "tensorrt_llm/serve/",
        "tests/integration/defs/cpp/test_multi_gpu.py",
        "tests/integration/test_lists/test-db/l0_dgx_h100.yml",
        "tests/integration/test_lists/test-db/l0_dgx_h200.yml",
        "tests/unittest/auto_deploy/multigpu",
        "tests/unittest/_torch/multi_gpu/",
        "tests/unittest/_torch/multi_gpu_modeling/",
        "tests/unittest/_torch/visual_gen/multi_gpu/",
        "tests/unittest/disaggregated/",
        "tests/unittest/llmapi/test_llm_multi_gpu.py",
        "tests/unittest/llmapi/test_llm_multi_gpu_pytorch.py",
        "tests/integration/defs/accuracy/test_disaggregated_serving.py",
        "tests/unittest/_torch/ray_orchestrator/multi_gpu/",
        "tests/integration/defs/examples/test_ray.py",
        "tests/integration/defs/accuracy/test_llm_api_autodeploy.py",
        "tests/unittest/llmapi/test_async_llm.py",
        "docker/common/install_ucx.sh",
        "docker/common/install_nixl.sh",
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
        "Docs": [
            // Matched by prefix here, plus any "*.md" file anywhere in the repo (handled below).
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
        def matchesGroup = { file ->
            // Any *.md file, anywhere in the repo, counts as Docs-only.
            if (group == "Docs" && file.endsWith(".md")) {
                return true
            }
            return groupPrefixes.any { prefix -> file.startsWith(prefix) }
        }
        def allFilesInGroup = changedFileList.every(matchesGroup)

        if (allFilesInGroup) {
            pipeline.echo("Only ${group} files changed.")
            return group
        } else {
            def nonGroupFile = changedFileList.find { file -> !matchesGroup(file) }
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
        stage ("Collect Test Result") {
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
            trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, false, true)
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
        stage("Rerun Report") {
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
            catchError(
                buildResult: 'SUCCESS',
                stageResult: 'UNSTABLE') {
                error "Some failed tests were reruned, please check the rerun report."
            }
        } // Rerun report stage
        try {
            stage("Test Coverage") {
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

def launchJob(pipeline, jobName, reuseBuild, enableFailFast, globalVars, platform="x86_64", additionalParameters = [:]) {
    def parameters = getCommonParameters()
    // Build a local copy to avoid racey growth from shared parallel mutations.
    // In particular, CACHED_CHANGED_FILE_LIST can become very large and may
    // trigger "Argument list too long" when passed to downstream jobs.
    def globalVarsToPass = globalVars.findAll { key, value -> key != CACHED_CHANGED_FILE_LIST }
    String globalVarsJson = writeJSON returnText: true, json: globalVarsToPass
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

    if (jobName.startsWith("/")) {
        jobName = jobName.substring(1)
    } else {
        def pos = env.JOB_NAME.lastIndexOf("/")
        if (pos != -1) {
            jobDir = env.JOB_NAME.substring(0, pos + 1)
        } else {
            jobDir = ""
        }
        jobName = "${jobDir}${jobName}"
    }

    echo "Trigger ${jobName} job, params: ${parameters}"

    def logger = new Logger(pipeline)
    def (jenkinsURL, buildStatus) = JobBuilder.build(pipeline, logger, jobName, parameters, 1, false)
    if (buildStatus != "SUCCESS") {
        error "Downstream job did not succeed"
    }
    return buildStatus
}

def launchStages(pipeline, reuseBuild, testFilter, enableFailFast, globalVars)
{
    stages = [
        "Release-Check": {
            script {
                if (GEN_POST_MERGE_BUILDS_ONLY) {
                    echo "Skipping Release-Check (GenPostMergeBuilds mode: builds only)"
                    return
                }
                launchReleaseCheck(this, globalVars)
            }
        },
        "x86_64-Linux": {
            script {
                // CBTS deliberately does NOT short-circuit at the arch / Build
                // layer. Build always runs so a wheel exists for sanity checks
                // and post-merge consumers; case-level narrowing happens later
                // in L0_Test.groovy::launchTestJobs (Layer 2) and renderTestDB
                // (Layer 3).
                def testStageName = "[Build-x86_64] Remote Run"
                stage(testStageName) {
                    def additionalParameters = [
                        'dockerImage': globalVars["LLM_DOCKER_IMAGE"],
                        'wheelDockerImagePy310': globalVars["LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE"],
                        'wheelDockerImagePy312': globalVars["LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE"],
                    ]
                    launchJob(pipeline, "/LLM/helpers/Build-x86_64", reuseBuild, enableFailFast, globalVars, "x86_64", additionalParameters)
                }

                if (GEN_POST_MERGE_BUILDS_ONLY) {
                    echo "Skipping x86_64 tests (GenPostMergeBuilds mode: builds only)"
                    return
                }

                testStageName = "[Test-x86_64-Single-GPU] Remote Run"
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

                        launchJob(pipeline, "L0_Test-x86_64-Single-GPU", false, enableFailFast, globalVars, "x86_64", additionalParameters)
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
                    if (env.JOB_NAME ==~ /.*PostMerge.*/ || !enableFailFast) {
                        echo "In the official post-merge pipeline or when fail fast is disabled, x86_64 single-GPU test failed, whereas multi-GPU test is still kept running."
                    } else {
                        stage("[Test-x86_64-Multi-GPU] Blocked") {
                            error "This pipeline requires running multi-GPU test, but x86_64 single-GPU test has failed."
                        }
                        return
                    }
                }

                testStageName = "[Test-x86_64-Multi-GPU] Remote Run"
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

                        launchJob(pipeline, "L0_Test-x86_64-Multi-GPU", false, enableFailFast, globalVars, "x86_64", additionalParameters)

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
        "SBSA-Linux": {
            script {
                if (testFilter[(ONLY_ONE_GROUP_CHANGED)] == "Docs") {
                    echo "SBSA build job is skipped due to Jenkins configuration or conditional pipeline run"
                    return
                }
                // CBTS deliberately does NOT short-circuit the SBSA Build —
                // see x86 track above for the rationale.

                def testStageName = "[Build-SBSA] Remote Run"
                stage(testStageName) {
                    def additionalParameters = [
                        "dockerImage": globalVars["LLM_SBSA_DOCKER_IMAGE"],
                    ]
                    launchJob(pipeline, "/LLM/helpers/Build-SBSA", reuseBuild, enableFailFast, globalVars, "SBSA", additionalParameters)
                }

                if (GEN_POST_MERGE_BUILDS_ONLY) {
                    echo "Skipping SBSA tests (GenPostMergeBuilds mode: builds only)"
                    return
                }

                testStageName = "[Test-SBSA-Single-GPU] Remote Run"
                def singleGpuTestFailed = false
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

                        launchJob(pipeline, "L0_Test-SBSA-Single-GPU", false, enableFailFast, globalVars, "SBSA", additionalParameters)
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
                    if (env.JOB_NAME ==~ /.*PostMerge.*/ || !enableFailFast) {
                        echo "In the official post-merge pipeline or when fail fast is disabled, SBSA single-GPU test failed, whereas multi-GPU test is still kept running."
                    } else {
                        stage("[Test-SBSA-Multi-GPU] Blocked") {
                            error "This pipeline requires running SBSA multi-GPU test, but SBSA single-GPU test has failed."
                        }
                        return
                    }
                }

                testStageName = "[Test-SBSA-Multi-GPU] Remote Run"
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

                        launchJob(pipeline, "L0_Test-SBSA-Multi-GPU", false, enableFailFast, globalVars, "SBSA", additionalParameters)

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
                def testStageName = "[Build-Docker-Images] Remote Run"
                stage(testStageName) {
                    try {
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

                        launchJob(pipeline, "/LLM/helpers/BuildDockerImages", false, enableFailFast, globalVars, "x86_64", additionalParameters)
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception e) {
                        if (BUILD_CHECK_CHOICE == STAGE_CHOICE_IGNORE) {
                            catchError(
                                buildResult: 'SUCCESS',
                                stageResult: 'FAILURE') {
                                error "Build-Docker-Images job failed but ignored due to Jenkins configuration"
                            }
                        } else {
                            throw e
                        }
                    }
                }
            }
        }
    ]

    if (env.JOB_NAME ==~ /.*PostMerge.*/ && !GEN_POST_MERGE_BUILDS_ONLY) {
        stages += dockerBuildJob
    }
    if (!GEN_POST_MERGE_BUILDS_ONLY && (testFilter[(TEST_STAGE_LIST)]?.contains("Build-Docker-Images") || testFilter[(EXTRA_STAGE_LIST)]?.contains("Build-Docker-Images"))) {
        stages += dockerBuildJob
        testFilter[(TEST_STAGE_LIST)]?.remove("Build-Docker-Images")
        testFilter[(EXTRA_STAGE_LIST)]?.remove("Build-Docker-Images")
        echo "Will run Build-Docker-Images job"
        stages.remove("x86_64-Linux")
        stages.remove("SBSA-Linux")
        echo "Build-Docker-Images job is set explicitly. Both x86_64-Linux and SBSA-Linux sub-pipelines will be disabled."
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
            script {
                if (!GEN_POST_MERGE_BUILDS_ONLY) {
                    updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: "failed"
                }
            }
        }
        success {
            script {
                if (enableUpdateGitlabStatus) {
                    updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: "success"
                } else if (!GEN_POST_MERGE_BUILDS_ONLY) {
                    updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: "canceled"
                    updateGitlabCommitStatus name: "Custom Jenkins build", state: "success"
                }
            }
        }
        aborted {
            script {
                if (!GEN_POST_MERGE_BUILDS_ONLY) {
                    updateGitlabCommitStatus name: "${BUILD_STATUS_NAME}", state: 'canceled'
                }
            }
        }
        failure {
            script {
                try {
                    def prNumber = null
                    if (globalVars[GITHUB_PR_API_URL]) {
                        def prMatch = (globalVars[GITHUB_PR_API_URL] =~ /\/pulls?\/(\d+)/)
                        if (prMatch) {
                            prNumber = prMatch[0][1]
                        }
                    }
                    def analysis = trtllm_utils.analyzePipelineFailureWithAgent(
                        this, env.JOB_NAME, env.BUILD_NUMBER, prNumber)
                    if (analysis) {
                        def bucket = 'sw-tensorrt-ci-analysis'
                        def key = "${env.JOB_NAME}/${env.BUILD_NUMBER}/failure_analysis.html"
                        def htmlUrl = "https://pbss.s8k.io/v1/AUTH_svc_tensorrt/${bucket}/${key}"
                        // Self-rendering HTML page: marked.js parses the analysis at page load
                        // and DOMPurify sanitises the result before injection into the DOM. The
                        // analysis text comes from the CI agent which consumes build logs (which
                        // can include attacker-controlled PR content), so we treat it as untrusted.
                        // Hardening:
                        //   1. CDN scripts pinned to specific versions and protected with SRI.
                        //   2. Analysis embedded in a `<script type="application/json">` data
                        //      block read via textContent + JSON.parse — never inlined into
                        //      executable JS source. Every `<` in the JSON is rewritten to its
                        //      JSON unicode escape so a payload cannot smuggle a `</script>`
                        //      and break out of the data block.
                        //   3. marked output is run through DOMPurify before innerHTML assignment
                        //      to strip event-handler attributes and other XSS vectors.
                        def jsonAnalysis = groovy.json.JsonOutput.toJson(analysis).replace("<", "\\u003c")
                        def htmlDoc = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>CI Failure Analysis &middot; ${env.JOB_NAME} #${env.BUILD_NUMBER}</title>
<script src="https://cdn.jsdelivr.net/npm/marked@14.1.4/marked.min.js" integrity="sha384-lqPzN0kmFw9t2syAMwVPM4VbAyqsz/lPyYWbb2Xt6nSPM0WPNrpSWCUBgdcAdgnC" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.4/dist/purify.min.js" integrity="sha384-eEu5CTj3qGvu9PdJuS+YlkNi7d2XxQROAFYOr59zgObtlcux1ae1Il3u7jvdCSWu" crossorigin="anonymous"></script>
<style>body{font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:900px;margin:2em auto;padding:0 1em;color:#24292e}h1,h2,h3{border-bottom:1px solid #eaecef;padding-bottom:.3em}pre{background:#f6f8fa;padding:1em;overflow:auto;border-radius:6px}code{background:#f6f8fa;padding:.2em .4em;border-radius:3px}pre code{background:none;padding:0}a{color:#0366d6}blockquote{border-left:4px solid #dfe2e5;padding:0 1em;color:#6a737d}table{border-collapse:collapse}th,td{border:1px solid #dfe2e5;padding:6px 13px}header{margin-bottom:1.5em;color:#586069}</style>
</head><body>
<header><a href="${env.BUILD_URL}">${env.JOB_NAME} #${env.BUILD_NUMBER}</a></header>
<main id="md"></main>
<script id="md-source" type="application/json">${jsonAnalysis}</script>
<script>
  // Disable marked's strikethrough tokenizer: CI failure-analysis text routinely
  // contains literal tildes (~/path, ~50ms, regex anchors, etc.) that should not
  // be interpreted as markup. Other GFM extensions (tables, fences, autolinks,
  // task lists) stay enabled.
  marked.use({ tokenizer: { del() { return false; } } });
  const src = JSON.parse(document.getElementById('md-source').textContent);
  document.getElementById('md').innerHTML = DOMPurify.sanitize(marked.parse(src));
</script>
</body></html>
"""
                        writeFile file: 'failure_analysis.html', text: htmlDoc
                        container("alpine") {
                            trtllm_utils.llmExecStepWithRetry(this, script: 'apk add --no-cache aws-cli')
                            // Alpine's musl libc fires A and AAAA queries in parallel; pbss.s8k.io's AAAA
                            // returns SERVFAIL and musl treats that as a fatal lookup failure (glibc would
                            // not). Pin the A-record IP in /etc/hosts so getaddrinfo resolves from files.
                            trtllm_utils.llmExecStepWithRetry(this, script: '''
                                if ! grep -q 'pbss.s8k.io' /etc/hosts; then
                                    ip=$(nslookup -type=A pbss.s8k.io 2>/dev/null | awk '/^Address[: ]/ && $NF !~ /:53$/ && $NF !~ /#53$/ { print $NF; exit }')
                                    if [ -n "$ip" ]; then
                                        printf '%s\\n' "$ip pbss.s8k.io" >> /etc/hosts
                                    fi
                                fi
                            ''')
                            withCredentials([string(
                                    credentialsId: 'svc_tensorrt-swift-stack-key',
                                    variable: 'AWS_SECRET_ACCESS_KEY')]) {
                                trtllm_utils.llmExecStepWithRetry(this, script:
                                    "AWS_ACCESS_KEY_ID=svc_tensorrt aws s3 cp failure_analysis.html" +
                                    " 's3://${bucket}/${key}' --endpoint-url https://pbss.s8k.io" +
                                    " --content-type text/html")
                            }
                        }
                        // Surface the URL via currentBuild.description so the upstream PR_Github
                        // wrapper can extract it and include it in the GitHub PR comment.
                        def existingDesc = currentBuild.description ?: ""
                        currentBuild.description = existingDesc +
                            (existingDesc ? "<br/>" : "") +
                            "<a href='${htmlUrl}'>CI Agent Failure Analysis</a>"
                        echo "CI Agent Failure Analysis: ${htmlUrl}"
                    }
                } catch (Exception e) {
                    // Analysis is best-effort; do not fail the pipeline
                }
            }
        }
        always {
            script {
                if (!isReleaseCheckMode && !GEN_POST_MERGE_BUILDS_ONLY) {
                    collectTestResults(this, testFilter)
                }
                stage("Upload Build Info") {
                    try {
                        def branch = env.gitlabBranch ? env.gitlabBranch : "main"
                        if (globalVars[GITHUB_PR_API_URL]) {
                            branch = "github-pr-" + globalVars[GITHUB_PR_API_URL].split('/').last()
                        }
                        def buildInfo = "commit=${env.gitlabCommit}\n" +
                            "branch=${branch}\n" +
                            "date=${new Date().format('yyyy-MM-dd HH:mm:ss z', TimeZone.getTimeZone('UTC'))}\n" +
                            "jenkins_url=${env.BUILD_URL}"
                        writeFile file: 'build_info.txt', text: buildInfo
                        trtllm_utils.uploadArtifacts("build_info.txt", "${UPLOAD_PATH}/")
                        echo "Build info: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/build_info.txt"
                    } catch (Exception e) {
                        echo "Upload Build Info failed: ${e.toString()}"
                    }
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
        stage("Build And Test") {
            steps {
                script {
                    if (isReleaseCheckMode) {
                        stage("Release-Check") {
                            script {
                                launchReleaseCheck(this, globalVars)
                            }
                        }
                    } else {
                        // globalVars[CACHED_CHANGED_FILE_LIST] is only used in setupPipelineEnvironment
                        // Remove it to workaround the "Argument list too long" error
                        globalVars.remove(CACHED_CHANGED_FILE_LIST)
                        launchStages(this, reuseBuild, testFilter, enableFailFast, globalVars)
                    }
                }
            }
        }
    } // stages
} // pipeline
