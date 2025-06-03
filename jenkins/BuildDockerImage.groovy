@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.Exception
import groovy.transform.Field

// Docker image registry
IMAGE_NAME = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging"

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}
LLM_ROOT = "llm"

LLM_BRANCH = env.gitlabBranch ?: params.branch
LLM_BRANCH_TAG = LLM_BRANCH.replaceAll('/', '_')

LLM_COMMIT_OR_BRANCH = env.gitlabCommit ?: LLM_BRANCH

LLM_SHORT_COMMIT = env.gitlabCommit ? env.gitlabCommit.substring(0, 7) : "undefined"

LLM_DEFAULT_TAG = env.defaultTag ?: "${LLM_SHORT_COMMIT}-${LLM_BRANCH_TAG}-${BUILD_NUMBER}"

RUN_SANITY_CHECK = env.runSanityCheck ?: false

BUILD_JOBS = "32"
BUILD_JOBS_RELEASE_X86_64 = "32"
BUILD_JOBS_RELEASE_SBSA = "32"

// UPLOAD_PATH = env.uploadPath ?: "sw-tensorrt-artifacts"
UPLOAD_PATH = "sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/2052"

CCACHE_DIR="/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"

@Field
def GITHUB_PR_API_URL = "github_pr_api_url"
@Field
def CACHED_CHANGED_FILE_LIST = "cached_changed_file_list"
@Field
def ACTION_INFO = "action_info"
@Field
def IMAGE_KEY_TO_TAG = "image_key_to_tag"
def globalVars = [
    (GITHUB_PR_API_URL): null,
    (CACHED_CHANGED_FILE_LIST): null,
    (ACTION_INFO): null,
    (IMAGE_KEY_TO_TAG): [:],
]

@Field
def imageKeyToTag = [:]

def createKubernetesPodConfig(type, arch = "amd64", build_wheel = false)
{
    def targetCould = "kubernetes-cpu"
    def containerConfig = ""
    def selectors = """
                nodeSelector:
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux
                  kubernetes.io/arch: ${arch}"""

    if (build_wheel && arch == "arm64") {
        // For aarch64, we need to use hostname to fix the ucxx issue when building wheels
        selectors += """
                affinity:
                    nodeAffinity:
                        requiredDuringSchedulingIgnoredDuringExecution:
                            nodeSelectorTerms:
                                - matchExpressions:
                                    - key: "kubernetes.io/hostname"
                                      operator: In
                                      values:
                                        - "rl300-0008.ipp2u1.colossus"
                                        - "rl300-0014.ipp2u1.colossus"
                                        - "rl300-0023.ipp2u1.colossus"
                                        - "rl300-0024.ipp2u1.colossus"
                                        - "rl300-0030.ipp2u1.colossus"
                                        - "rl300-0040.ipp2u1.colossus"
                                        - "rl300-0041.ipp2u1.colossus"
                                        - "rl300-0042.ipp2u1.colossus"
                                        - "rl300-0043.ipp2u1.colossus"
                                        - "rl300-0044.ipp2u1.colossus"
                                        - "rl300-0045.ipp2u1.colossus"
                                        - "rl300-0046.ipp2u1.colossus"
                                        - "rl300-0047.ipp2u1.colossus"
        """
    }

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
        break
    case "build":
        // Use a customized docker:dind image with essential dependencies
        containerConfig = """
                  - name: docker
                    image: urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:202505221445_docker_dind_withbash
                    tty: true
                    resources:
                      requests:
                        cpu: 16
                        memory: 72Gi
                        ephemeral-storage: 200Gi
                      limits:
                        cpu: 16
                        memory: 256Gi
                        ephemeral-storage: 200Gi
                    imagePullPolicy: Always
                    securityContext:
                      privileged: true
                      capabilities:
                        add:
                        - SYS_ADMIN"""
        break
    }
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
    def nodeLabelPrefix = "cpu"
    def jobName = "llm-build-images"
    def buildID = env.BUILD_ID
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
                ${selectors}
                containers:
                  ${containerConfig}
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
                volumeMounts:
                    - name: sw-tensorrt-pvc
                      mountPath: "/mnt/sw-tensorrt-pvc"
                      readOnly: false
                volumes:
                ${pvcVolume}
        """.stripIndent(),
    ]

    return podConfig
}


def buildImage(config, imageKeyToTag)
{
    def target = config.target
    def action = config.action
    def torchInstallType = config.torchInstallType
    def args = config.args ?: ""
    def customTag = config.customTag
    def postTag = config.postTag
    def dependentTarget = config.dependentTarget
    def arch = config.arch == 'arm64' ? 'sbsa' : 'x86_64'

    def tmpTag = "8166649-main-108" // TODO: remove this
    // def tmpTag = LLM_DEFAULT_TAG

    def tag = "${arch}-${target}-torch_${torchInstallType}${postTag}-${tmpTag}"

    def dependentTargetTag = tag.replace("${arch}-${target}-", "${arch}-${dependentTarget}-")

    if (target == "ngc-release") {
        imageKeyToTag["NGC Devel Image ${config.arch}"] = "${IMAGE_NAME}/devel:${dependentTargetTag}"
        imageKeyToTag["NGC Release Image ${config.arch}"] = "${IMAGE_NAME}/release:${tag}"
    }
    return // TODO: remove this

    args += " GITHUB_MIRROR=https://urm.nvidia.com/artifactory/github-go-remote"

    stage (config.stageName) {
        // Step 1: Clone TRT-LLM source codes
        // If using a forked repo, svc_tensorrt needs to have the access to the forked repo.
        trtllm_utils.checkoutSource(LLM_REPO, LLM_COMMIT_OR_BRANCH, LLM_ROOT, true, true)
    }

    // Step 2: Build the images
    stage ("Install packages") {
        sh "pwd && ls -alh"
        sh "env"
        sh "apk add make git"
        sh "git config --global --add safe.directory '*'"

        withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
            sh "docker login urm.nvidia.com -u ${USERNAME} -p ${PASSWORD}"
        }

        withCredentials([
            usernamePassword(
                credentialsId: "svc_tensorrt_gitlab_read_api_token",
                usernameVariable: 'USERNAME',
                passwordVariable: 'PASSWORD'
            ),
            string(credentialsId: 'default-git-url', variable: 'DEFAULT_GIT_URL')
        ]) {
            sh "docker login ${DEFAULT_GIT_URL}:5005 -u ${USERNAME} -p ${PASSWORD}"
        }
    }
    try {
        def build_jobs = BUILD_JOBS
        // Fix the triton image pull timeout issue
        def TRITON_IMAGE = sh(script: "cd ${LLM_ROOT} && grep 'ARG TRITON_IMAGE=' docker/Dockerfile.multi | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()
        def TRITON_BASE_TAG = sh(script: "cd ${LLM_ROOT} && grep 'ARG TRITON_BASE_TAG=' docker/Dockerfile.multi | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()
        containerGenFailure = null

        if (dependentTarget) {
            stage ("make ${dependentTarget}_${action} (${arch})") {
                retry(3) {
                    retry(3) {
                        sh "docker pull ${TRITON_IMAGE}:${TRITON_BASE_TAG}"
                    }
                    sh """
                    cd ${LLM_ROOT} && make -C docker ${dependentTarget}_${action} \
                    TORCH_INSTALL_TYPE=${torchInstallType} \
                    IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${dependentTargetTag} \
                    BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args}
                    """
                }
                args += " DEVEL_IMAGE=${IMAGE_NAME}/${dependentTarget}:${dependentTargetTag}"
            }
        }

        // Avoid the frequency of OOM issue when building the wheel
        if (target == "trtllm") {
            if (arch == "x86_64") {
                build_jobs = BUILD_JOBS_RELEASE_X86_64
            } else {
                build_jobs = BUILD_JOBS_RELEASE_SBSA
            }
        }
        stage ("make ${target}_${action} (${arch})") {
            retry(3) {
                retry(3) {
                    sh "docker pull ${TRITON_IMAGE}:${TRITON_BASE_TAG}"
                }

                sh """
                cd ${LLM_ROOT} && make -C docker ${target}_${action} \
                TORCH_INSTALL_TYPE=${torchInstallType} \
                IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${tag} \
                BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args}
                """
            }
        }

        if (customTag) {
            stage ("custom tag: ${customTag} (${arch})") {
                sh """
                cd ${LLM_ROOT} && make -C docker ${target}_${action} \
                TORCH_INSTALL_TYPE=${torchInstallType} \
                IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${customTag} \
                BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args}
                """
            }
        }
    } catch (Exception ex) {
        containerGenFailure = ex
    } finally {
        stage ("Docker logout") {
            withCredentials([string(credentialsId: 'default-git-url', variable: 'DEFAULT_GIT_URL')]) {
                sh "docker logout urm.nvidia.com"
                sh "docker logout ${DEFAULT_GIT_URL}:5005"
            }
        }
        if (containerGenFailure != null) {
            throw containerGenFailure
        }
    }
}


def launchBuildJobs(pipeline, globalVars, imageKeyToTag) {
    def defaultBuildConfig = [
        target: "tritondevel",
        action: params.action,
        customTag: "",
        postTag: "",
        args: "",
        torchInstallType: "skip",
        arch: "amd64",
        build_wheel: false,
        dependentTarget: "",
    ]

    def release_action = env.JOB_NAME ==~ /.*PostMerge.*/ ? "push" : params.action
    def buildConfigs = [
        "Build trtllm release (x86_64)": [
            target: "trtllm",
            action: release_action,
            customTag: LLM_BRANCH_TAG + "-x86_64",
            build_wheel: true,
        ],
        "Build trtllm release (SBSA)": [
            target: "trtllm",
            action: release_action,
            customTag: LLM_BRANCH_TAG + "-sbsa",
            build_wheel: true,
            arch: "arm64"
        ],
        "Build CI image (x86_64 tritondevel)": [:],
        "Build CI image (SBSA tritondevel)": [
            arch: "arm64",
        ],
        "Build CI image (RockyLinux8 Python310)": [
            target: "rockylinux8",
            args: "PYTHON_VERSION=3.10.12",
            postTag: "-py310",
        ],
        "Build CI image(RockyLinux8 Python312)": [
            target: "rockylinux8",
            args: "PYTHON_VERSION=3.12.3 STAGE=tritondevel",
            postTag: "-py312",
        ],
        "Build NGC devel and release (x86_64)": [
            target: "ngc-release",
            action: release_action,
            customTag: "ngc-" + LLM_BRANCH_TAG + "-x86_64",
            args: "DOCKER_BUILD_OPTS='--load --platform linux/amd64'",
            build_wheel: true,
            dependentTarget: "devel",
        ],
        "Build NGC devel and release(SBSA)": [
            target: "ngc-release",
            action: release_action,
            customTag: "ngc-" + LLM_BRANCH_TAG + "-sbsa",
            args: "DOCKER_BUILD_OPTS='--load --platform linux/arm64'",
            arch: "arm64",
            build_wheel: true,
            dependentTarget: "devel",
        ],
    ]
    // Override all fields in build config with default values
    buildConfigs.each { key, config ->
        defaultBuildConfig.each { defaultKey, defaultValue ->
            if (!(defaultKey in config)) {
                config[defaultKey] = defaultValue
            }
        }
        config.podConfig = createKubernetesPodConfig("build", config.arch, config.build_wheel)
    }
    echo "Build configs:"
    println buildConfigs

    def buildJobs = buildConfigs.collectEntries { key, config ->
        [key, {
            script {
                stage(key) {
                    config.stageName = key
                    trtllm_utils.launchKubernetesPod(pipeline, config.podConfig, "docker") {
                        buildImage(config, imageKeyToTag)
                    }
                }
            }
        }]
    }

    echo "enableFailFast is: ${env.enableFailFast}, but we currently don't use it due to random ucxx issue"
    //pipeline.failFast = env.enableFailFast
    pipeline.parallel buildJobs

}


def getCommonParameters()
{
    return [
        'gitlabSourceRepoHttpUrl': LLM_REPO,
        'gitlabCommit': '8166649d033109319d7d08cf9541d8996848018f',
        'artifactPath': env.artifactPath,
        'uploadPath': UPLOAD_PATH,
    ]
}


pipeline {
    agent {
        kubernetes createKubernetesPodConfig("agent")
    }

    parameters {
        string(
            name: "branch",
            defaultValue: "main",
            description: "Branch to launch job."
        )
        choice(
            name: "action",
            choices: ["build", "push"],
            description: "Docker image generation action. build: only perform image build step; push: build docker image and push it to artifacts"
        )
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
        CCACHE_DIR="${CCACHE_DIR}"
        PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
    }
    stages {
        stage("Setup environment") {
            steps {
                script {
                    echo "branch is: ${LLM_BRANCH}"
                    echo "env.gitlabBranch is: ${env.gitlabBranch}"
                    echo "params.branch is: ${params.branch}"
                    echo "params.action is: ${params.action}"
                    echo "env.defaultTag is: ${env.defaultTag}"
                    echo "env.gitlabCommit is: ${env.gitlabCommit}"
                    echo "LLM_REPO is: ${LLM_REPO}"
                    echo "env.globalVars is: ${env.globalVars}"
                    globalVars = trtllm_utils.updateMapWithJson(this, globalVars, env.globalVars, "globalVars")
                    globalVars[ACTION_INFO] = trtllm_utils.setupPipelineDescription(this, globalVars[ACTION_INFO])
                }
            }
        }
        stage("Build") {
            steps{
                script{
                    launchBuildJobs(this, globalVars, imageKeyToTag)
                }
            }
        }
        stage("Upload Artifacts") {
            steps {
                script {
                    String imageKeyToTagJson = writeJSON returnText: true, json: imageKeyToTag
                    echo "imageKeyToTag is: ${imageKeyToTagJson}"
                    writeFile file: "imageKeyToTag.json", text: imageKeyToTagJson
                    archiveArtifacts artifacts: 'imageKeyToTag.json', fingerprint: true
                }
            }
        }
        stage("Wait for build jobs finish") {
            when {
                expression {
                    RUN_SANITY_CHECK
                }
            }
            steps {
                script {
                    collectResultPodSpec = createKubernetesPodConfig("agent")
                    trtllm_utils.launchKubernetesPod(this, collectResultPodSpec, "alpine", {
                        // 安装wget工具
                        trtllm_utils.llmExecStepWithRetry(this, script: "apk add --no-cache wget")
                        // 轮询检查构建产物文件是否存在
                        def artifactBaseUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/"
                        def requiredFiles = [
                            "TensorRT-LLM-GH200.tar.gz",
                            "tensorrt-llm-release-src-",
                            "tensorrt-llm-sbsa-release-src-",
                            "TensorRT-LLM.tar.gz"
                        ]
                        def maxWaitMinutes = 180
                        def pollIntervalSeconds = 60

                        echo "开始等待构建产物文件..."
                        echo "需要等待的文件: ${requiredFiles}"
                        echo "检查路径: ${artifactBaseUrl}"

                        def startTime = System.currentTimeMillis()
                        def maxWaitMs = maxWaitMinutes * 60 * 1000

                        while ((System.currentTimeMillis() - startTime) < maxWaitMs) {
                            def missingFiles = []

                            try {
                                // 只下载一次目录索引
                                trtllm_utils.llmExecStepWithRetry(this, script: "wget ${artifactBaseUrl} -O index.html", allowStepFailed: true)
                                def indexContent = sh(script: "cat index.html 2>/dev/null || echo ''", returnStdout: true).trim()

                                // 检查所有需要的文件
                                for (file in requiredFiles) {
                                    if (!indexContent.contains(file)) {
                                        missingFiles.add(file)
                                    }
                                }

                                // 删除index文件
                                sh(script: "rm -f index.html", returnStdout: false)

                            } catch (Exception e) {
                                echo "检查构建产物时出错: ${e.message}"
                                // 如果出错，假设所有文件都缺失
                                missingFiles = requiredFiles.clone()
                                // 确保删除可能存在的index文件
                                sh(script: "rm -f index.html", returnStdout: false)
                            }

                            if (missingFiles.isEmpty()) {
                                echo "所有构建产物文件都已就绪!"
                                return
                            }

                            def elapsedMinutes = (System.currentTimeMillis() - startTime) / (60 * 1000)
                            echo "等待中... (已等待 ${elapsedMinutes.intValue()} 分钟)"
                            echo "缺失的文件: ${missingFiles}"
                            sleep(pollIntervalSeconds)
                        }

                        def elapsedMinutes = (System.currentTimeMillis() - startTime) / (60 * 1000)
                        error "等待构建产物超时 (${elapsedMinutes.intValue()} 分钟)，部分文件仍未就绪"
                    })
                }
            }
        }
        stage("Sanity Check") {
            when {
                expression {
                    RUN_SANITY_CHECK
                }
            }
            steps {
                script {
                    globalVars[IMAGE_KEY_TO_TAG] = imageKeyToTag
                    String globalVarsJson = writeJSON returnText: true, json: globalVars
                    def parameters = getCommonParameters()
                    parameters += [
                        'enableFailFast': false,
                        'branch': LLM_BRANCH,
                        'globalVars': globalVarsJson,
                        'targetArch': "aarch64-linux-gnu",
                    ]

                    echo "trigger BuildDockerImageSanityTest job, params: ${parameters}"

                    def status = ""
                    def jobName = "/LLM/helpers/BuildDockerImageSanityTest"
                    def jobPath = trtllm_utils.resolveFullJobName(jobName).replace('/', '/job/').substring(1)
                    def jenkinsUrl = env.JENKINS_URL
                    def credentials = env.localJobCredentials
                    def handle = triggerRemoteJob(
                        job: "${jenkinsUrl}${jobPath}/",
                        auth: CredentialsAuth(credentials: credentials),
                        parameters: trtllm_utils.toRemoteBuildParameters(parameters),
                        pollInterval: 60,
                        abortTriggeredJob: true,
                    )
                    status = handle.getBuildResult().toString()

                    if (status != "SUCCESS") {
                        error "Downstream job did not succeed"
                    }
                }
            }
        }
    } // stages
} // pipeline
