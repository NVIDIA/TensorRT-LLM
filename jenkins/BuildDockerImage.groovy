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

LLM_BRANCH = env.gitlabBranch? env.gitlabBranch : params.branch
LLM_BRANCH_TAG = LLM_BRANCH.replaceAll('/', '_')

LLM_COMMIT_OR_BRANCH = env.gitlabCommit ?: (params.commit ? params.commit : LLM_BRANCH)

BUILD_JOBS = "32"
BUILD_JOBS_RELEASE_X86_64 = "16"
BUILD_JOBS_RELEASE_SBSA = "8"

CCACHE_DIR="/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"

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

def createKubernetesPodConfig(type, arch = "amd64")
{
    def targetCould = "kubernetes-cpu"
    def containerConfig = ""

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
        containerConfig = """
                  - name: docker
                    image: urm.nvidia.com/docker/docker:dind
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
    def podConfig = [
        cloud: targetCould,
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                qosClass: Guaranteed
                nodeSelector:
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux
                  kubernetes.io/arch: ${arch}
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


def buildImage(config)
{
    def target = config.target
    def action = config.action
    def torchInstallType = config.torchInstallType
    def args = config.args
    def customTag = config.customTag
    def postTag = config.postTag
    def arch = if (config.arch == 'arm64') 'aarch64' else 'x86_64'

    def tag = "${arch}-${target}-torch_${torchInstallType}${postTag}-${LLM_BRANCH_TAG}-${BUILD_NUMBER}"

    // Step 1: cloning tekit source code
    // allow to checkout from forked repo, svc_tensorrt needs to have access to the repo, otherwise clone will fail
    trtllm_utils.checkoutSource(LLM_REPO, LLM_COMMIT_OR_BRANCH, LLM_ROOT, true, true)

    // Step 2: building wheels in container
    container("docker") {
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
            // Fix the build OOM issue of release builds
            def build_jobs = BUILD_JOBS
            if (target == "trtllm") {
                build_jobs = BUILD_JOBS_RELEASE_X86_64
            } else if (target == "trtllm_sbsa") {
                build_jobs = BUILD_JOBS_RELEASE_SBSA
            }
            containerGenFailure = null
            stage ("make ${target}_${action}") {
                retry(3)
                {
                  // Fix the triton image pull timeout issue
                  def TRITON_IMAGE = sh(script: "cd ${LLM_ROOT} && grep 'ARG TRITON_IMAGE=' docker/Dockerfile.multi | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()
                  def TRITON_BASE_TAG = sh(script: "cd ${LLM_ROOT} && grep 'ARG TRITON_BASE_TAG=' docker/Dockerfile.multi | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()
                  retry(3) {
                    sh "docker pull ${TRITON_IMAGE}:${TRITON_BASE_TAG}"
                  }

                  sh """
                  cd ${LLM_ROOT} && make -C docker ${target}_${action} \
                  TORCH_INSTALL_TYPE=${torchInstallType} \
                  IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${tag} \
                  BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args} \
                  SHARED_CCACHE_DIR=${CCACHE_DIR} \
                  GITHUB_MIRROR=https://urm.nvidia.com/artifactory/github-go-remote
                  """
                }
            }

            if (customTag) {
                stage ("custom tag: ${customTag}") {
                  sh """
                  cd ${LLM_ROOT} && make -C docker ${target}_${action} \
                  TORCH_INSTALL_TYPE=${torchInstallType} \
                  IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${customTag} \
                  BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args} \
                  SHARED_CCACHE_DIR=${CCACHE_DIR} \
                  GITHUB_MIRROR=https://urm.nvidia.com/artifactory/github-go-remote
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
}


def launchBuildJobs(pipeline) {
    def defaultBuildConfig = [
        target: "tritondevel",
        action: params.action,
        customTag: "",
        postTag: "",
        args: "",
        torchInstallType: "skip",
        arch: "amd64"
    ]
    def buildConfigs = [
        "Build trtllm release(x86_64)": [
            target: "trtllm_x86_64",
            action: "push",
            customTag: LLM_BRANCH_TAG + "-amd64",
        ],
        "Build trtllm release(aarch64)": [
            target: "trtllm_aarch64",
            action: "push",
            customTag: LLM_BRANCH_TAG + "-arm64",
            arch: "arm64"
        ],
        "Build CI image(x86_64)": [],
        "Build CI image(aarch64)": [
            arch: "arm64",
        ],
        "Build CI image(rockylinux8-py310)": [
            target: "rockylinux8",
            args: "PYTHON_VERSION=3.10.12 STAGE=tritondevel",
            postTag: "-py310",
        ],
        "Build CI image(rockylinux8-py312)": [
            target: "rockylinux8",
            args: "PYTHON_VERSION=3.12.3 STAGE=tritondevel",
            postTag: "-py312",
        ],
        "Build NGC devel(x86_64)": [
            target: "devel",
        ],
        "Build NGC devel(aarch64)": [
            target: "devel",
            arch: "arm64",
        ],
        "Build NGC release(x86_64)": [
            target: "ngc_release",
            action: "push",
            customTag: "ngc-" + LLM_BRANCH_TAG + "-amd64",
        ],
        "Build NGC release(aarch64)": [
            target: "ngc_release",
            action: "push",
            customTag: "ngc-" + LLM_BRANCH_TAG + "-arm64",
            arch: "arm64",
        ],
    ]
    // Override all fields in build config with default values
    buildConfigs.each { key, config ->
        defaultConfig.each { defaultKey, defaultValue ->
            if (!config.containsKey(defaultKey)) {
                config[defaultKey] = defaultValue
            }
        }
    }
    echo "Build configs:"
    println buildConfigs

    def buildJobs = buildConfigs.collectEntries { key, config ->
        [key, {
            stage(key) {
                agent {
                    kubernetes createKubernetesPodConfig("build", config.arch)
                }
                steps
                {
                    buildImage(config)
                }
            }
        }]
    }
    echo "Build jobs:"
    println buildJobs

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
                    echo "env.gitlabCommit is: ${env.gitlabCommit}"
                    echo "LLM_REPO is: ${LLM_REPO}"
                    echo "env.globalVars is: ${env.globalVars}"
                    globalVars = trtllm_utils.updateMapWithJson(this, globalVars, env.globalVars, "globalVars")
                    globalVars[ACTION_INFO] = trtllm_utils.setupPipelineDescription(this, globalVars[ACTION_INFO])
                }
            }
        }
        stage("Build")
            steps{
                script{
                    launchBuildJobs(this)
                }
            }
        {
            parallel {
                stage("Build trtllm release") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("trtllm_x86_64", "push", "skip", "", LLM_BRANCH_TAG + "-x86_64")
                    }
                }
                stage("Build trtllm release-sbsa") {
                    agent {
                        kubernetes createKubernetesPodConfig("build", "arm64")
                    }
                    steps
                    {
                        buildImage("trtllm_sbsa", "push", "skip", "", LLM_BRANCH_TAG + "-sbsa", "", true)
                    }
                }
                stage("Build x86_64-skip") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("tritondevel", params.action, "skip")
                    }
                }
                stage("Build SBSA-skip") {
                    agent {
                        kubernetes createKubernetesPodConfig("agent")
                    }
                    steps
                    {
                        buildImage("tritondevel", params.action, "skip", "", "", "", true)
                    }
                }
                stage("Build rockylinux8 x86_64-skip-py3.10") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("rockylinux8", params.action, "skip", "PYTHON_VERSION=3.10.12 STAGE=tritondevel", "", "-py310")
                    }
                }
                stage("Build rockylinux8 x86_64-skip-py3.12") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("rockylinux8", params.action, "skip", "PYTHON_VERSION=3.12.3 STAGE=tritondevel", "", "-py312")
                    }
                }
                stage("Build NGC x86_64") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("devel", params.action, "skip")
                    }
                }
                stage("Build NGC SBSA") {
                    agent {
                        kubernetes createKubernetesPodConfig("build", "arm64")
                    }
                    steps
                    {
                        buildImage("devel", params.action, "skip", "", "", "", true)
                    }
                }
                stage("Build NGC release x86_64") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("devel", params.action, "skip")
                    }
                }
            }
        }
    } // stages
} // pipeline
