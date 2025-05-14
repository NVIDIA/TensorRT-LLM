@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@emma_move_funcs']) _

import java.lang.Exception
import groovy.transform.Field

// Docker image registry
IMAGE_NAME = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging"
DOCKER_DIND_IMAGE = "urm.nvidia.com/docker/docker:dind"

BUILD_CORES_REQUESTED = "16"
BUILD_CORES_LIMIT = "16"
BUILD_MEMORY_REQUESTED = "72Gi"
BUILD_MEMORY_LIMIT = "256Gi"

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}
LLM_ROOT = "llm"

LLM_BRANCH = env.gitlabBranch? env.gitlabBranch : params.branch
LLM_BRANCH_TAG = LLM_BRANCH.replaceAll('/', '_')

BUILD_JOBS = "32"
BUILD_JOBS_RELEASE_X86_64 = "16"
BUILD_JOBS_RELEASE_SBSA = "8"


def buildImage(target, action="build", torchInstallType="skip", args="", custom_tag="", post_tag="", is_sbsa=false)
{
    def arch = is_sbsa ? "sbsa" : "x86_64"
    def tag = "${arch}-${target}-torch_${torchInstallType}${post_tag}-${LLM_BRANCH_TAG}-${BUILD_NUMBER}"

    // Step 1: cloning tekit source code
    // allow to checkout from forked repo, svc_tensorrt needs to have access to the repo, otherwise clone will fail
    trtllm_utils.checkoutSource(LLM_REPO, LLM_BRANCH, LLM_ROOT, true, true)

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
                if (arch == "x86_64") {
                    build_jobs = BUILD_JOBS_RELEASE_X86_64
                } else {
                    build_jobs = BUILD_JOBS_RELEASE_SBSA
                }
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
                  GITHUB_MIRROR=https://urm.nvidia.com/artifactory/github-go-remote
                  """
                }
            }

            if (custom_tag) {
                stage ("custom tag: ${custom_tag}") {
                  sh """
                  cd ${LLM_ROOT} && make -C docker ${target}_${action} \
                  TORCH_INSTALL_TYPE=${torchInstallType} \
                  IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${custom_tag} \
                  BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args} \
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


pipeline {
    agent {
        kubernetes trtllm_utils.createKubernetesPodConfig(type: "agent")
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
        PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
    }
    stages {
        stage("Build")
        {
            parallel {
                stage("Build trtllm release") {
                    agent {
                        kubernetes trtllm_utils.createKubernetesPodConfig(
                            image: DOCKER_DIND_IMAGE,
                            type: "build",
                            cpuRequest: BUILD_CORES_REQUESTED,
                            cpuLimit: BUILD_CORES_LIMIT,
                            memoryRequest: BUILD_MEMORY_REQUESTED,
                            memoryLimit: BUILD_MEMORY_LIMIT,
                        )
                    }
                    steps
                    {
                        buildImage("trtllm", env.JOB_NAME ==~ /.*PostMerge.*/ ? "push" : params.action, "skip", "", LLM_BRANCH_TAG)
                    }
                }
                stage("Build x86_64-skip") {
                    agent {
                        kubernetes trtllm_utils.createKubernetesPodConfig(
                            image: DOCKER_DIND_IMAGE,
                            type: "build",
                            cpuRequest: BUILD_CORES_REQUESTED,
                            cpuLimit: BUILD_CORES_LIMIT,
                            memoryRequest: BUILD_MEMORY_REQUESTED,
                            memoryLimit: BUILD_MEMORY_LIMIT,
                        )
                    }
                    steps
                    {
                        buildImage("tritondevel", params.action, "skip")
                    }
                }
                stage("Build trtllm release-sbsa") {
                    {
                        kubernetes trtllm_utils.createKubernetesPodConfig(
<<<<<<< HEAD
                            image = DOCKER_DIND_IMAGE,
                            type = "build",
                            arch = "arm64"
                            cpuRequest = BUILD_CORES_REQUESTED,
                            cpuLimit = BUILD_CORES_LIMIT,
                            memoryRequest = BUILD_MEMORY_REQUESTED,
                            memoryLimit = BUILD_MEMORY_LIMIT
=======
                            image: DOCKER_DIND_IMAGE,
                            type: "build",
                            cpuRequest: BUILD_CORES_REQUESTED,
                            cpuLimit: BUILD_CORES_LIMIT,
                            memoryRequest: BUILD_MEMORY_REQUESTED,
                            memoryLimit: BUILD_MEMORY_LIMIT,
>>>>>>> 4e95dba5 (Try to fix errors when build docker image)
                        )
                    }
                    steps
                    {
<<<<<<< HEAD
                        buildImage("trtllm", env.JOB_NAME ==~ /.*PostMerge.*/ ? "push" : params.action, "skip", "", LLM_BRANCH_TAG + "-sbsa", "", true)
=======
                        buildImage("devel", params.action, "src_non_cxx11_abi")
                    }
                }
                stage("Build x86_64-cxx11_abi") {
                    agent {
                        kubernetes trtllm_utils.createKubernetesPodConfig(
                            image: DOCKER_DIND_IMAGE,
                            type: "build",
                            cpuRequest: BUILD_CORES_REQUESTED,
                            cpuLimit: BUILD_CORES_LIMIT,
                            memoryRequest: BUILD_MEMORY_REQUESTED,
                            memoryLimit: BUILD_MEMORY_LIMIT,
                        )
                    }
                    steps
                    {
                        buildImage("devel", params.action, "src_cxx11_abi")
>>>>>>> 4e95dba5 (Try to fix errors when build docker image)
                    }
                }
                stage("Build rockylinux8 x86_64-skip-py3.10") {
                    agent {
                        kubernetes trtllm_utils.createKubernetesPodConfig(
                            image: DOCKER_DIND_IMAGE,
                            type: "build",
                            cpuRequest: BUILD_CORES_REQUESTED,
                            cpuLimit: BUILD_CORES_LIMIT,
                            memoryRequest: BUILD_MEMORY_REQUESTED,
                            memoryLimit: BUILD_MEMORY_LIMIT,
                        )
                    }
                    steps
                    {
                        buildImage("rockylinux8", params.action, "skip", "PYTHON_VERSION=3.10.12 STAGE=tritondevel", "", "-py310")
                    }
                }
                stage("Build rockylinux8 x86_64-skip-py3.12") {
                    agent {
                        kubernetes trtllm_utils.createKubernetesPodConfig(
                            image: DOCKER_DIND_IMAGE,
                            type: "build",
                            cpuRequest: BUILD_CORES_REQUESTED,
                            cpuLimit: BUILD_CORES_LIMIT,
                            memoryRequest: BUILD_MEMORY_REQUESTED,
                            memoryLimit: BUILD_MEMORY_LIMIT,
                        )
                    }
                    steps
                    {
                        buildImage("rockylinux8", params.action, "skip", "PYTHON_VERSION=3.12.3 STAGE=tritondevel", "", "-py312")
                    }
                }
                stage("Build SBSA-skip") {
                    agent {
                        kubernetes trtllm_utils.createKubernetesPodConfig(
                            image = DOCKER_DIND_IMAGE,
                            type = "build",
                            arch = "arm64",
                            cpuRequest = BUILD_CORES_REQUESTED,
                            cpuLimit = BUILD_CORES_LIMIT,
                            memoryRequest = BUILD_MEMORY_REQUESTED,
                            memoryLimit = BUILD_MEMORY_LIMIT
                        )
                    }
                    steps
                    {
                        buildImage("tritondevel", params.action, "skip", "", "", "", true)
                    }
                }
            }
        }
    } // stages
} // pipeline
