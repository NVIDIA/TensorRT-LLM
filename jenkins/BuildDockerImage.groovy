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

BUILD_JOBS = "32"

// Utilities
def checkoutSource(String repo, String branch, String directory) {
    def extensionsList = [
        lfs(),
        [
            $class: 'CleanCheckout'
        ],
        [
            $class: 'CloneOption',
            shallow: true,
            depth: 1,
            noTags: true,
            honorRefspec: true,
        ],
        [
            $class: 'RelativeTargetDirectory',
            relativeTargetDir: directory
        ],
        [
            $class: 'SubmoduleOption',
            parentCredentials: true,
            recursiveSubmodules: true,
            shallow: true,
            timeout: 60
        ]
    ]

    def scmSpec = [
        $class: "GitSCM",
        doGenerateSubmoduleConfigurations: false,
        submoduleCfg: [],
        branches: [[name: branch]],
        userRemoteConfigs: [
            [
                credentialsId: "svc_tensorrt_gitlab_api_token",
                name: "origin",
                refspec: "${branch}:refs/remotes/origin/${branch}",
                url: repo,
            ]
        ],
        extensions: extensionsList,
    ]
    echo "Cloning with SCM spec: ${scmSpec.toString()}"
    checkout(scm: scmSpec, changelog: true)
}


def createKubernetesPodConfig(type)
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
        """.stripIndent(),
    ]

    return podConfig
}


def buildImage(target, action="build", torchInstallType="skip", args="", custom_tag="", post_tag="")
{
    def tag = "x86_64-${target}-torch_${torchInstallType}-${LLM_BRANCH_TAG}-${BUILD_NUMBER}${post_tag}"

    // Step 1: cloning tekit source code
    // allow to checkout from forked repo, svc_tensorrt needs to have access to the repo, otherwise clone will fail
    checkoutSource(LLM_REPO, LLM_BRANCH, LLM_ROOT)

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
            containerGenFailure = null
            stage ("make ${target}_${action}") {
                retry(3)
                {
                  sh """
                  cd ${LLM_ROOT} && make -C docker ${target}_${action} \
                  TORCH_INSTALL_TYPE=${torchInstallType} \
                  IMAGE_NAME=${IMAGE_NAME} IMAGE_TAG=${tag} \
                  BUILD_WHEEL_OPTS='-j ${BUILD_JOBS}' ${args} \
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
                  BUILD_WHEEL_OPTS='-j ${BUILD_JOBS}' ${args} \
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


def triggerSBSARemoteJob(action, type)
{
    script
    {
        def parameters = """
            token=L1_Nightly_Token
            hostJobName=${JOB_NAME}
            hostBuildNumber=${BUILD_NUMBER}
            gitlabBranch=${LLM_BRANCH}
            action=${action}
            type=${type}
        """.stripIndent()

        catchError(buildResult: 'FAILURE', stageResult: 'FAILURE')
        {
            def handle = triggerRemoteJob(
                job: "https://prod.blsm.nvidia.com/sw-tensorrt-static-1/job/LLM/job/helpers/job/gh200-BuildImage/",
                auth: CredentialsAuth(credentials: "STATIC_1_TOKEN"),
                parameters: parameters,
                pollInterval: 60,
                abortTriggeredJob: true,
            )
            def status = handle.getBuildResult().toString()

            if (status != "SUCCESS") {
                error "Downstream job did not succeed"
            }
        }
    }
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
        PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
    }
    stages {
        stage("Build")
        {
            parallel {
                stage("Build trtllm release") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("trtllm", "push", "skip", "", LLM_BRANCH_TAG)
                    }
                }
                stage("Build x86_64-skip") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("devel", params.action, "skip")
                    }
                }
                stage("Build x86_64-pre_cxx11_abi") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("devel", params.action, "src_non_cxx11_abi")
                    }
                }
                stage("Build x86_64-cxx11_abi") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("devel", params.action, "src_cxx11_abi")
                    }
                }
                stage("Build rockylinux8 x86_64-skip-py3.10") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("rockylinux8", params.action, "skip", "PYTHON_VERSION=3.10.12", "", "-py310")
                    }
                }
                stage("Build rockylinux8 x86_64-skip-py3.12") {
                    agent {
                        kubernetes createKubernetesPodConfig("build")
                    }
                    steps
                    {
                        buildImage("rockylinux8", params.action, "skip", "PYTHON_VERSION=3.12.3", "", "-py312")
                    }
                }
                stage("Build SBSA-skip") {
                    agent {
                        kubernetes createKubernetesPodConfig("agent")
                    }
                    steps
                    {
                        triggerSBSARemoteJob(params.action, "skip")
                    }
                }
                // Waived due to a pytorch issue: https://github.com/pytorch/pytorch/issues/141083
                // stage("Build SBSA-pre_cxx11_abi") {
                //     agent {
                //         kubernetes createKubernetesPodConfig("agent")
                //     }
                //     steps
                //     {
                //         triggerSBSARemoteJob(params.action, "src_non_cxx11_abi")
                //     }
                // }
                // stage("Build SBSA-cxx11_abi") {
                //     agent {
                //         kubernetes createKubernetesPodConfig("agent")
                //     }
                //     steps
                //     {
                //         triggerSBSARemoteJob(params.action, "src_cxx11_abi")
                //     }
                // }
            }
        }
    } // stages
} // pipeline
