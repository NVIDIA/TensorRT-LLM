@Library(['trtllm-jenkins-shared-lib@main']) _

def createKubernetesPodConfig()
{
    def targetCloud = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux"""
    def image = "urm.nvidia.com/docker/ubuntu:22.04"
    def podConfig = [
        cloud: targetCloud,
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                qosClass: Guaranteed
                nodeSelector: ${selectors}
                containers:
                  - name: alpine
                    image: ${image}
                    command: ['cat']
                    tty: true
                    resources:
                      requests:
                        cpu: '8'
                        memory: 32Gi
                        ephemeral-storage: 200Gi
                      limits:
                        cpu: '8'
                        memory: 32Gi
                        ephemeral-storage: 200Gi
                    imagePullPolicy: Always
                qosClass: Guaranteed
        """.stripIndent(),
    ]

    return podConfig
}

def generate()
{
    sh "pwd && ls -alh"

    container("alpine") {
        LLM_REPO = "https://github.com/NVIDIA/TensorRT-LLM.git"
        sh "apt update"
        sh "apt install -y python3-dev git curl git-lfs"
        sh "git config --global --add safe.directory ${env.WORKSPACE}"
        sh "git config --global user.email \"90828364+tensorrt-cicd@users.noreply.github.com\""
        sh "git config --global user.name \"TensorRT LLM\""
        trtllm_utils.checkoutSource(LLM_REPO, params.llmBranch, env.WORKSPACE, false, true)
        sh "python3 --version"
        sh "curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.5 python3 -"
        sh "cd ${env.WORKSPACE}"
        sh "/root/.local/bin/poetry -h"
        sh "export PATH=\"/root/.local/bin:\$PATH\" && python3 scripts/generate_lock_file.py"
        def count = sh(script: "git status --porcelain security_scanning/ | wc -l", returnStdout: true).trim()
        echo "Changed/untracked file count: ${count}"
        if (count == "0") {
            echo "No changes in Git"
        } else {
            sh "git status"
            sh "git add \$(find . -type f \\( -name 'poetry.lock' -o -name 'pyproject.toml' \\))"
            sh "git commit -s -m \"[None][infra] Check in most recent lock file from nightly pipeline\""
            withCredentials([usernamePassword(credentialsId: 'github-cred-trtllm-ci', usernameVariable: 'GIT_USER', passwordVariable: 'GIT_PASS')]) {
                def authedUrl = LLM_REPO.replaceFirst('https://', "https://${GIT_USER}:${GIT_PASS}@")
                sh "git remote set-url origin ${authedUrl}"
                sh "git fetch origin ${params.llmBranch}"
                sh "git status"
                sh "git rebase origin/${params.llmBranch}"
                sh "git push origin HEAD:${params.llmBranch}"
            }
        }
    }
}


pipeline {
    agent {
        kubernetes createKubernetesPodConfig()
    }
    parameters {
        string(name: 'llmBranch', defaultValue: 'main', description: 'the branch to generate the lock files')
    }
    options {
        skipDefaultCheckout()
        // to better analyze the time for each step/test
        timestamps()
    }

    stages {
        stage("Generating Poetry Locks"){
            agent {
                kubernetes createKubernetesPodConfig()
            }
            steps
            {
                generate()
            }
        }
    } // stages
} // pipeline
