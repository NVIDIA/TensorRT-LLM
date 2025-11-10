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

def getGitCredentialId (String repoUrlKey) {
    if (repoUrlKey == "tensorrt_llm_internal") {
        return 'svc_tensorrt_gitlab_api_token_no_username_as_string'
    } else {
        return 'github-token-trtllm-ci'
    }
}

def generate()
{
    sh "pwd && ls -alh"

    container("alpine") {
        def LLM_REPO = "https://github.com/NVIDIA/TensorRT-LLM.git"
        if (params.repoUrlKey == "tensorrt_llm_internal") {
            withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
                LLM_REPO = DEFAULT_LLM_REPO
            }
        }
        if (params.repoUrlKey == "custom_repo") {
            if (params.customRepoUrl == "") {
                throw new Exception("Invalid custom repo url provided")
            }
            LLM_REPO = params.customRepoUrl
        }
        def CREDENTIAL_ID = getGitCredentialId(params.repoUrlKey)
        sh "apt update"
        sh "apt install -y python3-dev git curl git-lfs"
        sh "git config --global --add safe.directory ${env.WORKSPACE}"
        sh "git config --global user.email \"90828364+tensorrt-cicd@users.noreply.github.com\""
        sh "git config --global user.name \"TensorRT LLM\""
        trtllm_utils.checkoutSource(LLM_REPO, params.branchName, env.WORKSPACE, false, true)
        sh "python3 --version"
        sh "curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.5 python3 -"
        sh "cd ${env.WORKSPACE}"
        sh "/root/.local/bin/poetry -h"
        sh "export PATH=\"/root/.local/bin:\$PATH\" && python3 scripts/generate_lock_file.py"
        def count = sh(script: "git status --porcelain security_scanning/ | wc -l", returnStdout: true).trim()
        echo "Changed/untracked file count: ${count}"
        if (count == "0") {
            echo "No update that needs to be checked in"
        } else {
            sh "git status"
            sh "git add \$(find . -type f \\( -name 'poetry.lock' -o -name 'pyproject.toml' -o -name 'metadata.json' \\))"
            sh "git commit -s -m \"[None][infra] Check in most recent lock file from nightly pipeline\""
            withCredentials([string(credentialsId: CREDENTIAL_ID, variable: 'API_TOKEN')]) {
                def authedUrl = LLM_REPO.replaceFirst('https://', "https://svc_tensorrt:${API_TOKEN}@")
                sh "git remote set-url origin ${authedUrl}"
                sh "git fetch origin ${params.branchName}"
                sh "git status"
                sh "git rebase origin/${params.branchName}"
                sh "git push origin HEAD:${params.branchName}"
            }
        }
    }
}


pipeline {
    agent {
        kubernetes createKubernetesPodConfig()
    }
    parameters {
        string(name: 'branchName', defaultValue: 'main', description: 'the branch to generate the lock files')
        choice(name: 'repoUrlKey', choices: ['tensorrt_llm_github','tensorrt_llm_internal', 'custom_repo'], description: "The repo url to process, choose \"custom_repo\" if you want to set your own repo")
        string(name: 'customRepoUrl', defaultValue: '', description: 'Your custom repo to get processed, need to select \"custom_repo\" for repoUrlKey, otherwise it will be ignored')
    }
    options {
        skipDefaultCheckout()
        // to better analyze the time for each step/test
        timestamps()
    }

    triggers {
        parameterizedCron('''
            H 2 * * * %branchName=main;repoUrlKey=tensorrt_llm_github
        ''')
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
