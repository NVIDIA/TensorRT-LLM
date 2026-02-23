@Library(['trtllm-jenkins-shared-lib@main']) _

def createKubernetesPodConfig()
{
    def targetCloud = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux"""
    def image = "urm.nvidia.com/docker/ubuntu:24.04"
    def podConfig = [
        cloud: targetCloud,
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                nodeSelector: ${selectors}
                containers:
                  - name: cpu
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
                        - SYS_ADMIN
                qosClass: Guaranteed
        """.stripIndent(),
    ]
    return podConfig
}

boolean isValidGithubUser(String owner) {
    def pattern = ~/^(?!-)(?!.*--)[A-Za-z0-9-]{1,39}(?<!-)$/
    return owner ==~ pattern
}

def getLLMRepo () {
    def LLM_REPO = "https://github.com/NVIDIA/TensorRT-LLM.git"
    if (params.repoUrlKey == "tensorrt_llm_internal") {
        withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
            LLM_REPO = DEFAULT_LLM_REPO
        }
    }
    if (params.repoUrlKey == "github_fork") {
        if (!isValidGithubUser(params.forkOwner)) {
            throw new Exception("Invalid fork owner provided")
        }
        LLM_REPO = "https://github.com/${params.forkOwner}/TensorRT-LLM.git"
    }
    return LLM_REPO
}

def installTools() {
    container("cpu") {
        sh "apt update"
        sh "apt install -y git git-lfs openjdk-17-jdk python3-dev python3-venv curl unzip wget"
    }
}

def checkoutSource ()
{
    container("cpu") {
        trtllm_utils.setupGitMirror()
        stage("Checkout TRTLLM Source") {
            def LLM_REPO = getLLMRepo()
            sh "git config --global --add safe.directory ${env.WORKSPACE}"
            trtllm_utils.checkoutSource(LLM_REPO, params.branchName, env.WORKSPACE, false, true)
        }
    }
}

def getPulseToken() {
    def token
    //Configure credential 'starfleet-client-id' under Jenkins Credential Manager
    withCredentials([usernamePassword(
        credentialsId: "NSPECT_CLIENT-prod",
        usernameVariable: 'SF_CLIENT_ID',
        passwordVariable: 'SF_CLIENT_SECRET'
    )]) {
        // Do not save AUTH_HEADER to a groovy variable since that
        // will expose the auth_header without being masked
        token= sh(script: '''
            AUTH_HEADER=$(echo -n $SF_CLIENT_ID:$SF_CLIENT_SECRET | base64 -w0)
            curl -s --request POST --header "Authorization: Basic $AUTH_HEADER" --header "Content-Type: application/x-www-form-urlencoded" "https://4ubglassowmtsi7ogqwarmut7msn1q5ynts62fwnr1i.ssa.nvidia.com/token?grant_type=client_credentials&scope=verify:nspectid%20sourcecode:blackduck%20update:report" | jq ".access_token" |  tr -d '"'
        ''', returnStdout: true).trim()
    }
    return token
}

def generateLockFiles(llmRepo, branchName)
{
    container("cpu") {
        sh "python3 --version"
        sh "curl -sSL https://install.python-poetry.org | python3 -"
        sh "/root/.local/bin/poetry -h"
        sh "git config --global --add safe.directory ${env.WORKSPACE}"
        sh "git config --global user.email \"90828364+tensorrt-cicd@users.noreply.github.com\""
        sh "git config --global user.name \"TensorRT LLM\""
        sh "export PATH=\"/root/.local/bin:\$PATH\" && python3 scripts/generate_lock_file.py"
        def count = sh(script: "git status --porcelain security_scanning/ | wc -l", returnStdout: true).trim()
        echo "Changed/untracked file count: ${count}"
        if (count == "0") {
            echo "No update that needs to be checked in"
        } else {
            sh "git status"
            sh "git add -u security_scanning/"
            sh "git add \$(find . -type f \\( -name 'poetry.lock' -o -name 'pyproject.toml' -o -name 'metadata.json' \\))"
            sh "git commit -s -m \"[None][infra] Check in most recent lock file from nightly pipeline\""
            withCredentials([
                string(credentialsId: 'svc_tensorrt_gitlab_api_token_no_username_as_string', variable: 'GITLAB_API_TOKEN'),
                usernamePassword(
                    credentialsId: 'github-cred-trtllm-ci',
                    usernameVariable: 'NOT_IN_USE',
                    passwordVariable: 'GITHUB_API_TOKEN'
                )
            ]) {
                def authedUrl
                if (params.repoUrlKey == "tensorrt_llm_internal") {
                    authedUrl = llmRepo.replaceFirst('https://', "https://svc_tensorrt:${GITLAB_API_TOKEN}@")
                } else {
                    authedUrl = llmRepo.replaceFirst('https://', "https://svc_tensorrt:${GITHUB_API_TOKEN}@")
                }
                sh "git remote set-url origin ${authedUrl}"
                sh "git fetch origin ${branchName}"
                sh "git status"
                sh "git rebase origin/${branchName}"
                sh "git push origin HEAD:${branchName}"
            }
        }
    }
}

def sonar_scan()
{
    container("cpu") {
        def sonarScannerCliVer = "8.0.0.6341"
        sh "wget https://repo1.maven.org/maven2/org/sonarsource/scanner/cli/sonar-scanner-cli/${sonarScannerCliVer}/sonar-scanner-cli-${sonarScannerCliVer}.zip"
        sh "unzip sonar-scanner-cli-${sonarScannerCliVer}.zip"
        sh "mv sonar-scanner-${sonarScannerCliVer} ./sonar-scanner"
        withSonarQubeEnv() {
          sh "./sonar-scanner/bin/sonar-scanner -Dsonar.projectKey=GPUSW_TensorRT-LLM-Team_TensorRT-LLM_tensorrt-llm -Dsonar.sources=. -Dsonar.branch.name=${params.branchName}"
        }
    }
}

def pulseScan(llmRepo, branchName) {
    container("docker") {
        sh "apk add jq curl"
        def token = getPulseToken()
        if (!token) {
            throw new Exception("Invalid token get")
        }
        withCredentials([
            usernamePassword(
                credentialsId: "svc_tensorrt_gitlab_read_api_token",
                usernameVariable: 'USERNAME',
                passwordVariable: 'PASSWORD'
            ),
            string(credentialsId: 'default-git-url', variable: 'DEFAULT_GIT_URL')
        ]) {
            trtllm_utils.llmExecStepWithRetry(this, script: "docker login ${DEFAULT_GIT_URL}:5005 -u ${USERNAME} -p ${PASSWORD}")
            docker.withRegistry("https://${DEFAULT_GIT_URL}:5005") {
                docker.image("pstooling/pulse-group/pulse-open-source-scanner/pulse-oss-cli:stable")
                  .inside("--user 0 --privileged -v /var/run/docker.sock:/var/run/docker.sock") {
                    withEnv([
                        "PULSE_NSPECT_ID=NSPECT-95LK-6FZF",
                        "PULSE_BEARER_TOKEN=${token}",
                        "PULSE_REPO_URL=${llmRepo}",
                        "PULSE_REPO_BRANCH=${(params.repoUrlKey == "github_fork") ? "" : branchName}",
                        "PULSE_SCAN_PROJECT=TRT-LLM",
                        "PULSE_SCAN_PROJECT_VERSION=${branchName.replace("release/", "")}",
                        "PULSE_SCAN_VULNERABILITY_REPORT=nspect_scan_report.json"
                    ]) {
                        if (params.repoUrlKey == "github_fork") {
                            sh 'pulse scan --no-fail --sbom .'
                        } else {
                            sh 'pulse scan --no-fail --sbom --override .'
                        }
                    }
                  }
            }
        }
    }
    container("cpu") {
        sh "cat nspect_scan_report.json"
        sh "cat sbom.cdx.json"
        sh """
            SBOM_ZIP="./sbom.zip"
            if [ -f "\$SBOM_ZIP" ]; then
                EXTRACTED_FOLDER=\$(unzip -Z1 "\$SBOM_ZIP" | head -1 | cut -d/ -f1)
                JSON_FILE=\$(find "\$EXTRACTED_FOLDER" -type f -name "*.json" | head -n 1)
                if [ -n "\$JSON_FILE" ]; then
                    cat "\$JSON_FILE"
                else
                    echo "No JSON file found in SBOM archive"
                fi
            else
                echo "SBOM zip does not exist"
            fi
        """
        withCredentials([string(credentialsId: 'trtllm_plc_slack_webhook', variable: 'PLC_SLACK_WEBHOOK')]) {
            def jobPath = env.JOB_NAME.replaceAll("/", "%2F")
            def pipelineUrl = "${env.JENKINS_URL}blue/organizations/jenkins/${jobPath}/detail/${jobPath}/${env.BUILD_NUMBER}/pipeline"
            sh """
                export TRTLLM_PLC_WEBHOOK=${PLC_SLACK_WEBHOOK}
                python3 -m venv venv
                venv/bin/pip install requests
                venv/bin/python ./jenkins/scripts/submit_vulnerability_report.py --build-url ${pipelineUrl}
            """
        }
    }
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig()
    }
    parameters {
        string(name: 'branchName', defaultValue: 'main', description: 'the branch to generate the lock files')
        choice(name: 'repoUrlKey', choices: ['tensorrt_llm_github','tensorrt_llm_internal', 'github_fork'], description: "The repo url to process")
        string(name: 'forkOwner', defaultValue: '', description: 'Name of the fork owner, need to select \"github_fork\" for repoUrlKey, otherwise it will be ignored')
    }
    options {
        skipDefaultCheckout()
        timestamps()
        timeout(time: 150, unit: 'MINUTES')
    }
    environment {
        LLM_REPO = getLLMRepo()
        BRANCH_NAME = "${params.branchName}"
    }

    triggers {
        parameterizedCron('''
            H 2 * * * %branchName=main;repoUrlKey=tensorrt_llm_github
            H 3 * * * %branchName=release/1.2;repoUrlKey=tensorrt_llm_github
        ''')
    }
    stages {
        stage("Prepare Environment"){
            steps {
                script {
                    installTools()
                    checkoutSource()
                }
            }
        }
        stage('Run TRT-LLM PLC Jobs') {
            parallel {
                stage("Source Code OSS Scanning"){
                    stages {
                        stage("Generate Lock Files"){
                            steps
                            {
                                script {
                                    generateLockFiles(env.LLM_REPO, env.BRANCH_NAME)
                                }
                            }
                        }
                        stage("Run Pulse Scanning"){
                            steps
                            {
                                script {
                                    pulseScan(env.LLM_REPO, env.BRANCH_NAME)
                                }
                            }
                        }
                    }
                }
                stage("SonarQube Code Analysis"){
                    steps
                    {
                        script {
                            sonar_scan()
                        }
                    }
                }
            }
        }
    } // stages
} // pipeline
