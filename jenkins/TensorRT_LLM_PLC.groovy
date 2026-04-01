@Library(['trtllm-jenkins-shared-lib@main']) _
import groovy.json.JsonSlurper

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
        sh "apt install -y git git-lfs openjdk-17-jdk python3-dev python3-venv curl zip unzip wget"
    }
}

def validateBranchName(String branch) {
    container("cpu") {
        def rc = sh(script: "git check-ref-format --branch '${branch}'", returnStatus: true)
        if (rc != 0) {
            error("Invalid branch name: '${branch}'")
        }
    }
}

def checkoutSource ()
{
    container("cpu") {
        trtllm_utils.setupGitMirror()
        def LLM_REPO = getLLMRepo()
        sh "git config --global --add safe.directory ${env.WORKSPACE}"
        trtllm_utils.checkoutSource(LLM_REPO, params.branchName, env.WORKSPACE, false, true)
    }
}

def getPulseToken(serviceId, scopes) {
    def token
    //Configure credential 'starfleet-client-id' under Jenkins Credential Manager
    withCredentials([usernamePassword(
        credentialsId: "NSPECT_CLIENT-prod",
        usernameVariable: 'SF_CLIENT_ID',
        passwordVariable: 'SF_CLIENT_SECRET'
    )]) {
        // Do not save AUTH_HEADER to a groovy variable since that
        // will expose the auth_header without being masked
        token= sh(script: """
            AUTH_HEADER=\$(echo -n \$SF_CLIENT_ID:\$SF_CLIENT_SECRET | base64 -w0)
            curl -s --request POST --header "Authorization: Basic \$AUTH_HEADER" --header "Content-Type: application/x-www-form-urlencoded" "https://${serviceId}.ssa.nvidia.com/token?grant_type=client_credentials&scope=${scopes}" | jq ".access_token" |  tr -d '"'
        """, returnStdout: true).trim()
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

def sonarScan()
{
    container("cpu") {
        def sonarScannerCliVer = "8.0.0.6341"
        sh "wget https://repo1.maven.org/maven2/org/sonarsource/scanner/cli/sonar-scanner-cli/${sonarScannerCliVer}/sonar-scanner-cli-${sonarScannerCliVer}.zip"
        sh "unzip sonar-scanner-cli-${sonarScannerCliVer}.zip"
        sh "mv sonar-scanner-${sonarScannerCliVer} ../sonar-scanner"
        sh "rm sonar-scanner-cli-${sonarScannerCliVer}.zip"
        withSonarQubeEnv() {
          sh "../sonar-scanner/bin/sonar-scanner -Dsonar.projectKey=GPUSW_TensorRT-LLM-Team_TensorRT-LLM_tensorrt-llm -Dsonar.sources=. -Dsonar.branch.name=${params.branchName}"
        }
    }
}

def pulseScanSourceCode(llmRepo, branchName) {
    container("docker") {
        sh "apk add jq curl"
        def token = getPulseToken("4ubglassowmtsi7ogqwarmut7msn1q5ynts62fwnr1i", "verify:nspectid%20sourcecode:blackduck%20update:report")
        if (!token) {
            throw new Exception("Invalid token get")
        }
        withCredentials([
            usernamePassword(
                credentialsId: "svc_tensorrt_gitlab_read_api_token",
                usernameVariable: 'GITLAB_USERNAME',
                passwordVariable: 'GITLAB_PASSWORD'
            ),
            string(credentialsId: 'default-git-url', variable: 'DEFAULT_GIT_URL')
        ]) {
            trtllm_utils.llmExecStepWithRetry(this, script: "docker login ${DEFAULT_GIT_URL}:5005 -u ${GITLAB_USERNAME} -p ${GITLAB_PASSWORD}")
            docker.withRegistry("https://${DEFAULT_GIT_URL}:5005") {
                docker.image("pstooling/pulse-group/pulse-open-source-scanner/pulse-oss-cli:stable")
                  .inside("--user 0 --privileged -v /var/run/docker.sock:/var/run/docker.sock") {
                    def versionMatcher = branchName =~ /^release\/(\d+\.\d+)$/
                    def version = versionMatcher ? "${versionMatcher[0][1]}.0" : branchName
                    withEnv([
                        "PULSE_NSPECT_ID=NSPECT-95LK-6FZF",
                        "PULSE_BEARER_TOKEN=${token}",
                        "PULSE_REPO_URL=${llmRepo}",
                        "PULSE_REPO_BRANCH=${(params.repoUrlKey == "github_fork") ? "" : branchName}",
                        "PULSE_SCAN_PROJECT=TRT-LLM",
                        "PULSE_SCAN_PROJECT_VERSION=${version}",
                        "PULSE_SCAN_VULNERABILITY_REPORT=nspect_scan_report.json",
                        "PULSE_SCAN_OVERRIDE=false"
                    ]) {
                        sh 'pulse scan --no-fail --sbom .'
                    }
                  }
            }
        }
    }
    container("cpu") {
        def outputDir = "scan_report/source_code"
        sh "mkdir -p ${outputDir}"
        sh "unzip -p sbom.zip \"*.json\" > ${outputDir}/sbom.json"
        sh "mv nspect_scan_report.json ${outputDir}/vulns.json"
    }
}
def pulseScanContainer(llmRepo, branchName) {
    // imageTags: key -> [image: <full image:tag>, platform: <platform or empty>]
    def imageTags = [:]
    container("cpu") {
        def output = sh(
            script: "python3 ./jenkins/scripts/get_image_key_to_tag.py ${params.branchName}",
            returnStdout: true
        ).trim()
        println(output)
        def containerTagMap = new JsonSlurper().parseText(output)
        imageTags["release_amd64"] = [image: containerTagMap["NGC Release Image amd64"], platform: "linux/amd64"]
        imageTags["release_arm64"] = [image: containerTagMap["NGC Release Image arm64"], platform: "linux/arm64"]

        def baseImage = sh(script: "grep -m1 '^ARG BASE_IMAGE=' docker/Dockerfile.multi | cut -d= -f2", returnStdout: true).trim()
        def baseTag = sh(script: "grep -m1 '^ARG BASE_TAG=' docker/Dockerfile.multi | cut -d= -f2", returnStdout: true).trim()
        imageTags["base_amd64"] = [image: "${baseImage}:${baseTag}", platform: "linux/amd64"]
        imageTags["base_arm64"] = [image: "${baseImage}:${baseTag}", platform: "linux/arm64"]
    }
    container("docker") {
        sh "apk add jq curl"
        def token = getPulseToken("x9thwm-cootr2q1jdv5p7b8iw4fs4ob3x6nqqsoznyk", "nspect.verify%20scan.anchore")
        if (!token) {
            throw new Exception("Invalid token get")
        }
        withCredentials([
            usernamePassword(
                credentialsId: "svc_tensorrt_gitlab_read_api_token",
                usernameVariable: 'GITLAB_USERNAME',
                passwordVariable: 'GITLAB_PASSWORD'
            ),
            usernamePassword(
                credentialsId: "urm-artifactory-creds",
                usernameVariable: 'URM_USERNAME',
                passwordVariable: 'URM_PASSWORD'
            ),
            string(credentialsId: 'default-git-url', variable: 'DEFAULT_GIT_URL'),
        ]) {
            trtllm_utils.llmExecStepWithRetry(this, script: "docker login ${DEFAULT_GIT_URL}:5005 -u ${GITLAB_USERNAME} -p ${GITLAB_PASSWORD}")
            trtllm_utils.llmExecStepWithRetry(this, script: "docker login urm.nvidia.com -u ${URM_USERNAME} -p ${URM_PASSWORD}")
            docker.withRegistry("https://${DEFAULT_GIT_URL}:5005") {
                docker.image("gitlab-master.nvidia.com:5005/pstooling/pulse-group/pulse-container-scanner/pulse-cli:5.1.0")
                .inside("--user 0 --privileged -v /var/run/docker.sock:/var/run/docker.sock") {
                    withEnv([
                        "NSPECT_ID=NSPECT-95LK-6FZF",
                        "SSA_TOKEN=${token}",
                    ]) {
                        imageTags.each { key, entry ->
                            def platform = entry.platform.replace("linux/", "")
                            def outputDir = "scan_report/${key}"
                            sh "mkdir -p ${outputDir}"
                            echo "Scanning ${key}: ${entry.image} (${entry.platform}) -> ${outputDir}"
                            sh "pulse-cli -n \$NSPECT_ID --ssa \$SSA_TOKEN scan-image -i ${entry.image} --platform ${entry.platform} --sbom=cyclonedx-json --output-dir=${outputDir} -o"
                        }
                    }
                }
            }
        }
    }
}

def processScanResults(branchName) {
    container("cpu") {
        def ELASTICSEARCH_POST_URL = "http://nvdataflow.nvidia.com/dataflow/swdl-tensorrt-infra-plc-scan/posting"
        def ELASTICSEARCH_QUERY_URL = "https://gpuwa.nvidia.com/elasticsearch"
        def TRTLLM_ES_INDEX_BASE = "df-swdl-tensorrt-infra-plc-scan"
        def TRTLLM_ES_INDEX_PREAPPROVED_BASE = "df-swdl-tensorrt-infra-plc-container-pre-approve"
        def jobPath = env.JOB_NAME.replaceAll("/", "%2F")
        def pipelineUrl = "${env.JENKINS_URL}blue/organizations/jenkins/${jobPath}/detail/${jobPath}/${env.BUILD_NUMBER}/pipeline"
        withCredentials([string(credentialsId: 'trtllm_plc_slack_webhook', variable: 'PLC_SLACK_WEBHOOK')]) {
            withEnv([
                "TRTLLM_ES_POST_URL=${ELASTICSEARCH_POST_URL}",
                "TRTLLM_ES_QUERY_URL=${ELASTICSEARCH_QUERY_URL}",
                "TRTLLM_ES_INDEX_BASE=${TRTLLM_ES_INDEX_BASE}",
                "TRTLLM_ES_INDEX_PREAPPROVED_BASE=${TRTLLM_ES_INDEX_PREAPPROVED_BASE}",
                "TRTLLM_PLC_WEBHOOK=${PLC_SLACK_WEBHOOK}"
            ]) {
                sh """
                    python3 -m venv venv
                    venv/bin/pip install requests elasticsearch==7.13.4
                    venv/bin/python ./jenkins/scripts/pulse_in_pipeline_scanning/main.py \
                        --build-url ${pipelineUrl} \
                        --build-number ${env.BUILD_NUMBER} \
                        --branch ${branchName} \
                        --report-directory ${pwd()}/scan_report
                """
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
        // Schedule is only active when running from the official pipeline folder.
        // Jobs in other folders (e.g. personal/dev pipelines) will have no cron trigger.
        parameterizedCron(env.JOB_NAME.startsWith('LLM/helpers/') ? '''
            H 2 * * * %branchName=main;repoUrlKey=tensorrt_llm_github
        ''' : '')
    }
    stages {
        stage("Prepare Environment"){
            steps {
                script {
                    installTools()
                    checkoutSource()
                    validateBranchName(params.branchName)
                }
            }
        }
        stage('Run TRT-LLM PLC Jobs') {
            parallel {
                stage("Source Code OSS Scanning") {
                    steps {
                        script {
                            generateLockFiles(env.LLM_REPO, env.BRANCH_NAME)
                            pulseScanSourceCode(env.LLM_REPO, env.BRANCH_NAME)
                        }
                    }
                }
                stage("Run Container Scanning") {
                    steps {
                        script {
                            pulseScanContainer(env.LLM_REPO, env.BRANCH_NAME)
                        }
                    }
                }
                stage("SonarQube Code Analysis") {
                    steps {
                        script {
                            sonarScan()
                        }
                    }
                }
            }
        }
        stage("Process Scan Result") {
            steps {
                script {
                    processScanResults(env.BRANCH_NAME)
                }
            }
        }
    } // stages
} // pipeline
