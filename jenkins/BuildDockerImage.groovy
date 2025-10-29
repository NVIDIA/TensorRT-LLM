@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.Exception
import groovy.transform.Field

// Docker image registry
IMAGE_NAME = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging"
NGC_IMAGE_NAME = "${IMAGE_NAME}/ngc"

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}

ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
UPLOAD_PATH = env.uploadPath ? env.uploadPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"

LLM_ROOT = "llm"

LLM_BRANCH = env.gitlabBranch ?: params.branch
LLM_BRANCH_TAG = LLM_BRANCH.replaceAll('/', '_')

LLM_COMMIT_OR_BRANCH = env.gitlabCommit ?: LLM_BRANCH

LLM_SHORT_COMMIT = env.gitlabCommit ? env.gitlabCommit.substring(0, 7) : "undefined"

LLM_DEFAULT_TAG = env.defaultTag ?: "${LLM_SHORT_COMMIT}-${LLM_BRANCH_TAG}-${BUILD_NUMBER}"

RUN_SANITY_CHECK = params.runSanityCheck ?: false
TRIGGER_TYPE = env.triggerType ?: "manual"

ENABLE_USE_WHEEL_FROM_BUILD_STAGE = params.useWheelFromBuildStage ?: false

WAIT_TIME_FOR_BUILD_STAGE = 60  // minutes

BUILD_JOBS = "32"
BUILD_JOBS_RELEASE_X86_64 = "32"
BUILD_JOBS_RELEASE_SBSA = "32"

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

    if (arch == "amd64") {
        // For x86_64, we block some nodes to avoid unstable network access.
        selectors += """
                affinity:
                    nodeAffinity:
                        requiredDuringSchedulingIgnoredDuringExecution:
                            nodeSelectorTerms:
                                - matchExpressions:
                                    - key: "kubernetes.io/hostname"
                                      operator: NotIn
                                      values:
                                        - "sc-ipp-blossom-prod-k8w-105"
                                        - "sc-ipp-blossom-prod-k8w-114"
                                        - "sc-ipp-blossom-prod-k8w-115"
                                        - "sc-ipp-blossom-prod-k8w-121"
                                        - "sc-ipp-blossom-prod-k8w-123"
                                        - "sc-ipp-blossom-prod-k8w-124"
        """
    }

    def archSuffix = arch == "arm64" ? "arm" : "amd"
    def jnlpImage = "urm.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_${archSuffix}_linux:jdk17"

    switch(type)
    {
    case "agent":
        containerConfig = """
                  - name: python3
                    image: urm.nvidia.com/docker/python:3.12-slim
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
                    image: ${jnlpImage}
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


def prepareWheelFromBuildStage(dockerfileStage, arch) {
    if (!ENABLE_USE_WHEEL_FROM_BUILD_STAGE) {
        echo "useWheelFromBuildStage is false, skip preparing wheel from build stage"
        return ""
    }

    if (TRIGGER_TYPE != "post-merge") {
        echo "Trigger type is not post-merge, skip preparing wheel from build stage"
        return ""
    }

    if (!dockerfileStage || !arch) {
        echo "Error: dockerfileStage and arch are required parameters"
        return ""
    }

    if (dockerfileStage != "release") {
        echo "prepareWheelFromBuildStage: ${dockerfileStage} is not release"
        return ""
    }

    def wheelScript = 'scripts/get_wheel_from_package.py'
    def wheelArgs = "--arch ${arch} --timeout ${WAIT_TIME_FOR_BUILD_STAGE} --artifact_path " + env.uploadPath
    return " BUILD_WHEEL_SCRIPT=${wheelScript} BUILD_WHEEL_ARGS='${wheelArgs}'"
}

def buildImage(config, imageKeyToTag)
{
    def target = config.target
    def action = config.action
    def torchInstallType = config.torchInstallType
    def args = config.args ?: ""
    def customTag = config.customTag
    def postTag = config.postTag
    def dependent = config.dependent
    def arch = config.arch == 'arm64' ? 'sbsa' : 'x86_64'
    def dockerfileStage = config.dockerfileStage

    def tag = "${arch}-${target}-torch_${torchInstallType}${postTag}-${LLM_DEFAULT_TAG}"

    def dependentTag = tag.replace("${arch}-${target}-", "${arch}-${dependent.target}-")

    def imageWithTag = "${IMAGE_NAME}/${dockerfileStage}:${tag}"
    def dependentImageWithTag = "${IMAGE_NAME}/${dependent.dockerfileStage}:${dependentTag}"
    def customImageWithTag = "${IMAGE_NAME}/${dockerfileStage}:${customTag}"

    if (target == "ngc-release" && TRIGGER_TYPE == "post-merge") {
        echo "Use NGC artifacts for post merge build"
        dependentImageWithTag = "${NGC_IMAGE_NAME}:${dependentTag}"
        imageWithTag = "${NGC_IMAGE_NAME}:${tag}"
        customImageWithTag = "${NGC_IMAGE_NAME}:${customTag}"
    }

    args += " GITHUB_MIRROR=https://urm.nvidia.com/artifactory/github-go-remote"

    stage (config.stageName) {
        // Step 1: Clone TRT-LLM source codes
        // If using a forked repo, svc_tensorrt needs to have the access to the forked repo.
        trtllm_utils.checkoutSource(LLM_REPO, LLM_COMMIT_OR_BRANCH, LLM_ROOT, true, true)
    }

    // Step 2: Build the images
    stage ("Install packages") {
        sh "pwd && ls -alh"
        sh "env | sort"
        sh "apk add make git"
        sh "git config --global --add safe.directory '*'"

        withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
            trtllm_utils.llmExecStepWithRetry(this, script: "docker login urm.nvidia.com -u ${USERNAME} -p ${PASSWORD}")
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
        }
    }
    def containerGenFailure = null
    try {
        def build_jobs = BUILD_JOBS
        // Fix the triton image pull timeout issue
        def BASE_IMAGE = sh(script: "cd ${LLM_ROOT} && grep '^ARG BASE_IMAGE=' docker/Dockerfile.multi | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()
        def TRITON_IMAGE = sh(script: "cd ${LLM_ROOT} && grep '^ARG TRITON_IMAGE=' docker/Dockerfile.multi | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()
        def TRITON_BASE_TAG = sh(script: "cd ${LLM_ROOT} && grep '^ARG TRITON_BASE_TAG=' docker/Dockerfile.multi | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()

        if (target == "rockylinux8") {
            BASE_IMAGE = sh(script: "cd ${LLM_ROOT} && grep '^jenkins-rockylinux8_%: BASE_IMAGE =' docker/Makefile | grep -o '=.*' | tr -d '=\"'", returnStdout: true).trim()
        }

        // Replace the base image and triton image with the internal mirror
        BASE_IMAGE = BASE_IMAGE.replace("nvcr.io/", "urm.nvidia.com/docker/")
        TRITON_IMAGE = TRITON_IMAGE.replace("nvcr.io/", "urm.nvidia.com/docker/")

        if (dependent) {
            stage ("make ${dependent.target}_${action} (${arch})") {
                def randomSleep = (Math.random() * 600 + 600).toInteger()
                trtllm_utils.llmExecStepWithRetry(this, script: "docker pull ${TRITON_IMAGE}:${TRITON_BASE_TAG}", sleepInSecs: randomSleep, numRetries: 6, shortCommondRunTimeMax: 7200)
                trtllm_utils.llmExecStepWithRetry(this, script: """
                cd ${LLM_ROOT} && make -C docker ${dependent.target}_${action} \
                BASE_IMAGE=${BASE_IMAGE} \
                TRITON_IMAGE=${TRITON_IMAGE} \
                TORCH_INSTALL_TYPE=${torchInstallType} \
                IMAGE_WITH_TAG=${dependentImageWithTag} \
                STAGE=${dependent.dockerfileStage} \
                BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args}
                """, sleepInSecs: randomSleep, numRetries: 6, shortCommondRunTimeMax: 7200)
                args += " DEVEL_IMAGE=${dependentImageWithTag}"
                if (target == "ngc-release") {
                    imageKeyToTag["NGC Devel Image ${config.arch}"] = dependentImageWithTag
                }
            }
        }

        args += prepareWheelFromBuildStage(dockerfileStage, arch)
        // Avoid the frequency of OOM issue when building the wheel
        if (target == "trtllm") {
            if (arch == "x86_64") {
                build_jobs = BUILD_JOBS_RELEASE_X86_64
            } else {
                build_jobs = BUILD_JOBS_RELEASE_SBSA
            }
        }
        stage ("make ${target}_${action} (${arch})") {
            sh "env | sort"
            def randomSleep = (Math.random() * 600 + 600).toInteger()
            trtllm_utils.llmExecStepWithRetry(this, script: "docker pull ${TRITON_IMAGE}:${TRITON_BASE_TAG}", sleepInSecs: randomSleep, numRetries: 6, shortCommondRunTimeMax: 7200)
            trtllm_utils.llmExecStepWithRetry(this, script: """
            cd ${LLM_ROOT} && make -C docker ${target}_${action} \
            BASE_IMAGE=${BASE_IMAGE} \
            TRITON_IMAGE=${TRITON_IMAGE} \
            TORCH_INSTALL_TYPE=${torchInstallType} \
            IMAGE_WITH_TAG=${imageWithTag} \
            STAGE=${dockerfileStage} \
            BUILD_WHEEL_OPTS='-j ${build_jobs}' ${args}
            """, sleepInSecs: randomSleep, numRetries: 6, shortCommondRunTimeMax: 7200)
            if (target == "ngc-release") {
                imageKeyToTag["NGC Release Image ${config.arch}"] = imageWithTag
            }
        }

        if (customTag) {
            stage ("custom tag: ${customTag} (${arch})") {
                sh """
                cd ${LLM_ROOT} && make -C docker ${target}_${action} \
                BASE_IMAGE=${BASE_IMAGE} \
                TRITON_IMAGE=${TRITON_IMAGE} \
                TORCH_INSTALL_TYPE=${torchInstallType} \
                IMAGE_WITH_TAG=${customImageWithTag} \
                STAGE=${dockerfileStage} \
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
        dependent: [:],
        dockerfileStage: "tritondevel",
    ]

    def release_action = params.action
    def buildConfigs = [
        "Build trtllm release (x86_64)": [
            target: "trtllm",
            action: release_action,
            customTag: LLM_BRANCH_TAG + "-x86_64",
            build_wheel: true,
            dockerfileStage: "release",
        ],
        "Build trtllm release (SBSA)": [
            target: "trtllm",
            action: release_action,
            customTag: LLM_BRANCH_TAG + "-sbsa",
            build_wheel: true,
            arch: "arm64",
            dockerfileStage: "release",
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
        "Build CI image (RockyLinux8 Python312)": [
            target: "rockylinux8",
            args: "PYTHON_VERSION=3.12.3",
            postTag: "-py312",
        ],
        "Build NGC devel and release (x86_64)": [
            target: "ngc-release",
            action: release_action,
            args: "DOCKER_BUILD_OPTS='--load --platform linux/amd64'",
            build_wheel: true,
            dependent: [
                target: "ngc-devel",
                dockerfileStage: "devel",
            ],
            dockerfileStage: "release",
        ],
        "Build NGC devel and release (SBSA)": [
            target: "ngc-release",
            action: release_action,
            args: "DOCKER_BUILD_OPTS='--load --platform linux/arm64'",
            arch: "arm64",
            build_wheel: true,
            dependent: [
                target: "ngc-devel",
                dockerfileStage: "devel",
            ],
            dockerfileStage: "release",
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
                    try {
                        trtllm_utils.launchKubernetesPod(pipeline, config.podConfig, "docker") {
                            buildImage(config, imageKeyToTag)
                        }
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception e) {
                        catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                            echo "Build ${key} failed."
                            throw e
                        }
                    }
                }
            }
        }]
    }

    echo "enableFailFast is: ${params.enableFailFast}, but we currently don't use it due to random ucxx issue"
    // pipeline.failFast = params.enableFailFast
    pipeline.parallel buildJobs

}


def getCommonParameters()
{
    return [
        'gitlabSourceRepoHttpUrl': LLM_REPO,
        'gitlabCommit': env.gitlabCommit,
        'artifactPath': ARTIFACT_PATH,
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
        stage("Setup Environment") {
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
                    sh "env | sort"
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
                    trtllm_utils.uploadArtifacts("imageKeyToTag.json", "${UPLOAD_PATH}/")
                }
            }
        }
        stage("Wait for Build Jobs Complete") {
            when {
                expression {
                    RUN_SANITY_CHECK
                }
            }
            steps {
                script {
                    catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                        container("python3") {
                            // Install wget
                            trtllm_utils.llmExecStepWithRetry(this, script: "apt-get update && apt-get -y install wget")

                            // Poll for build artifacts
                            def artifactBaseUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/"
                            def requiredFiles = [
                                "TensorRT-LLM-GH200.tar.gz",
                                "TensorRT-LLM.tar.gz"
                            ]
                            def maxWaitMinutes = 60
                            def pollIntervalSeconds = 60

                            echo "Waiting for build artifacts..."
                            echo "Required files: ${requiredFiles}"

                            def startTime = System.currentTimeMillis()
                            def maxWaitMs = maxWaitMinutes * 60 * 1000

                            while ((System.currentTimeMillis() - startTime) < maxWaitMs) {
                                def missingFiles = []

                                for (file in requiredFiles) {
                                    def fileUrl = "${artifactBaseUrl}${file}"
                                    def exitCode = sh(
                                        script: "wget --spider --quiet --timeout=30 --tries=1 '${fileUrl}'",
                                        returnStatus: true
                                    )

                                    if (exitCode != 0) {
                                        missingFiles.add(file)
                                    }
                                }

                                if (missingFiles.isEmpty()) {
                                    echo "All build artifacts are ready!"
                                    return
                                }

                                def elapsedMinutes = (System.currentTimeMillis() - startTime) / (60 * 1000)
                                echo "Waiting... (${elapsedMinutes.intValue()} minutes elapsed)"
                                echo "Missing files: ${missingFiles}"
                                sleep(pollIntervalSeconds)
                            }

                            def elapsedMinutes = (System.currentTimeMillis() - startTime) / (60 * 1000)
                            error "Timeout waiting for build artifacts (${elapsedMinutes.intValue()} minutes)"
                        }
                    }
                }
            }
        }
        stage("Sanity Check for NGC Images") {
            when {
                expression {
                    RUN_SANITY_CHECK
                }
            }
            steps {
                script {
                    catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                        globalVars[IMAGE_KEY_TO_TAG] = imageKeyToTag
                        String globalVarsJson = writeJSON returnText: true, json: globalVars
                        def parameters = getCommonParameters()
                        parameters += [
                            'enableFailFast': false,
                            'globalVars': globalVarsJson,
                        ]

                        echo "Trigger BuildDockerImageSanityTest job, params: ${parameters}"

                        def status = ""
                        def jobName = "/LLM/helpers/BuildDockerImageSanityTest"
                        def handle = build(
                            job: jobName,
                            parameters: trtllm_utils.toBuildParameters(parameters),
                            propagate: false,
                        )
                        echo "Triggered job: ${handle.absoluteUrl}"
                        status = handle.result

                        if (status != "SUCCESS") {
                            error "Downstream job did not succeed"
                        }
                    }
                }
            }
        }
        stage("Register NGC Images for Security Checks") {
            when {
                expression {
                    return params.nspect_id && params.action == "push"
                }
            }
            steps {
                script {
                    container("python3") {
                        trtllm_utils.llmExecStepWithRetry(this, script: "pip3 install --upgrade pip")
                        trtllm_utils.llmExecStepWithRetry(this, script: "pip3 install --upgrade requests")
                        def nspect_commit = "0e46042381ae25cb7af2f1d45853dfd8e1d54e2d"
                        withCredentials([string(credentialsId: "TRTLLM_NSPECT_REPO", variable: "NSPECT_REPO")]) {
                            trtllm_utils.checkoutSource("${NSPECT_REPO}", nspect_commit, "nspect")
                        }
                        def nspect_env = params.nspect_env ? params.nspect_env : "prod"
                        def program_version_name = params.program_version_name ? params.program_version_name : "PostMerge"
                        def cmd = """./nspect/nspect.py \
                            --env ${nspect_env} \
                            --nspect_id ${params.nspect_id} \
                            --program_version_name '${program_version_name}' \
                            """
                        if (params.register_images) {
                            cmd += "--register "
                        }
                        if (params.osrb_ticket) {
                            cmd += "--osrb_ticket ${params.osrb_ticket} "
                        }
                        if (params.wait_success_seconds) {
                            cmd += "--check_launch_api "
                            cmd += "--wait_success ${params.wait_success_seconds} "
                        }
                        cmd += imageKeyToTag.values().join(" ")
                        withCredentials([usernamePassword(credentialsId: "NSPECT_CLIENT-${nspect_env}", usernameVariable: 'NSPECT_CLIENT_ID', passwordVariable: 'NSPECT_CLIENT_SECRET')]) {
                            trtllm_utils.llmExecStepWithRetry(this, script: cmd, numRetries: 6, shortCommondRunTimeMax: 7200)
                        }
                    }
                }
            }
        }
    } // stages
} // pipeline
