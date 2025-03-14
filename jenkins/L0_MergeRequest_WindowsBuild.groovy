@Library('trtllm-jenkins-shared-lib@main') _
import groovy.transform.Field

// LLM repository configuration
LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl: 'https://gitlab-master.nvidia.com/ftp/tekit.git'
LLM_BRANCH = env.gitlabCommit? env.gitlabCommit: env.gitlabBranch
LLM_ROOT = 'llm'

BUILD_CORES_REQUEST = "8"
BUILD_CORES_LIMIT = "8"
BUILD_MEMORY_REQUEST = "48Gi"
BUILD_MEMORY_LIMIT = "48Gi"
BUILD_JOBS = "4"

@Field
String BUILD_TYPE = 'Release'

// Literals for easier access.
@Field
String CMAKEVARS = 'cmakeVars'
@Field
String TARNAME = 'tarName'

@Field
String SINGLE_DEVICE_CONFIG = 'SingleDevice'

// Build, test need to have same parameters.
@Field
def BUILD_CONFIGS = [
    (SINGLE_DEVICE_CONFIG) : [(CMAKEVARS) : 'ENABLE_MULTI_DEVICE=0', (TARNAME) : 'single-device-TensorRT-LLM-Windows.zip']
]

// Utilities
def checkoutSource(String repo, String branch, String directory) {
    def extensionsList = [
        // lfs() is unsupported on https://blossom.nvidia.com/sw-tensorrt-jenkins.
        // lfs(),
        [
            $class: 'CleanCheckout'
        ],
        [
            $class: 'CloneOption',
            shallow: true,
            depth: 500,
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
        $class: 'GitSCM',
        doGenerateSubmoduleConfigurations: false,
        submoduleCfg: [],
        branches: [[name: branch]],
        userRemoteConfigs: [
            [
                credentialsId: 'svc_tensorrt_gitlab_api_token',
                name: 'origin',
                refspec: "${branch}:refs/remotes/origin/${branch}",
                url: repo
            ]
        ],
        extensions: extensionsList
    ]

    echo "Cloning with SCM spec: ${scmSpec.toString()}"
    checkout(scm: scmSpec, changelog: true)
}

// Formal documentation is not really available for the pod spec
// structure used here.  Originally constructed using an example
// in use in an active pipeline.
//
// Some further exploring showed this example to be similar:
// <https://gitlab-master.nvidia.com/ipp/cloud-infra/blossom/blossom-examples/-/blob/master/Blossom-Hello-Window-Docker.groovy?ref_type=heads>
def createWindowsKubernetesPodConfig() {
    def jnlpConfig = """
                - name: jnlp
                  image: urm.nvidia.com/sw-ipp-blossom-sre-docker-local/jenkins/jnlp-agent:latest-windows
                  env:
                  - name: JENKINS_AGENT_WORKDIR
                    value: C:/Jenkins/agent
                  - name: DOCKER_HOST
                    value: "win-docker-proxy.blossom-system.svc.cluster.local"
                  resources:
                    requests:
                      cpu: ${BUILD_CORES_REQUEST}
                      memory: ${BUILD_MEMORY_REQUEST}
                      ephemeral-storage: 200Gi
                    limits:
                      cpu: ${BUILD_CORES_LIMIT}
                      memory: ${BUILD_MEMORY_LIMIT}
                      ephemeral-storage: 200Gi"""

    def podConfig = [
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
              nodeSelector:
                beta.kubernetes.io/os: windows
              containers:
                ${jnlpConfig}
        """.stripIndent(),
    ]

    return podConfig
}

void dockerLogin() {
    withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        powershell "docker login urm.nvidia.com -u ${USERNAME} -p ${PASSWORD}"
    }
}

def getWindowsYear() {
    def productName = powershell (script: "(Get-ItemProperty -Path 'HKLM:\\SOFTWARE\\Microsoft\\Windows NT\\CurrentVersion' -Name 'ProductName').ProductName", returnStdout: true).trim()
    echo "productName: ${productName}"

    // REVIEW: Regular expression extraction would be better but was failing.

    if (productName.contains('2022')) {
        return '2022'
    }

    return '2019'
}

def getArchName(String archList) {
    return archs.replace(";", "_") // "86-real;89-real" -> "86-real_89-real"
}

def getImageLocation() {
    return "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm"
}

// This method is used for extracting 1 or more fields from a version string.
// Examples:
//  getFields(1, '10.3.0.26') = 10
//  getFields(2, '10.3.0.26') = 10.3 ...
def getFields(int fields, String version) {
    def parts = version.split('\\.')
    return parts[0..(fields-1)].join('.')
}

def getCUDAToolkitDrvVersion(int fields) {
    return getFields(fields, '571.96')
}

def getCUDAToolkitVersion(int fields) {
    return getFields(fields, '12.8.0')
}

def getPythonVersion(int fields) {
    return getFields(fields, '3.10.11')
}

def getTRTCTVersion(int fields) {
    return getFields(fields, '12.8')
}

def getTRTVersion(int fields) {
    return getFields(fields, '10.8.0.43')
}

def getInputTag(String windowsYear, String inputType) {
    return "windows-${windowsYear}-${inputType}-input-py310-cuda${getCUDAToolkitVersion(3)}-trt${getTRTVersion(4)}"
}

def getBuildOutputTag(String windowsYear) {
    return "windows-${windowsYear}-build-output-py310-cuda${getCUDAToolkitVersion(3)}-trt${getTRTVersion(4)}"
}

def getInputName(String windowsYear, String inputType) {
    def inputLocation = getImageLocation()
    def inputTag = getInputTag(windowsYear, inputType)
    return "${inputLocation}:${inputTag}"
}

def shouldPushLastImage(String publishLastKnownGood) {
  return publishLastKnownGood=="TRUE";
}

def shouldDisableIncrementalBuild(String disableIncrementalBuild) {
  return disableIncrementalBuild=="TRUE";
}

def dockerCreateImageBuildInput(String windowsYear) {
    echo 'Create image: build-input'

    // Ensure login (for push) before spending time on build.
    dockerLogin()

    def hostWindowsYear = getWindowsYear()

    def baseImage = "mcr.microsoft.com/windows/servercore:ltsc${windowsYear}"
    def dockerfilePath = "${LLM_ROOT}\\jenkins\\windows\\Dockerfile.build-input"
    def isolation = (windowsYear != hostWindowsYear) ? 'hyperv' : 'process'
    def targetImage = getInputName(windowsYear, 'build')
    echo "baseImage: ${baseImage}"
    echo "dockerfilePath: ${dockerfilePath}"
    echo "isolation: ${isolation}"
    echo "targetImage: ${targetImage}"

    bat "docker pull ${baseImage}"

    def buildOptions = [
        "--isolation ${isolation}",
        "--build-arg BASE_IMAGE=${baseImage}",
        "--build-arg CUDA_TOOLKIT_DRV_VERSION_2=${getCUDAToolkitDrvVersion(2)}",
        "--build-arg CUDA_TOOLKIT_VERSION_2=${getCUDAToolkitVersion(2)}",
        "--build-arg CUDA_TOOLKIT_VERSION_3=${getCUDAToolkitVersion(3)}",
        "--build-arg PYTHON_VERSION_3=${getPythonVersion(3)}",
        "--build-arg TRT_CT_VERSION_2=${getTRTCTVersion(2)}",
        "--build-arg TRT_VERSION_3=${getTRTVersion(3)}",
        "--build-arg TRT_VERSION_4=${getTRTVersion(4)}",
        "--tag=${targetImage}",
        "-m 32g"
    ]
    bat "docker build ${buildOptions.join(' ')} - < ${dockerfilePath}"

    bat "docker push ${targetImage}"
    return targetImage
}

def dockerPullImageBuildInput(String windowsYear) {
    echo 'Pull image: build-input'

    def targetImage = getInputName(windowsYear, 'build')
    echo "targetImage: ${targetImage}"

    bat "docker pull ${targetImage}"
    return targetImage
}

def dockerRequireImageBuildInput(String windowsYear) {
    def image = null

    def buildInputLocationArtifactoryUrl = getImageLocation().replace("urm.nvidia.com", "urm.nvidia.com/artifactory")
    echo "buildInputLocationArtifactoryUrl: ${buildInputLocationArtifactoryUrl}"

    def buildInputTag = getInputTag(windowsYear, 'build')
    echo "buildInputTag: ${buildInputTag}"

    def buildInputExists = powershell (script: "(wget -UseBasicParsing 'https://${buildInputLocationArtifactoryUrl}').Content.contains('${buildInputTag}')", returnStdout: true).trim()
    echo "buildInputExists: ${buildInputExists}"
    if (buildInputExists == "True") {
        image = dockerPullImageBuildInput(windowsYear)
    } else {
        image = dockerCreateImageBuildInput(windowsYear)
    }

    return image
}

def dockerCreateImageBuildOutput(String windowsYear, String hostJobName, String hostBuildNumber, String targetBranch, String archs, String cmakeVars, String publishLastKnownGood, String disableIncrementalBuild) {
    echo 'Create image: build-output'

    def archName = getArchName(archs)

    def isMergeRequest = hostJobName.contains('/L0_MergeRequest')
    def isPostMerge = hostJobName.contains('/L0_PostMerge')
    def allowIncrementalBuild = ((isMergeRequest || isPostMerge) && !shouldDisableIncrementalBuild(disableIncrementalBuild))

    // Ensure login (for push) before spending time on build.
    dockerLogin()

    def buildOutputTag = getBuildOutputTag(windowsYear)

    def baseImage = getInputName(windowsYear, 'build')
    def lastTargetImage = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm/${targetBranch}/${archName}/last-known-good:${buildOutputTag}".toLowerCase()

    def cachedBaseImage = null
    if (allowIncrementalBuild) {
        cachedBaseImage = lastTargetImage;
        if (cachedBaseImage != null) {
            try {
                bat "docker pull ${cachedBaseImage}"
                baseImage = cachedBaseImage
            } catch (Exception ex) {
                echo "Exception pulling ${cachedBaseImage}: ${ex.message}"
                cachedBaseImage = null
            } finally {
            }
        }
    }

    def dockerfilePath = "${LLM_ROOT}\\jenkins\\windows\\Dockerfile.build-output"
    def isolation = 'process'
    def targetImage = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm/${hostJobName}/${hostBuildNumber}/${archName}:${buildOutputTag}".toLowerCase()
    echo "baseImage: ${baseImage}"
    echo "dockerfilePath: ${dockerfilePath}"
    echo "isolation: ${isolation}"
    echo "targetImage: ${targetImage}"

    if (cachedBaseImage == null) {
        bat "docker pull ${baseImage}"
    }

    def buildOptions = [
        "--isolation ${isolation}",
        "--build-arg ARCHS=${archs}",
        "--build-arg BASE_IMAGE=${getInputName(windowsYear, 'build')}",
        "--build-arg BUILD_IMAGE=${baseImage}",
        "--build-arg CMAKEVARS=${cmakeVars}",
        "--build-arg JOB_COUNT=${BUILD_JOBS}",
        "--build-arg TRT_VERSION_4=${getTRTVersion(4)}",
        "--tag=${targetImage}",
        "-m 32g"
    ]
    bat "docker build ${buildOptions.join(' ')} -f ${dockerfilePath} .\\${LLM_ROOT}"

    if (shouldPushLastImage(publishLastKnownGood)) {
        // Also push under the name 'last' for easy pick-up by next jobs
        // for incremental builds.
        bat "docker tag ${targetImage} ${lastTargetImage}"
    }

    // REVIEW: During one pipeline testing run, the first push
    // succeeded while the second failed.  Suspected a possible
    // expiration on the docker login.  So login again to give
    // a bit more resilience.
    dockerLogin()
    bat "docker push ${targetImage}"
    if (shouldPushLastImage(publishLastKnownGood)) {
        bat "docker push ${lastTargetImage}"
    }

    return targetImage
}

def dockerCreateImageArtifacts(String windowsYear, String buildOutputImage, String archs, String tarName) {
    def dockerfilePath = "${LLM_ROOT}\\jenkins\\windows\\Dockerfile.collect-artifacts"

    def hostWindowsYear = getWindowsYear()
    def baseImage = "mcr.microsoft.com/windows/servercore:ltsc${windowsYear}"
    def isolation = (windowsYear != hostWindowsYear) ? 'hyperv' : 'process'

    def targetArch = "x86_64-windows-msvc"
    def archName = getArchName(archs)
    def targetImage = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm/${hostJobName}/${hostBuildNumber}/${archName}:collect".toLowerCase()

    echo "baseImage: ${baseImage}"
    echo "buildOutputImage: ${buildOutputImage}"
    echo "dockerfilePath: ${dockerfilePath}"
    echo "isolation: ${isolation}"
    echo "targetArch: ${targetArch}"
    echo "targetImage: ${targetImage}"

    bat "docker build --isolation ${isolation} -m 8g --build-arg BASE_IMAGE=${baseImage} --build-arg BUILD_IMAGE=${buildOutputImage} --build-arg TARGET_ARCH=${targetArch} --build-arg TAR_NAME=${tarName} --tag ${targetImage} - < ${dockerfilePath}"

    return targetImage
}

def dockerCreateImageTestInput(String windowsYear) {
    echo 'Create image: test-input'

    // Ensure login (for push) before spending time on build.
    dockerLogin()

    def hostWindowsYear = getWindowsYear()

    def baseImage = "mcr.microsoft.com/windows/servercore:ltsc${windowsYear}"
    def dockerfilePath = "${LLM_ROOT}\\jenkins\\windows\\Dockerfile.test-input"
    def isolation = (windowsYear != hostWindowsYear) ? 'hyperv' : 'process'
    def targetImage = getInputName(windowsYear, "test")
    echo "baseImage: ${baseImage}"
    echo "dockerfilePath: ${dockerfilePath}"
    echo "isolation: ${isolation}"
    echo "targetImage: ${targetImage}"

    def buildOptions = [
        "--isolation ${isolation}",
        "--build-arg BASE_IMAGE=${baseImage}",
        "--build-arg CUDA_TOOLKIT_DRV_VERSION_2=${getCUDAToolkitDrvVersion(2)}",
        "--build-arg CUDA_TOOLKIT_VERSION_2=${getCUDAToolkitVersion(2)}",
        "--build-arg CUDA_TOOLKIT_VERSION_3=${getCUDAToolkitVersion(3)}",
        "--build-arg PYTHON_VERSION_3=${getPythonVersion(3)}",
        "--build-arg TRT_CT_VERSION_2=${getTRTCTVersion(2)}",
        "--build-arg TRT_VERSION_3=${getTRTVersion(3)}",
        "--build-arg TRT_VERSION_4=${getTRTVersion(4)}",
        "--tag=${targetImage}",
        "-m 32g"
    ]
    def buildCmd = "docker build ${buildOptions.join(' ')} - < ${dockerfilePath}"
    echo "buildCmd: ${buildCmd}"
    def pushCmd = "docker push ${targetImage}"
    echo "pushCmd: ${pushCmd}"

    bat "docker pull ${baseImage}"
    bat "${buildCmd}"
    bat "${pushCmd}"

    return targetImage
}

def dockerEnsureImageTestInput(String windowsYear) {
    def image = null

    def testInputLocationArtifactoryUrl = getImageLocation().replace("urm.nvidia.com", "urm.nvidia.com/artifactory")
    echo "testInputLocationArtifactoryUrl: ${testInputLocationArtifactoryUrl}"

    def testInputTag = getInputTag(windowsYear, "test")
    echo "testInputTag: ${testInputTag}"

    def testInputExists = powershell (script: "(wget -UseBasicParsing 'https://${testInputLocationArtifactoryUrl}').Content.contains('${testInputTag}')", returnStdout: true).trim()
    echo "testInputExists: ${testInputExists}"
    if (testInputExists == "True") {
        image = getInputName(windowsYear, "test")
    } else {
        image = dockerCreateImageTestInput(windowsYear)
    }

    return image
}

void runLLMBuild(String hostJobName, String hostBuildNumber, String targetBranch, String archs, String publishLastKnownGood, String disableIncrementalBuild) {
    BUILD_CONFIGS.each { config, buildFlags ->
        container("jnlp") {
            script {
                bat "docker images"

                checkoutSource(LLM_REPO, LLM_BRANCH, LLM_ROOT)

                def imageTestInput = dockerEnsureImageTestInput('2022')
                echo "dockerEnsureImageTestInput() => ${imageTestInput}"

                def imageBuildInput = dockerRequireImageBuildInput(getWindowsYear())
                echo "dockerRequireImageBuildInput() => ${imageBuildInput}"

                def imageBuildOutput = dockerCreateImageBuildOutput(getWindowsYear(), hostJobName, hostBuildNumber, targetBranch, archs, buildFlags[CMAKEVARS], publishLastKnownGood, disableIncrementalBuild)
                echo "dockerCreateImageBuildOutput() => ${imageBuildOutput}"

                def imageArtifacts = dockerCreateImageArtifacts(getWindowsYear(), imageBuildOutput, archs, buildFlags[TARNAME])
                echo "dockerCreateImageArtifacts() => ${imageArtifacts}"

                // The node does not allow (403/Forbidden) a number of docker
                // commands, but it does allow "docker image save".  So as an
                // alternative to create+cp, the following is done:
                // 1. Use "docker image save" to store to a tar archive
                // 2. Use "tar -xf" to extract the manifest.json
                // 3. Load manifest.json, parse as json, and
                //    identify final layer relative path
                // 4. Use "tar -xf" to extract the final layer's tar
                // 5. Use "tar -xf" to extract workspace\Package, but exclude
                //    the Staging subfolder within it
                bat "docker image save ${imageArtifacts} -o local.collect.tar"
                bat "dir local.collect.tar"

                bat 'tar -xf local.collect.tar manifest.json'
                powershell '$manifest = (Get-Content manifest.json | ConvertFrom-Json); $lastLayer = $manifest[0].Layers[-1]; tar -xf local.collect.tar $lastLayer; tar -xf $lastLayer --exclude Files/workspace/Package/Staging/ Files/workspace/Package/*;'
                bat "tree /f Files"

                def archName = getArchName(archs)
                def targetArch = "x86_64-windows-msvc"

                def artifacts = [
                    buildFlags[TARNAME],
                    "tensorrt_llm_batch_manager_static.lib",
                    "tensorrt_llm_executor_static.lib",
                    "tensorrt_llm_internal_cutlass_kernels_static.lib",
                    "tensorrt_llm_nvrtc_wrapper.dll",
                    "tensorrt_llm_nvrtc_wrapper.lib"
                ]
                def deployURL = "sw-tensorrt-generic/llm-artifacts/${hostJobName}/${hostBuildNumber}/${targetArch}/${archName}/"
                for (artifact in artifacts) {
                    trtllm_utils.uploadArtifacts("Files\\workspace\\Package\\${artifact}", deployURL)
                }

                // Save space on host and mitigate risk of error when creating a new layer:
                //  re-exec error: exit status 1: output: hcsshim::ImportLayer - failed failed in Win32: An attempt
                //  was made to create more links on a file than the file system supports. (0x476)
                bat "docker image rm -f ${imageArtifacts}"
                bat "docker image rm -f ${imageBuildOutput}"
            }
        }
    }
}

pipeline {
    agent none

    options {
        skipDefaultCheckout()
        timestamps()
        timeout(time: 24, unit: 'HOURS')
    }

    environment {
        PIP_INDEX_URL="https://urm-rn.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
    }

    stages {
        stage('Prepare') {
            steps {
                echo "hostJobName: ${env.hostJobName}"
                echo "hostBuildNumber: ${env.hostBuildNumber}"
                echo "gitlabSourceRepoHttpUrl: ${env.gitlabSourceRepoHttpUrl}"
                echo "gitlabBranch: ${env.gitlabBranch}"
                echo "gitlabCommit: ${env.gitlabCommit}"
                echo "gitlabTargetBranch: ${env.gitlabTargetBranch}"
                echo "archs: ${env.archs}"
                echo "publishLastKnownGood: ${env.publishLastKnownGood}"
                echo "disableIncrementalBuild: ${env.disableIncrementalBuild}"
            }
        }
        stage('Build') {
            agent {
                kubernetes createWindowsKubernetesPodConfig()
            }
            steps {
                runLLMBuild(env.hostJobName, env.hostBuildNumber, env.gitlabTargetBranch, env.archs, env.publishLastKnownGood, env.disableIncrementalBuild)
            }
        }
    }
}
