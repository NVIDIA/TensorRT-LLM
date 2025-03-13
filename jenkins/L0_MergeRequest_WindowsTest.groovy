/* groovylint-disable NoDef, VariableTypeRequired */
import groovy.transform.Field

// LLM repository configuration
LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl: "https://gitlab-master.nvidia.com/ftp/tekit.git"
LLM_BRANCH = env.gitlabCommit? env.gitlabCommit: env.gitlabBranch
LLM_ROOT = 'llm'

// TURTLE repository configuration
TURTLE_REPO = 'https://gitlab-master.nvidia.com/TensorRT/Infrastructure/turtle.git'
TURTLE_BRANCH = "v6.4.7"
TURTLE_ROOT = 'turtle'

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

@Field
String TEST_PYTHON_VERSION = '3.10.4'

// https://learn.microsoft.com/en-us/windows-hardware/drivers/install/guid-devinterface-display-adapter
// The GUID_DEVINTERFACE_DISPLAY_ADAPTER device interface class is
// defined for display views that are supported by display adapters.
@Field
String GUID_DEVINTERFACE_DISPLAY_ADAPTER = '5B45201D-F2F2-4F3B-85BB-30FF1F953599'

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
            disableSubmodules: true,
            parentCredentials: true,
            recursiveSubmodules: false,
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

// This method is used for extracting 1 or more fields from a version string.
// Examples:
//  getFields(1, '10.3.0.26') = 10
//  getFields(2, '10.3.0.26') = 10.3 ...
def getFields(int fields, String version) {
    def parts = version.split('\\.')
    return parts[0..(fields-1)].join('.')
}

def getCUDAToolkitVersion(int fields) {
    return getFields(fields, '12.8.0')
}

def getTRTVersion(int fields) {
    return getFields(fields, '10.8.0.43')
}

def getImageLocation() {
    return "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm"
}

def getInputTag(String windowsYear, String inputType) {
    return "windows-${windowsYear}-${inputType}-input-py310-cuda${getCUDAToolkitVersion(3)}-trt${getTRTVersion(4)}"
}

def getInputName(String windowsYear, String inputType) {
    def inputLocation = getImageLocation()
    def inputTag = getInputTag(windowsYear, inputType)
    return "${inputLocation}:${inputTag}"
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

void uploadArtifacts(String patternAbsPath, String target) {
    // Step 3: upload test results and logs to artifactory
    credentials = [
                    usernamePassword(credentialsId: 'urm-artifactory-creds',
                    usernameVariable: 'SVC_TENSORRT_USER',
                    passwordVariable: 'ARTIFACTORY_PASS'),
    ]
    withCredentials(credentials)
    {
        String serverId =  'Artifactory'
        rtServer(
            id: "$serverId",
            url: 'https://urm.nvidia.com/artifactory',
            // If you're using username and password:
            username: 'svc_tensorrt',
            password: "$ARTIFACTORY_PASS",
            // If Jenkins is configured to use an http proxy, you can bypass the proxy when using this Artifactory server:
            bypassProxy: true,
            // Configure the connection timeout (in seconds).
            // The default value (if not configured) is 300 seconds:
            timeout: 300
        )
        rtUpload(
            serverId: "$serverId",
            spec: """{
                "files": [
                    {
                    "pattern": "${patternAbsPath}",
                    "target": "${target}"
                    }
                ]
            }""",
        )
    }
}

def runLLMTest(String hostJobName, String hostBuildNumber, String gpu, String testList = "", String config = SINGLE_DEVICE_CONFIG) {
    def imageTestInput = getInputName(getWindowsYear(), 'test')
    echo "imageTestInput: ${imageTestInput}"

    def archName = env.platform.replace(";", "_") // "86-real;89-real" -> "86-real_89-real"
    echo "archName: ${archName}"

    def zipName = "${BUILD_CONFIGS[config][TARNAME]}"
    echo "zipName: ${zipName}"

    def zipUrl = "https://urm.nvidia.com/artifactory/sw-tensorrt-generic/llm-artifacts/${hostJobName}/${hostBuildNumber}/x86_64-windows-msvc/${archName}/${zipName}"
    echo "zipUrl: ${zipUrl}"

    bat "docker images"

    bat "docker pull ${imageTestInput}"

    def llmAbsolutePath = powershell (script: "(Resolve-Path ${LLM_ROOT}).Path", returnStdout: true).trim()
    echo "llmAbsolutePath: ${llmAbsolutePath}"

    bat "dir ${llmAbsolutePath}"

    def testRunParams = [
        '-config ' + config,
        '-llmSrc C:\\tekit',
        '-gpu ' + gpu,
        '-testList ' + testList,
        '-trtVersion4 10.8.0.43'
    ]
    def testRunCommandLine = 'C:\\tekit\\jenkins\\windows\\test.run_list.ps1 ' + testRunParams.join(' ')
    echo "testRunCommandLine: ${testRunCommandLine}"

    def testContainerPowershellRunCommands = [
        // Copies loader files from host to container.
        // Note: Not required on newer Windows 11 builds.
        'C:\\tekit\\jenkins\\windows\\test.install_gpu_support.ps1',

        'Get-Date -Format \'u\'',

        // Download the zip
        "Write-Host Download ${zipUrl}",
        '$ProgressPreference = \'SilentlyContinue\'',
        "Invoke-WebRequest -Uri ${zipUrl} -OutFile ${zipName}",

        'Get-Date -Format \'u\'',

        // Expand the zip
        'Write-Host Extract whl',
        'Add-Type -AssemblyName System.IO.Compression.FileSystem',
        '$zipFile = [System.IO.Compression.ZipFile]::OpenRead(\'single-device-TensorRT-LLM-Windows.zip\')',
        '$zipFile.Entries | Where-Object { $_.Name -like \'*.whl\' } | ForEach-Object { [System.IO.Compression.ZipFileExtensions]::ExtractToFile($_, $_.Name) }',

        'Get-Date -Format \'u\'',

        // Install the whl
        '$whlName = @(Get-Item *.whl)[0].Name',
        'pip install ${whlName} --extra-index-url https://download.pytorch.org/whl/',

        'Get-Date -Format \'u\'',

        // Invoke turtle
        testRunCommandLine
    ]
    def testDockerRunCommand = 'powershell "' + testContainerPowershellRunCommands.join('; ') + '"'
    echo "testDockerRunCommand: ${testDockerRunCommand}"

    bat "docker run --rm --isolation process --device class/${GUID_DEVINTERFACE_DISPLAY_ADAPTER} -v C:\\Windows:C:\\HostWindows -v ${llmAbsolutePath}:C:\\tekit ${imageTestInput} ${testDockerRunCommand}"
}

pipeline {
    agent {
        // https://nvbugs/4668795 - Intermittent failures in the L0 Windows test pipeline.
        label 'win10native-geforce-rtx-4090 && WindowsServer2022 && !x570-0039'
    }
    environment
    {
        //Workspace normally is: C:\Jenkins_Agent\workspace\L0_MergeRequest-<hash>@tmp/
        HF_HOME="${env.WORKSPACE_TMP}\\.cache\\huggingface"
        PIP_INDEX_URL="https://urm-rn.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
    }
    stages {
        stage('Prepare') {
            steps {
                powershell 'Get-Location'
                powershell 'gci -Force'
                // By default, the code is checked out in workspace, remove this and postpone this to the build stage.
                powershell 'Remove-Item -Recurse -Force .\\*'
                powershell 'gci -Force'
                echo "hostJobName: ${env.hostJobName}"
                echo "hostBuildNumber: ${env.hostBuildNumber}"
                echo "gitlabSourceRepoHttpUrl: ${env.gitlabSourceRepoHttpUrl}"
                echo "gitlabBranch: ${env.gitlabBranch}"
                echo "gitlabCommit: ${env.gitlabCommit}"
                echo "gitlabTargetBranch: ${env.gitlabTargetBranch}"
                echo "stage: ${env.stage}"
                echo "platform: ${env.platform}"
                echo "testList: ${env.testList}"
            }
        }
        stage('Test') {
            when {
                expression { env.stage == 'Test' }
            }
            steps {
                script {
                    powershell "Write-Host ([System.Environment]::OSVersion.Version)"

                    checkoutSource(LLM_REPO, LLM_BRANCH, LLM_ROOT)

                    runLLMTest(env.hostJobName, env.hostBuildNumber, "RTX4090", env.testList)
                }
            }
        }
    }
}
