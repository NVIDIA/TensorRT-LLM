@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.InterruptedException
import groovy.transform.Field
import groovy.json.JsonSlurper
import groovy.json.JsonOutput
import com.nvidia.bloom.KubernetesManager
import com.nvidia.bloom.Constants
import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.SlurmCluster
import com.nvidia.bloom.SlurmPartition
import com.nvidia.bloom.Utils
import com.nvidia.bloom.ContainerRuntime
import org.jenkinsci.plugins.workflow.cps.CpsThread
import org.jsoup.Jsoup
import org.jenkinsci.plugins.pipeline.modeldefinition.Utils as jUtils

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}
LLM_ROOT = "llm"

ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
UPLOAD_PATH = env.uploadPath ? env.uploadPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"

X86_64_TRIPLE = "x86_64-linux-gnu"
AARCH64_TRIPLE = "aarch64-linux-gnu"

// default package name
linuxPkgName = ( env.targetArch == AARCH64_TRIPLE ? "tensorrt-llm-sbsa-release-src-" : "tensorrt-llm-release-src-" ) + (env.artifactCommit ? env.artifactCommit : env.gitlabCommit) + ".tar.gz"

// Container configuration
// available tags can be found in: https://urm.nvidia.com/artifactory/sw-tensorrt-docker/tensorrt-llm/
// [base_image_name]-[arch]-[os](-[python_version])-[trt_version]-[torch_install_type]-[stage]-[date]-[mr_id]
LLM_DOCKER_IMAGE = env.dockerImage
LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE = env.wheelDockerImagePy310
LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE = env.wheelDockerImagePy312

// DLFW torch image
DLFW_IMAGE = "urm.nvidia.com/docker/nvidia/pytorch:25.12-py3"

//Ubuntu base image
UBUNTU_22_04_IMAGE = "urm.nvidia.com/docker/ubuntu:22.04"
UBUNTU_24_04_IMAGE = "urm.nvidia.com/docker/ubuntu:24.04"

POD_TIMEOUT_SECONDS_TEST = env.podTimeoutSeconds ? env.podTimeoutSeconds : "21600"
POD_TIMEOUT_SECONDS_BUILD = env.podTimeoutSeconds ? env.podTimeoutSeconds : "43200"
POD_TIMEOUT_SECONDS_SLURM = env.podTimeoutSeconds ? env.podTimeoutSeconds : "79200"  // Use 22 hours to allow for 2 hour of buffer.

// Literals for easier access.
@Field
def TARNAME = "tarName"

@Field
def VANILLA_CONFIG = "Vanilla"

@Field
def SINGLE_DEVICE_CONFIG = "SingleDevice"

@Field
def LLVM_CONFIG = "LLVM"

@Field
def LINUX_AARCH64_CONFIG = "linux_aarch64"

@Field
def BUILD_CONFIGS = [
  // Vanilla TARNAME is used for packaging in runLLMPackage
  (VANILLA_CONFIG) : [(TARNAME) : "TensorRT-LLM.tar.gz"],
  (SINGLE_DEVICE_CONFIG) : [(TARNAME) : "single-device-TensorRT-LLM.tar.gz"],
  (LLVM_CONFIG) : [(TARNAME) : "llvm-TensorRT-LLM.tar.gz"],
  (LINUX_AARCH64_CONFIG) : [(TARNAME) : "TensorRT-LLM-GH200.tar.gz"],
]

// TODO: Move common variables to an unified location
BUILD_CORES_REQUEST = "8"
BUILD_CORES_LIMIT = "8"
BUILD_MEMORY_REQUEST = "48Gi"
BUILD_MEMORY_LIMIT = "96Gi"
BUILD_JOBS = "4"

SLURM_CORES_REQUEST = "1"
SLURM_CORES_LIMIT = "1"
SLURM_MEMORY_REQUEST = "8Gi"
SLURM_MEMORY_LIMIT = "12Gi"

TESTER_CORES = "12"
TESTER_MEMORY = "96Gi"

CCACHE_DIR="/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"
MODEL_CACHE_DIR="/scratch.trt_llm_data/llm-models"

// GPU types that require open driver
REQUIRED_OPEN_DRIVER_TYPES = ["b100-ts2", "rtx-5080", "rtx-5090", "rtx-pro-6000", "rtx-pro-6000d"]

// GPU types that don't support dynamic driver flashing
REQUIRED_NO_DRIVER_TYPES = ["dgx-h100", "dgx-h200", "gh200", "gb10x"]

// ENABLE_NGC_DEVEL_IMAGE_TEST is currently disabled in the Jenkins BuildDockerImageSanityTest job config
ENABLE_NGC_DEVEL_IMAGE_TEST = params.enableNgcDevelImageTest ?: false
ENABLE_NGC_RELEASE_IMAGE_TEST = params.enableNgcReleaseImageTest ?: false

COMMON_SSH_OPTIONS = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o TCPKeepAlive=no -o ServerAliveInterval=30 -o ServerAliveCountMax=20"

def uploadResults(def pipeline, SlurmCluster cluster, String nodeName, String stageName){
    withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        def randomLoginNode = SlurmConfig.getRandomLoginNode(cluster.host)
        def remote = [
            ip           : randomLoginNode,
            host         : randomLoginNode,
            port         : cluster.sshPort?:22,
            user         : "${pipeline.USERNAME}",
            passwd       : "${pipeline.PASSWORD}",
            allowAnyHosts: true,
        ]

        Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")

        def hasTimeoutTest = false
        def downloadResultSucceed = false
        def downloadPerfResultSucceed = false

        pipeline.stage('Submit Test Result') {
            sh "mkdir -p ${stageName}"
            // Download timeout test results
            def timeoutTestFilePath = "/home/svc_tensorrt/bloom/scripts/${nodeName}/unfinished_test.txt"
            def downloadTimeoutTestSucceed = Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' scp -P ${remote.port} -r -p ${COMMON_SSH_OPTIONS} ${remote.user}@${remote.host}:${timeoutTestFilePath} ${stageName}/", returnStatus: true, numRetries: 3) == 0
            if (downloadTimeoutTestSucceed) {
                sh "ls ${stageName}"
                def timeoutTestXml = generateTimeoutTestResultXml(stageName, "unfinished_test.txt")
                if (timeoutTestXml != null) {
                    sh """
cat > ${stageName}/results-timeout.xml << 'EOF_TIMEOUT_XML'
${timeoutTestXml}
EOF_TIMEOUT_XML
                    """
                    hasTimeoutTest = true
                }
            }
            // Download normal test results
            def resultsFilePath = "/home/svc_tensorrt/bloom/scripts/${nodeName}/results.xml"
            downloadResultSucceed = Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' scp -P ${remote.port} -r -p ${COMMON_SSH_OPTIONS} ${remote.user}@${remote.host}:${resultsFilePath} ${stageName}/", returnStatus: true, numRetries: 3) == 0

            // Download perf test results
            def perfResultsBasePath = "/home/svc_tensorrt/bloom/scripts/${nodeName}"
            def folderListOutput = Utils.exec(
                pipeline,
                script: Utils.sshUserCmd(
                    remote,
                    "\"find '${perfResultsBasePath}' -maxdepth 1 -type d \\( -name 'aggr*' -o -name 'disagg*' \\) -printf '%f\\n' || true\""
                ),
                returnStdout: true,
                numRetries: 3
            )?.trim() ?: ""
            def perfFolders = folderListOutput.split(/\s+/).collect { it.trim().replaceAll(/\/$/, '') }.findAll { it }
            echo "Perf Result Folders: ${perfFolders}"
            if (perfFolders) {
                def scpSources = perfFolders.size() == 1
                    ? "${remote.user}@${remote.host}:${perfResultsBasePath}/${perfFolders[0]}"
                    : "${remote.user}@${remote.host}:{${perfFolders.collect { "${perfResultsBasePath}/${it}" }.join(',')}}"
                downloadPerfResultSucceed = Utils.exec(pipeline, script: "sshpass -p '${remote.passwd}' scp -P ${remote.port} -r -p ${COMMON_SSH_OPTIONS} ${scpSources} ${stageName}/", returnStatus: true, numRetries: 3) == 0
            }

            echo "hasTimeoutTest: ${hasTimeoutTest}, downloadResultSucceed: ${downloadResultSucceed}, downloadPerfResultSucceed: ${downloadPerfResultSucceed}"
            if (hasTimeoutTest || downloadResultSucceed || downloadPerfResultSucceed) {
                sh "ls ${stageName}"
                echo "Upload test results."
                sh "tar -czvf results-${stageName}.tar.gz ${stageName}/"
                ensureStageResultNotUploaded(stageName)
                trtllm_utils.uploadArtifacts(
                    "results-${stageName}.tar.gz",
                    "${UPLOAD_PATH}/test-results/"
                )
            } else {
                println("No results xml to submit")
            }
        }

        if (hasTimeoutTest || downloadResultSucceed) {
            junit(allowEmptyResults: true, testResults: "${stageName}/results*.xml")
        }
    }
}

def runIsolatedTests(preprocessedLists, testCmdLine, llmSrc, stageName) {
    // Run the isolated tests one by one to avoid any potential conflicts
    def isolateTestList = preprocessedLists.isolate
    def isolateTestLines = readFile(file: isolateTestList).readLines()
    def rerunFailed = false

    for (int i = 0; i < isolateTestLines.size(); i++) {
        def isolateTestName = isolateTestLines[i].trim()
        // Create a temporary file for this single isolated test
        def singleTestFile = "${isolateTestList}_isolated_${i}.txt"
        sh "echo '${isolateTestName}' > ${singleTestFile}"
        sh "cat ${singleTestFile}"

        def isolateTestCmdLine = testCmdLine.findAll { cmd ->
            !cmd.contains("--test-list=") &&
            !cmd.contains("--test-prefix=") &&
            !cmd.contains("--csv=") &&
            !cmd.contains("--periodic-junit-xmlpath")
        }
        isolateTestCmdLine += ["--test-list=${singleTestFile}"]
        isolateTestCmdLine += ["--test-prefix=${stageName}"]
        isolateTestCmdLine += ["--csv=${WORKSPACE}/${stageName}/report_isolated_${i}.csv"]
        isolateTestCmdLine += ["--periodic-junit-xmlpath ${WORKSPACE}/${stageName}/results_isolated_${i}.xml"]
        isolateTestCmdLine += ["--cov-append"]  // Append coverage data to avoid overwriting previous data

        try {
            sh """
                cd ${llmSrc}/tests/integration/defs && \
                ${isolateTestCmdLine.join(" ")}
            """
        } catch (InterruptedException e) {
            throw e
        } catch (Exception e) {
            def isRerunFailed = rerunFailedTests(stageName, llmSrc, isolateTestCmdLine, "results_isolated_${i}.xml", "isolated_${i}")
            if (isRerunFailed) {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    error "Isolated test ${i} (${isolateTestName}) failed after rerun attempt"
                }
                // Mark that at least one isolated test failed, but continue processing other tests
                rerunFailed = true
            }
        } finally {
            // Clean up the temporary test file
            sh "rm -f ${singleTestFile}"
        }
    }

    // After processing all isolated tests, set stage failure if any test failed
    if (rerunFailed) {
        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
            error "One or more isolated tests failed after rerun attempts"
        }
    }

    return rerunFailed  // Return the updated value
}

def processShardTestList(llmSrc, testDBList, splitId, splits, perfMode=false) {
    // Preprocess testDBList to extract ISOLATION markers
    echo "Preprocessing testDBList to extract ISOLATION markers..."

    def originalTestLines = readFile(file: testDBList).readLines()
    def cleanedTestLines = []
    def isolationTestLines = []

    originalTestLines.each { originalLine ->
        def trimmedLine = originalLine.trim()
        if (trimmedLine && trimmedLine.contains('ISOLATION')) {
            // Remove ISOLATION marker and nearby comma from the line
            def cleanedLine = trimmedLine

            // Handle different comma patterns around ISOLATION
            if (trimmedLine.contains('ISOLATION,')) {
                // Case: "ISOLATION,OTHER_MARKER" -> remove "ISOLATION,"
                cleanedLine = cleanedLine.replace('ISOLATION,', '').trim()
            } else if (trimmedLine.contains(',ISOLATION')) {
                // Case: "OTHER_MARKER,ISOLATION" -> remove ",ISOLATION"
                cleanedLine = cleanedLine.replace(',ISOLATION', '').trim()
            } else {
                // Case: standalone "ISOLATION" -> remove " ISOLATION"
                cleanedLine = cleanedLine.replace(' ISOLATION', '').trim()
            }

            // Add the cleaned line to isolationTestLines if original line had ISOLATION
            isolationTestLines.add(cleanedLine)
            cleanedTestLines.add(cleanedLine)

        } else if (trimmedLine) {
            // Line doesn't contain ISOLATION, add as-is
            cleanedTestLines.add(originalLine.trim())
        }
    }

    // Create cleaned testDBList file (without ISOLATION markers)
    def cleanedTestDBList = testDBList.replaceAll('\\.txt$', '_cleaned.txt')
    if (cleanedTestLines.size() > 0) {
        def cleanedContent = cleanedTestLines.join('\n')
        sh "echo '${cleanedContent.replace("'", "'\\''")}' > ${cleanedTestDBList}"
        echo "Created cleaned testDBList: ${cleanedTestDBList} with ${cleanedTestLines.size()} lines (ISOLATION markers removed)"
    } else {
        sh "touch ${cleanedTestDBList}"
        echo "No tests found, created empty cleaned testDBList: ${cleanedTestDBList}"
    }

    sh "cat ${cleanedTestDBList}"
    echo "Original testDBList contains ${isolationTestLines.size()} tests that had ISOLATION markers"

    def shardTestList = []

    if (perfMode) {
        // In perfMode, skip pytest collection as it may cause errors with automatically generated testcases
        // Instead, use all tests from the original testDBList
        echo "Performance mode enabled - skipping pytest collection, using all tests from testDBList"
    } else {
        def testListCmd = [
            "LLM_ROOT=${llmSrc}",
            "LLM_BACKEND_ROOT=${llmSrc}/triton_backend",
            "pytest",
            "--collect-only",
            "--splitting-algorithm least_duration",
            "--test-list=${cleanedTestDBList}",
            "--quiet",
            "--splits ${splits}",
            "--group ${splitId}"
        ]

        try {
            // First execute the pytest command and check if it succeeds
            def pytestOutput = sh(
                script: "cd ${llmSrc}/tests/integration/defs && ${testListCmd.join(' ')}",
                returnStdout: true
            ).trim()

            // Debug: Show the raw pytest output
            echo "<<<START_PYTEST_OUTPUT>>>"
            echo "${pytestOutput}"
            echo "<<<END_PYTEST_OUTPUT>>>"

            // Filter the output to get only test lines with '::' that occur after "Running X items in this shard"
            def lines = pytestOutput.split('\n')
            def foundRunningLine = false
            def lineIndex = 0
            shardTestList = lines.findAll { line ->
                lineIndex++

                if (line.matches(/.*Running \d+ items in this shard.*/) || line.matches(/.*\[pytest-split\] Running group.*/)) {
                    foundRunningLine = true
                    return false  // Don't include the "Running" line itself
                }
                // Stop collecting when we hit the warnings/errors summary separator
                if (foundRunningLine && line.contains('======================')) {
                    foundRunningLine = false  // Stop collecting
                    return false
                }

                def hasDoubleColon = line.contains('::')
                def shouldInclude = foundRunningLine && hasDoubleColon
                return shouldInclude
            }
            echo "Filtering complete. shardTestList size: ${shardTestList.size()}"
        } catch (Exception e) {
            echo "Error: Failed to execute pytest command for test collection: ${e.getMessage()}"
            error "Test collection failed for shard ${splitId}/${splits}. Cannot proceed without valid test list."
        }
    }

    if (shardTestList || perfMode) {
        // Split the shard test list into regular and isolate tests
        def shardRegularTests = []
        def shardIsolateTests = []

        if (perfMode) {
            // In perfMode, put all tests in regular and skip isolation
            echo "Performance mode enabled - all tests will run as regular tests (no isolation)"
            shardRegularTests = cleanedTestLines.findAll { it.trim() }
        } else {
            // Process each test from shardTestList
            shardTestList.each { test ->
                def trimmedTest = test.trim()
                if (trimmedTest) {
                    // Process test_unittests.py::test_unittests_v2[xxxx] pattern
                    if (trimmedTest.startsWith('test_unittests.py::test_unittests_v2[') && trimmedTest.endsWith(']')) {
                        // Extract content between [ and ]
                        def startIndex = trimmedTest.indexOf('[') + 1
                        def endIndex = trimmedTest.lastIndexOf(']')
                        trimmedTest = trimmedTest.substring(startIndex, endIndex)
                    }

                    // Check if this test is in the isolation list
                    def isolationTestLine = isolationTestLines.find { it.contains(trimmedTest) }
                    if (isolationTestLine) {
                        // This test needs isolation
                        shardIsolateTests.add(isolationTestLine)
                    } else {
                        // This test is a regular test - find the actual line from cleanedTestLines
                        def cleanedTestLine = cleanedTestLines.find { it.contains(trimmedTest) }
                        shardRegularTests.add(cleanedTestLine)
                    }
                }
            }
        }

        // Define file paths for regular and isolate tests
        def regularTestList = testDBList.replaceAll('\\.txt$', '_regular.txt')
        def isolateTestList = testDBList.replaceAll('\\.txt$', '_isolate.txt')

        // Create shard-specific test files
        if (shardRegularTests.size() > 0) {
            def shardRegularContent = shardRegularTests.join('\n')
            sh "echo '${shardRegularContent.replace("'", "'\\''")}' > ${regularTestList}"
            echo "Created ${regularTestList} with ${shardRegularTests.size()} regular tests for this shard"
        } else {
            sh "touch ${regularTestList}"
            echo "No regular tests in this shard, created empty file: ${regularTestList}"
        }
        sh "cat ${regularTestList}"

        if (shardIsolateTests.size() > 0) {
            def shardIsolateContent = shardIsolateTests.join('\n')
            sh "echo '${shardIsolateContent.replace("'", "'\\''")}' > ${isolateTestList}"
            echo "Created ${isolateTestList} with ${shardIsolateTests.size()} isolate tests for this shard"
        } else {
            sh "touch ${isolateTestList}"
            echo "No isolate tests in this shard, created empty file: ${isolateTestList}"
        }
        sh "cat ${isolateTestList}"

        // Return preprocessed lists object for compatibility
        return [
            regular: regularTestList,
            isolate: isolateTestList,
            regularCount: shardRegularTests.size(),
            isolateCount: shardIsolateTests.size()
        ]
    } else {
        echo "No tests found in current shard or failed to list tests"
        // Create empty files and preprocessed lists object
        def regularTestList = testDBList.replaceAll('\\.txt$', '_regular.txt')
        def isolateTestList = testDBList.replaceAll('\\.txt$', '_isolate.txt')
        sh "touch ${regularTestList}"
        sh "touch ${isolateTestList}"

        return [
            regular: regularTestList,
            isolate: isolateTestList,
            regularCount: 0,
            isolateCount: 0
        ]
    }
}

def cleanUpSlurmResources(def pipeline, SlurmCluster cluster, String jobUID){
    withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        def randomLoginNode = SlurmConfig.getRandomLoginNode(cluster.host)
        def remote = [
            ip           : randomLoginNode,
            host         : randomLoginNode,
            port         : cluster.sshPort,
            user         : "${pipeline.USERNAME}",
            passwd       : "${pipeline.PASSWORD}",
            allowAnyHosts: true,
        ]

        def jobWorkspace = "/home/svc_tensorrt/bloom/scripts/${jobUID}"

        Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")

        Utils.exec(pipeline, script: "echo Sleeping to allow Slurm job completion; sleep 30")

        def slurmJobID = Utils.exec(
            pipeline,
            // Try to grab the job id from ${jobWorkspace}/slurm_job_id.txt.
            // The slurm_run.sh will add the slurm job id in that file.
            script: Utils.sshUserCmd(
                remote,
                "\"cat ${jobWorkspace}/slurm_job_id.txt || true\""
            ),
            returnStdout: true
        ).trim()

        if (!slurmJobID || !slurmJobID.isNumber()) {
            echo "Slurm job may not submit successfully. No job ID found."
        } else {
            Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID}")

            Utils.exec(
                pipeline,
                script: Utils.sshUserCmd(
                    remote,
                    "\"scancel ${slurmJobID} || true; sacct -j ${slurmJobID} --format=JobID,JobName%100,Partition%15,Account%15,State,ExitCode,NodeList%30 || true; scontrol show job ${slurmJobID} || true\""
                )
            )
        }

        Utils.exec(pipeline, script: "echo Sleeping to allow Slurm job termination; sleep 30")

        def cleanupCommands = [
            "rm -rf ${cluster.scratchPath}/users/svc_tensorrt/containers/container-${slurmJobID}.sqsh || true",
            "rm -rf ${jobWorkspace} || true",
        ].join(" ; ")
        Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(
                remote,
                "\"${cleanupCommands}\""
            )
        )

        Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID} cleaned up")
    }
}

// Methods to run Slurm job with Jenkins Agent
def cleanUpNodeResources(def pipeline, SlurmCluster cluster, String nodeName, String slurmJobID) {
    withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
        def randomLoginNode = SlurmConfig.getRandomLoginNode(cluster.host)
        def remote = [
            ip           : randomLoginNode,
            host         : randomLoginNode,
            port         : cluster.sshPort,
            user         : "${pipeline.USERNAME}",
            passwd       : "${pipeline.PASSWORD}",
            allowAnyHosts: true,
        ]

        Utils.exec(pipeline, script: "echo Sleeping to allow docker stop; sleep 30")

        CloudManager.destroyNode(nodeName)

        Utils.exec(pipeline, script: "echo Sleeping to allow node destruction; sleep 30")

        Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")

        Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID}")

        Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(
                remote,
                "\"scancel ${slurmJobID} || true; sacct -j ${slurmJobID} --format=JobID,JobName%100,Partition%15,Account%15,State,ExitCode,NodeList%30 || true; scontrol show job ${slurmJobID} || true\""
            )
        )

        Utils.exec(pipeline, script: "echo Sleeping to allow Slurm job termination; sleep 30")

        def entrypoint = SlurmConfig.containerRuntimeToEntrypoint[cluster.containerRuntime]
        def cleanupCommands = [
            "rm -rf /home/svc_tensorrt/bloom/scripts/agent-${nodeName}.jar /home/svc_tensorrt/bloom/scripts/${nodeName}-${entrypoint} || true",
            "rm -rf ${cluster.scratchPath}/users/svc_tensorrt/containers/container-${slurmJobID}.sqsh || true",
        ].join(" ; ")
        Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(
                remote,
                "\"${cleanupCommands}\""
            )
        )

        Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID} cleaned up")
    }
}

def runLLMTestlistWithAgent(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, skipInstallWheel=false, cpver="cp312")
{
    SlurmPartition partition = SlurmConfig.resolvePlatform(platform)
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]

    def entrypoint = SlurmConfig.containerRuntimeToEntrypoint[cluster.containerRuntime]

    // Create a unique suffix for the node name and workspace
    String customSuffix = "${env.BUILD_TAG}-${UUID.randomUUID().toString().replaceAll("-", "").substring(0, 6)}".toLowerCase()
    def nodeName = "${cluster.host}-test-${customSuffix}"
    def customWorkspace = "/tmp/${nodeName}"
    def nodeSecret = CloudManager.createNode(nodeName, customWorkspace)

    def slurmJobID = null
    def dockerArgs = null

    try {
        // Run ssh command to start node in desired cluster via SLURM
        withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
            def randomLoginNode = SlurmConfig.getRandomLoginNode(cluster.host)
            def remote = [
                    ip           : randomLoginNode,
                    host         : randomLoginNode,
                    port         : cluster.sshPort,
                    user         : "${pipeline.USERNAME}",
                    passwd       : "${pipeline.PASSWORD}",
                    allowAnyHosts: true,
            ]

            Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")
            stage('Request Node Via Slurm') {
                println("Selected Cluster: ${cluster.name}")

                def jenkinsSetupPath = Utils.copyLibraryResource(pipeline, entrypoint)

                Utils.exec(pipeline, script: "cat ${jenkinsSetupPath}")

                Utils.copyFileToRemoteHost(pipeline, remote, jenkinsSetupPath, "/home/svc_tensorrt/bloom/scripts/${nodeName}-${entrypoint}", true)

                Utils.exec(pipeline, script: "echo Sleeping before Slurm job submission; sleep \$((RANDOM % 29 + 1))")

                def mounts = getMountListForSlurmTest(cluster, false).join(",")
                def slurmSubmitOutput = Utils.exec(
                    pipeline,
                    timeout: false,
                    script: Utils.sshUserCmd(
                        remote,
                        "\"${SlurmConfig.generateCommand(cluster, partition, nodeSecret, nodeName, Jenkins.instance.rootUrl, LLM_DOCKER_IMAGE, mounts)}\""
                    ),
                    returnStdout: true,
                    numRetries: 3
                )

                def jobIDs = slurmSubmitOutput
                    .readLines()
                    .collect { it.trim() }
                    .collectMany { line ->
                        def ids = []
                        def m1 = (line =~ /Submitted batch job (\d+)/)
                        if (m1) ids << m1[0][1]  // Extract the first captured group
                        def m2 = (line =~ /srun: job (\d+) (queued|has been allocated)/)
                        if (m2) ids << m2[0][1]  // Extract the first captured group
                        def m3 = (line =~ /SLURM_JOB_ID=(\d+)/)
                        if (m3) ids << m3[0][1]  // Extract the first captured group
                        def m4 = (line =~ /SLURM_JOBID=(\d+)/)
                        if (m4) ids << m4[0][1]  // Extract the first captured group
                        return ids
                    }

                slurmJobID = jobIDs ? jobIDs[-1] : null

                if (!slurmJobID || !slurmJobID.isNumber()) {
                    echo "Slurm job did not submit successfully. No job ID found.\nSubmission output:\n${slurmSubmitOutput}"
                }
                Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID}")
                Utils.exec(pipeline, script: "echo Sleeping to allow agent initialization; sleep 30")
            }
        }

        stage('Check If Node Is Online') {
            withCredentials([usernamePassword(credentialsId: 'svc_tensorrt', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD')]) {
                def randomLoginNode = SlurmConfig.getRandomLoginNode(cluster.host)
                def remote = [
                        ip           : randomLoginNode,
                        host         : randomLoginNode,
                        port         : cluster.sshPort,
                        user         : "${pipeline.USERNAME}",
                        passwd       : "${pipeline.PASSWORD}",
                        allowAnyHosts: true,
                ]
                def counter = 0
                // We submit the Slurm job with 5 hours timeout, and the K8S pod will be evicted after 22 hours.
                // Let's use 15 hours to check if the node is online, and with 2 hours buffer.
                while (!CloudManager.isNodeOnline(nodeName) && counter < 90) {
                    // Wait 10 minutes to check status of the node again
                    sleep(time: 10, unit: 'MINUTES')
                    // Avoid the node being stuck in the held state.
                    if (counter % 3 == 0) {
                        Utils.exec(pipeline, script: Utils.sshUserCmd(remote, "\"scontrol release ${slurmJobID} || true\""), numRetries: 3)
                    }
                    counter++
                    // If entrypoint script fails to start, do not poll for agent connection
                    try {
                        SlurmConfig.checkJobStatus(pipeline, cluster, slurmJobID, remote)
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception e) {
                        // If the exception is about job being inactive, enrich it with log path
                        if (e.message.contains("is no longer active")) {
                            throw new Exception("${e.message}. Check SLURM logs at /home/svc_tensorrt/slurm-logs/slurm-${slurmJobID}-${nodeName}.out on ${cluster.host}")
                        }
                        // Otherwise, log the error but continue (SSH might be temporarily unavailable)
                        pipeline.echo("Warning: Could not check SLURM job status: ${e.message}")
                    }
                }
            }

            if (CloudManager.isNodeOnline(nodeName)) {
                node(nodeName) {
                    sh """
                        env | sort
                        pwd && ls -alh
                        ls -alh ${env.WORKSPACE}
                        ls -alh ${env.WORKSPACE_TMP}
                    """

                    sh "nproc && free -g && hostname"
                    echoNodeAndGpuInfo(pipeline, stageName)
                    sh "nvidia-smi && nvidia-smi -q && nvidia-smi topo -m"
                    // Use single quotes to avoid Jenkins variable expansion
                    sh 'echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"'
                    sh 'echo "NV_GPU: $NV_GPU"'

                    // Dynamically set GPU arguments based on environment variables
                    // https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
                    // It's intentional to check NV_GPU first.
                    dockerArgs = sh(script: """
                        if [ -n "\$NV_GPU" ]; then
                            echo "--gpus '\\"device=\$NV_GPU\\"'"
                        elif [ -n "\$CUDA_VISIBLE_DEVICES" ]; then
                            echo "--gpus '\\"device=\$CUDA_VISIBLE_DEVICES\\"'"
                        else
                            echo "--gpus ${gpuCount}"
                        fi
                    """, returnStdout: true).trim()

                    if (cluster.host.contains("dlcluster")) {
                        dockerArgs += " " + sh(script: 'echo " -e NVIDIA_IMEX_CHANNELS=${NVIDIA_IMEX_CHANNELS:-0}"', returnStdout: true).trim()
                        if (fileExists('/dev/gdrdrv')) {
                            dockerArgs += " --device=/dev/gdrdrv:/dev/gdrdrv"
                        }
                    }
                    if (fileExists('/home/scratch.trt_llm_data_ci')) {
                        dockerArgs += " -v /home/scratch.trt_llm_data_ci:/scratch.trt_llm_data:ro "
                    } else if (fileExists('/home/scratch.trt_llm_data')) {
                        dockerArgs += " -v /home/scratch.trt_llm_data:/scratch.trt_llm_data:ro "
                    } else {
                        echo "Existing TRT-LLM data scratch mount points cannot be set up in this cluster, ignore..."
                    }
                }

                dockerArgs = "${dockerArgs} " +
                    "--cap-add=SYS_ADMIN " +
                    "--ipc=host " +
                    "--entrypoint=\"\" " +
                    "--security-opt seccomp=unconfined " +
                    "-u root:root " +
                    "-v /tmp/ccache:${CCACHE_DIR}:rw " +
                    "-v /tmp/pipcache/http-v2:/root/.cache/pip/http-v2:rw " +
                    "--cap-add=SYSLOG"

                echo "Final dockerArgs: ${dockerArgs}"
            } else {
                error "The Slurm node does not come online in the waiting period. Terminating the job."
            }
        }

        slurmRunner = null
        echo "${stageName} Slurm partition timeout: ${partition.time}"
        def partitionTimeout = partition?.time ? partition.time : SlurmConfig.DEFAULT_TIMEOUT_SHORT
        if (cluster.containerRuntime.toString() == "DOCKER") {
            slurmRunner = runInDockerOnNodeMultiStage(LLM_DOCKER_IMAGE, nodeName, dockerArgs, partitionTimeout, true)
        } else if (cluster.containerRuntime.toString() == "ENROOT") {
            slurmRunner = runInEnrootOnNode(nodeName, partitionTimeout)
        } else {
            throw new Exception("Unsupported container runtime: ${cluster.containerRuntime}")
        }
        executeLLMTestOnSlurm(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver, slurmRunner)
    } finally {
        stage("Clean Up Slurm Resource") {
            // Workaround to handle the interruption during clean up SLURM resources
            retry(3) {
                try {
                    cleanUpNodeResources(pipeline, cluster, nodeName, slurmJobID)
                } catch (Exception e) {
                    error "Error during clean up SLURM resources: ${e.getMessage()} and retrying."
                }
            }
        }
    }
}

def executeLLMTestOnSlurm(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312", runner)
{
    runner {
        // TODO: refactor the finallyRunner to reuse within slurm or nonslurm job.
        cacheErrorAndUploadResult(stageName, {
            runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver)
        }, {
            // If the execution test list is null, remove the test result xml
            sh """
                ls -all ${stageName}/
                if ! grep -q '<testcase' ${stageName}/results.xml; then
                    rm ${stageName}/results.xml || true
                fi
            """
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def llmSrc = "${llmPath}/${LLM_ROOT}${config}/TensorRT-LLM/src"
            // CPP tests will generate test result in ${llmSrc}/cpp/build_backup/, move these files to job result folder
            sh "ls -all ${llmSrc}/cpp/build_backup/ || true"
            sh "ls -all ${llmSrc}/cpp/build/ || true"
            // Sed for CPP test result
            sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/\" classname=\"/\" classname=\"${stageName}./g' *.xml || true"
            sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/testsuite name=\"[^\"]*\"/testsuite name=\"${stageName}\"/g' *.xml || true"
            // Sed for Pytest result
            sh "cd ${stageName} && sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' *.xml || true"
            // Copy CPP test result
            sh "cp ${llmSrc}/cpp/build_backup/*.xml ${stageName} || true"
            sh "ls ${stageName}/ -all"
        })
    }
}
// End of Methods to run Slurm job with Jenkins Agent

def getNodeArgs(int nodeCount, int gpuCount, boolean setSegment = false) {
    int gpusPerNode = ((gpuCount / nodeCount) as BigDecimal).setScale(0, BigDecimal.ROUND_CEILING).intValue()
    def args = nodeCount == 1 ? [
        "--nodes=${nodeCount}",
        "--gpus=${gpuCount}"
    ] : [
        "--nodes=${nodeCount}",
        "--ntasks=${gpuCount}",
        "--ntasks-per-node=${gpusPerNode}",
        "--gpus-per-node=${gpusPerNode}",
    ]
    if (setSegment && gpuCount > 1) {
        args += ["--segment=${nodeCount}"]
    }
    return args
}

def getPytestBaseCommandLine(
    String llmSrc,
    String stageName,
    String waivesFilePath,
    Boolean perfMode,
    String outputPath,
    String trtllmWheelPath,
    String coverageConfigFile,
    String pytestUtil = "",
    List<String> extraArgs = [],
    int containerPortStart = 0,
    int containerPortNum = 0
) {
    def extraInternalEnv = ""
    def pytestTestTimeout = "3600"

    // TRT uses half of the host logic cores for engine building which is bad for multi-GPU machines.
    extraInternalEnv = "__LUNOWUD=\"-thread_pool_size=${TESTER_CORES}\""
    // CPP test execution is timing out easily, so we always override its internal timeout to the same value as pytest
    extraInternalEnv += " CPP_TEST_TIMEOUT_OVERRIDDEN=${pytestTestTimeout}"
    // Enable NCCL debug information for multi-GPU tests
    extraInternalEnv += " NCCL_DEBUG=INFO"

    // Container port allocation environment variables for avoiding port conflicts
    def portEnvVars = ""
    if (containerPortStart > 0 && containerPortNum > 0) {
        portEnvVars = "CONTAINER_PORT_START=${containerPortStart} CONTAINER_PORT_NUM=${containerPortNum}"
    }

    def testCmdLine = [
        "LLM_ROOT=${llmSrc}",
        "LLM_BACKEND_ROOT=${llmSrc}/triton_backend",
        "LLM_MODELS_ROOT=${MODEL_CACHE_DIR}",
        "MODEL_CACHE_DIR=${MODEL_CACHE_DIR}",
        "COLUMNS=300",
        extraInternalEnv,
        portEnvVars,
        pytestUtil,
        "pytest",
        "-vv",
        testFilter[(DETAILED_LOG)] ? "-s" : "",
        "--timeout-method=thread",
        "--apply-test-list-correction",
        "--timeout=${pytestTestTimeout}",
        "--rootdir ${llmSrc}/tests/integration/defs",
        "--test-prefix=${stageName}",
        "--waives-file=${waivesFilePath}",
        "--output-dir=${outputPath}/",
        "--csv=${outputPath}/report.csv",
        "-o junit_logging=out-err",
        "--cov=${llmSrc}/examples/",
        "--cov=${llmSrc}/tensorrt_llm/",
        "--cov=${trtllmWheelPath}/tensorrt_llm/",
        "--cov-report=",
        "--cov-config=${coverageConfigFile}",
        "--periodic-junit",
        "--periodic-junit-xmlpath ${outputPath}/results.xml",
        "--periodic-batch-size=1",
        "--periodic-save-unfinished-test",
    ]

    if (perfMode) {
        testCmdLine += [
            "--perf",
            "--perf-log-formats csv",
            "--perf-log-formats yaml",
            "--enable-gpu-clock-lock"
        ]
    }
    if (stageName.contains("-Ray-")) {
        testCmdLine += ["--run-ray"]
    }
    if (extraArgs) {
        testCmdLine += extraArgs
    }
    return testCmdLine as String[]
}

def getMountListForSlurmTest(SlurmCluster cluster, boolean useSbatch = false)
{
    def mounts = []

    // mounts for SLURM job submission and logs
    if (useSbatch) {
        mounts += [
            "/home/svc_tensorrt/bloom/scripts",
        ]
    } else {
        mounts += [
            "/home/svc_tensorrt/bloom/scripts",
            "/home/svc_tensorrt/slurm-logs",
        ]
    }

    // data/cache mounts
    if (cluster.containerRuntime.toString() == "DOCKER") {
        mounts += [
            "/home/scratch.trt_llm_data_ci:/scratch.trt_llm_data:ro",
        ]
    } else if (cluster.containerRuntime.toString() == "ENROOT") {
        if (!cluster.scratchPath) {
            throw new Exception("Scratch path is not set for cluster: ${cluster.name}")
        }
        mounts += [
            "${cluster.scratchPath}:/scratch.trt_llm_data:ro",
        ]
    } else {
        throw new Exception("Unsupported container runtime: ${cluster.containerRuntime}")
    }

    return mounts
}

def runLLMTestlistWithSbatch(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, nodeCount=1, skipInstallWheel=false, cpver="cp312")
{
    SlurmPartition partition = SlurmConfig.resolvePlatform(platform)
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]

    // Create a unique suffix for the job name
    String customSuffix = "${env.BUILD_TAG}-${UUID.randomUUID().toString().replaceAll("-", "").substring(0, 6)}".toLowerCase()
    def jobUID = "${cluster.host}-multi_node_test-${customSuffix}"
    def disaggMode = stageName.contains("Disagg-PerfSanity")

    Utils.exec(pipeline, script: "env | sort && pwd && ls -alh")

    try {
        // Run ssh command to start node in desired cluster via SLURM
        withCredentials([
            usernamePassword(
                credentialsId: 'svc_tensorrt',
                usernameVariable: 'USERNAME',
                passwordVariable: 'PASSWORD'
            )
        ]) {
            def randomLoginNode = SlurmConfig.getRandomLoginNode(cluster.host)
            def remote = [
                    ip           : randomLoginNode,
                    host         : randomLoginNode,
                    port         : cluster.sshPort,
                    user         : "${pipeline.USERNAME}",
                    passwd       : "${pipeline.PASSWORD}",
                    allowAnyHosts: true,
            ]
            Utils.exec(pipeline, script: "apt-get update && apt-get install -y sshpass openssh-client")
            def tarName = BUILD_CONFIGS[config][TARNAME]
            def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def jobWorkspace = "/home/svc_tensorrt/bloom/scripts/${jobUID}"
            def resourcePathNode = "/tmp"
            def llmSrcNode = "${resourcePathNode}/TensorRT-LLM/src"
            def llmSrcLocal = "${llmPath}/TensorRT-LLM/src"
            def scriptRunLocalPath = "${llmSrcLocal}/jenkins/scripts/slurm_run.sh"
            def scriptRunPathNode = "${jobWorkspace}/${jobUID}-slurm_run.sh"
            def scriptInstallLocalPath = "${llmSrcLocal}/jenkins/scripts/slurm_install.sh"
            def scriptInstallPathNode = "${jobWorkspace}/${jobUID}-slurm_install.sh"
            def scriptBashUtilsLocalPath = "${llmSrcLocal}/jenkins/scripts/bash_utils.sh"
            def scriptBashUtilsPathNode = "${jobWorkspace}/${jobUID}-bash_utils.sh"
            def testListPathNode = "${jobWorkspace}/${testList}.txt"
            def waivesListPathNode = "${jobWorkspace}/waives.txt"
            def slurmJobLogPath = "${jobWorkspace}/job-output.log"
            def scriptLaunchPathLocal = Utils.createTempLocation(pipeline, "./slurm_launch.sh")
            def scriptLaunchPathNode = "${jobWorkspace}/${jobUID}-slurm_launch.sh"
            def scriptSubmitPathLocal = Utils.createTempLocation(pipeline, "./slurm_submit.sh")
            def scriptSubmitPathNode = "${jobWorkspace}/${jobUID}-slurm_submit.sh"
            def scriptTrackPathLocal = Utils.createTempLocation(pipeline, "./slurm_track.sh")
            def scriptTrackPathNode = "${jobWorkspace}/${jobUID}-slurm_track.sh"
            def coverageConfigFile = "${jobWorkspace}/.coveragerc"

            stage("Initialize Test") {
                println("Selected Cluster: ${cluster.name}")
                // Create Job Workspace folder in Frontend Node
                Utils.exec(pipeline, script: Utils.sshUserCmd(remote, "\"mkdir -p ${jobWorkspace}\""), numRetries: 3)

                // Download and Unzip Tar File
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
                sh "cd ${llmPath} && tar -zxf ${BUILD_CONFIGS[config][TARNAME]}"

                Utils.exec(pipeline, script: "echo \"Script for Slurm srun job to submit: \" && cat ${scriptRunLocalPath}")
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    scriptRunLocalPath,
                    scriptRunPathNode,
                    true
                )

                Utils.exec(pipeline, script: "echo \"Script to install TensorRT LLM dependencies: \" && cat ${scriptInstallLocalPath}")
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    scriptInstallLocalPath,
                    scriptInstallPathNode,
                    true
                )
                Utils.exec(pipeline, script: "echo \"Script for Bash utilities: \" && cat ${scriptBashUtilsLocalPath}")
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    scriptBashUtilsLocalPath,
                    scriptBashUtilsPathNode,
                    true
                )

                // Generate Test List and Upload to Frontend Node
                def makoArgs = getMakoArgsFromStageName(stageName, true)
                // TODO: currently the options will only be processed if the first
                // line is "Mako options:", maybe we can make it more generic, which
                // if the line cannot be split by "=", just ignore that line.
                def makoOptsJson = transformMakoArgsToJson(["Mako options:"] + makoArgs)
                def testListPathLocal = renderTestDB(testList, llmSrcLocal, stageName, makoOptsJson)
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    testListPathLocal,
                    testListPathNode
                )

                // Download and Merge waives.txt
                mergeWaivesTxt(pipeline, llmSrcLocal, stageName)

                // Add passed test list from previous pipeline run to the waives.txt
                if (testFilter[(REUSE_TEST)] != false) {
                    reusePassedTestResults(llmSrcLocal, stageName, "${llmSrcLocal}/tests/integration/test_lists/waives.txt")
                }

                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    "${llmSrcLocal}/tests/integration/test_lists/waives.txt",
                    waivesListPathNode
                )

                // generate .coveragerc in workspace and add file path to pytest command
                sh """
                    touch ./.coveragerc
                    echo '[run]' > ./.coveragerc
                    echo 'branch = True' >> ./.coveragerc
                    echo 'data_file = ${jobWorkspace}/.coverage.${stageName}' >> ./.coveragerc
                    echo '[paths]' >> ./.coveragerc
                    echo 'source =\n    ${llmSrcNode}/tensorrt_llm/\n    ---wheel_path---/tensorrt_llm//tensorrt_llm/' >> ./.coveragerc
                    cat ./.coveragerc
                """

                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    "./.coveragerc",
                    coverageConfigFile
                )

                // Generate Pytest command
                String pytestUtil = ""
                if (nodeCount > 1) {
                    pytestUtil = "$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"
                }

                def pytestCommand = getPytestBaseCommandLine(
                    llmSrcNode,
                    stageName,
                    waivesListPathNode,
                    perfMode,
                    jobWorkspace,
                    "__PLACEHOLDER_TRTLLM_WHL_PATH__",
                    "$jobWorkspace/.coveragerc",
                    pytestUtil,
                    [
                      "--test-list=$testListPathNode",
                      "--splitting-algorithm least_duration",
                      "--splits $splits",
                      "--group $splitId"
                    ]
                ).join(" ")

                // Generate Job Launch Script
                def container = LLM_DOCKER_IMAGE.replace("urm.nvidia.com/", "urm.nvidia.com#")
                def mounts = getMountListForSlurmTest(cluster, true).join(",")
                String[] taskArgs = getNodeArgs(nodeCount, gpuCount, disaggMode)
                if (taskArgs == null) {
                    error "Invalid Slurm test stage name is set"
                }
                taskArgs = [
                    *taskArgs,
                ]

                def containerImageArg = container
                def srunPrologue = ""
                if (cluster.containerRuntime.toString() == "ENROOT") {
                    def enrootImagePath = "${cluster.scratchPath}/users/svc_tensorrt/containers/container-\${SLURM_JOB_ID}.sqsh"
                    containerImageArg = enrootImagePath

                    srunPrologue = """
                    export ENROOT_CACHE_PATH='/home/svc_tensorrt/.cache/enroot'

                    importContainerWithRetries() {
                        local docker_uri=\$1
                        local output_path=\$2
                        local max_attempts=\${3:-3}
                        local delay=\${4:-60}
                        local attempt=1

                        rm -f "\$output_path"

                        until enroot import -o "\$output_path" -- "docker://\$docker_uri"
                        do
                            if ((attempt >= max_attempts))
                            then
                                echo "enroot import failed after \$max_attempts attempts"
                                return 1
                            fi

                            echo "enroot import failed (attempt \$attempt of \$max_attempts). Retrying in \${delay}s..."
                            rm -f "\$output_path"
                            sleep \$delay
                            ((attempt++))
                        done
                    }

                    importContainerWithRetries "$container" "$enrootImagePath"
                    """.replaceAll("(?m)^\\s*", "")
                }

                // Define environment variables to export
                def envVarNames = [
                    'OPEN_SEARCH_DB_BASE_URL',
                    'OPEN_SEARCH_DB_CREDENTIALS_USR',
                    'OPEN_SEARCH_DB_CREDENTIALS_PSW',
                    'BUILD_ID',
                    'BUILD_URL',
                    'JOB_NAME',
                    'globalVars',
                    'gitlabCommit'
                ]
                def envVarsToExport = [:]
                envVarNames.each { varName ->
                    envVarsToExport[varName] = env."${varName}"
                }

                srunArgs = [
                    "--container-name=multi_node_test-\${SLURM_JOB_ID}",
                    "--container-image=$containerImageArg",
                    "--container-workdir=$jobWorkspace",
                    "--container-mounts=$mounts",
                    "--container-env=NVIDIA_IMEX_CHANNELS"
                ]
                envVarsToExport.each { varName, varValue ->
                    srunArgs.add("--container-env=${varName}")
                }
                def exemptionComment = ""
                if (cluster.host.contains("oci-nrt") || cluster.host.contains("oci-hsg") || cluster.host.contains("lbd-lax")) {
                    exemptionComment = """--comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"90","reason":"other","description":"Long data and model loading time and disaggregated serving tests"}}'"""
                }

                def envExportStatements = envVarsToExport.collect { varName, varValue ->
                    def escapedValue = varValue?.toString() ?: ''
                    escapedValue = escapedValue
                        .replace('\\', '\\\\')    // Backslash
                        .replace('"', '\\"')      // Double quote
                        .replace('$', '\\$')      // Dollar sign (prevent variable expansion)
                        .replace('`', '\\`')      // Backtick (prevent command substitution)
                    "export ${varName}=\"${escapedValue}\""
                }.join('\n')

                def scriptLaunchPrefix = """#!/bin/bash
                    #SBATCH ${exemptionComment}
                    #SBATCH --output=${slurmJobLogPath}
                    ${taskArgs.collect { "#SBATCH $it" }.join('\n')}
                    #SBATCH ${partition.additionalArgs}
                    ${partition?.time ? "#SBATCH --time=${partition.time}" : "#SBATCH --time=${SlurmConfig.DEFAULT_TIMEOUT_SHORT}"}
                    ${(partition?.name && partition.name != "unspecified") ? "#SBATCH --partition=${partition.name}" : ""}

                    # SBATCH directives must appear before any executable commands.
                    set -xEeuo pipefail
                    trap 'rc=\$?; echo "Error in file \${BASH_SOURCE[0]} on line \$LINENO: \$BASH_COMMAND (exit \$rc)"; exit \$rc' ERR

                    echo "Starting Slurm job \$SLURM_JOB_ID on \$SLURM_NODELIST"
                    export jobWorkspace=$jobWorkspace
                    export tarName=$tarName
                    export llmTarfile=$llmTarfile
                    export llmSrcNode=$llmSrcNode
                    export stageName=$stageName
                    export perfMode=$perfMode
                    export resourcePathNode=$resourcePathNode
                    export pytestCommand="$pytestCommand"
                    export coverageConfigFile="$coverageConfigFile"
                    export NVIDIA_IMEX_CHANNELS=\${NVIDIA_IMEX_CHANNELS:-0}
                    export NVIDIA_VISIBLE_DEVICES=\${NVIDIA_VISIBLE_DEVICES:-\$(seq -s, 0 \$((\$(nvidia-smi --query-gpu=count -i 0 --format=csv,noheader)-1)))}
                    ${envExportStatements}

                    echo "Env NVIDIA_IMEX_CHANNELS: \$NVIDIA_IMEX_CHANNELS"
                    echo "Env NVIDIA_VISIBLE_DEVICES: \$NVIDIA_VISIBLE_DEVICES"

                    ${srunPrologue}
                """.replaceAll("(?m)^\\s*", "")

                if (disaggMode) {
                    if(nodeCount > 1) {
                        srunArgs.add("--mpi=pmix")
                    }

                    def scriptLaunchPrefixPathLocal = Utils.createTempLocation(pipeline, "./slurm_launch_prefix.sh")
                    def scriptLaunchSrunArgsPathLocal = Utils.createTempLocation(pipeline, "./slurm_srun_args.txt")
                    def scriptLaunchDraftPathLocal = "${llmSrcLocal}/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh"
                    def scriptSubmitLocalPath = "${llmSrcLocal}/jenkins/scripts/perf/disaggregated/submit.py"

                    pipeline.writeFile(file: scriptLaunchPrefixPathLocal, text: scriptLaunchPrefix)
                    pipeline.writeFile(file: scriptLaunchSrunArgsPathLocal, text: srunArgs.join(" "))

                    // Output is the corresponding scriptLaunchPathLocal script under the disaggMode
                    sh """
                        python3 ${scriptSubmitLocalPath} \\
                        --run-ci \\
                        --llm-src ${llmSrcLocal} \\
                        --test-list ${testListPathLocal} \\
                        --draft-launch-sh ${scriptLaunchDraftPathLocal} \\
                        --launch-sh ${scriptLaunchPathLocal} \\
                        --run-sh ${scriptRunPathNode} \\
                        --install-sh ${scriptInstallPathNode} \\
                        --script-prefix ${scriptLaunchPrefixPathLocal} \\
                        --srun-args ${scriptLaunchSrunArgsPathLocal}
                    """
                } else {
                    if(nodeCount > 1) {
                        srunArgs.add("--mpi=pmi2")
                    }

                    def scriptContent = """
                        ${scriptLaunchPrefix}
                        srun --kill-on-bad-exit=1 ${srunArgs.join(" ")} ${scriptRunPathNode}
                    """.replaceAll("(?m)^\\s*", "")
                    pipeline.writeFile(file: scriptLaunchPathLocal, text: scriptContent)
                }

                Utils.exec(pipeline, script: "echo \"Script for Slurm sbatch job to submit: \" && cat ${scriptLaunchPathLocal}")
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    scriptLaunchPathLocal,
                    scriptLaunchPathNode,
                    true
                )

                def filesToKeepWhenRetry = [
                    scriptRunPathNode,
                    scriptInstallPathNode,
                    scriptBashUtilsPathNode,
                    scriptLaunchPathNode,
                    scriptSubmitPathNode,
                    scriptTrackPathNode,
                    testListPathNode,
                    waivesListPathNode,
                    coverageConfigFile
                ]
                def findKeepWhenRetryArgs = filesToKeepWhenRetry.collect { " ! -name \"\$(basename \"${it}\")\"" }.join("")

                def scriptSubmit = """#!/bin/bash
                    set -xEeuo pipefail
                    trap 'rc=\$?; echo "Error in file \${BASH_SOURCE[0]} on line \$LINENO: \$BASH_COMMAND (exit \$rc)"; exit \$rc' ERR

                    # Clean up previous job intermediate files so that retry can work
                    if [ -f "${jobWorkspace}/slurm_job_id.txt" ]; then
                        previous_job_id=\$(cat "${jobWorkspace}/slurm_job_id.txt")
                        echo "Found previous Slurm job ID: \${previous_job_id}"
                        scancel "\${previous_job_id}" || true
                        # Wait for 120 seconds to ensure the previous job is canceled
                        sleep 120
                    fi

                    # Clean up workspace: remove all files/dirs not in the keep list
                    find "${jobWorkspace}" -maxdepth 1 -mindepth 1 ${findKeepWhenRetryArgs} -exec rm -rf {} +

                    touch ${slurmJobLogPath}
                    jobId=\$(sbatch ${scriptLaunchPathNode} | awk '{print \$4}')
                    if [ -z "\$jobId" ]; then
                        echo "Error: Slurm job submission failed, no job ID returned."
                        exit 1
                    fi
                    echo "Submitted Slurm job \$jobId"
                    # Save Slurm job ID for later steps to retrieve
                    echo "\$jobId" > "${jobWorkspace}/slurm_job_id.txt"
                """.replaceAll("(?m)^\\s*", "").trim()

                pipeline.writeFile(file: scriptSubmitPathLocal, text: scriptSubmit)
                Utils.exec(pipeline, script: "echo \"Script to submit the final Slurm job: \" && cat ${scriptSubmitPathLocal}")
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    scriptSubmitPathLocal,
                    scriptSubmitPathNode,
                    true
                )
            }

            stage("[${stageName}] Run Pytest") {
                // Submit the Slurm job
                Utils.exec(
                    pipeline,
                    timeout: false,
                    script: Utils.sshUserCmd(
                        remote,
                        scriptSubmitPathNode
                    ),
                    numRetries: 3
                )

                def slurmJobId = Utils.exec(
                    pipeline,
                    script: Utils.sshUserCmd(
                        remote,
                        "\"cat ${jobWorkspace}/slurm_job_id.txt\""
                    ),
                    returnStdout: true,
                    numRetries: 3
                ).trim()
                Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobId}")

                def scriptTrack = """#!/bin/bash
                    set -xEeuo pipefail
                    trap 'rc=\$?; echo "Error in file \${BASH_SOURCE[0]} on line \$LINENO: \$BASH_COMMAND (exit \$rc)"; exit \$rc' ERR

                    jobId=${slurmJobId}
                    tail -f ${slurmJobLogPath} &
                    tailPid=\$!

                    # Wait until Slurm job is done
                    while true; do
                        # Use --allocations to ensure we match the exact job ID and not job steps (like 123.batch, 123.0)
                        STATUS=\$(sacct -j \$jobId --format=State -Pn --allocations)

                        if [[ -z \$STATUS || \$STATUS == "RUNNING" || \$STATUS == "PENDING" || \$STATUS == "CONFIGURING" ]]; then
                            echo "Slurm job \$jobId is still running"
                            sleep 300
                        else
                            echo "Slurm job \$jobId finished with state: \$STATUS"
                            break
                        fi
                    done

                    # Kill tail -f process
                    kill \$tailPid

                    # Wait briefly to ensure accounting is consistent
                    sleep 10

                    # Get exit code (STATUS is already known from loop break)
                    # Retry for exit code if missing
                    for i in {1..3}; do
                        # Use awk to parse exit code from format like "0:0"
                        EXIT_CODE=\$(sacct -j \$jobId --format=ExitCode -Pn --allocations | awk -F: '{print \$1}')

                        if [ -n "\$EXIT_CODE" ]; then
                            break
                        fi
                        echo "Waiting for sacct exit code to update... attempt \$i"
                        sleep 10
                    done

                    if [ -z "\$EXIT_CODE" ]; then
                        echo "Error: Failed to get exit code from sacct after retries, defaulting to 1."
                        EXIT_CODE=1
                    fi

                    # We already have valid STATUS from the loop that caused the break
                    if [[ "\$STATUS" == "COMPLETED" && \$EXIT_CODE -eq 0 ]]; then
                        echo "Pytest succeed in Slurm job \$jobId"
                        echo "Status: \$STATUS | Exit_code \$EXIT_CODE"
                        exit 0
                    else
                        echo "Pytest failed in Slurm job \$jobId"
                        echo "Status: \$STATUS | Exit_code \$EXIT_CODE"
                        exit 1
                    fi
                """.replaceAll("(?m)^\\s*", "").trim()

                pipeline.writeFile(file: scriptTrackPathLocal, text: scriptTrack)
                Utils.exec(pipeline, script: "echo \"Script to track Slurm job and pull the log: \" && cat ${scriptTrackPathLocal}")
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    scriptTrackPathLocal,
                    scriptTrackPathNode,
                    true
                )

                // Track the Slurm job
                Utils.exec(
                    pipeline,
                    timeout: false,
                    script: Utils.sshUserCmd(
                        remote,
                        scriptTrackPathNode
                    ),
                    numRetries: 3
                )
            }
            echo "Finished test stage execution."
        }
    } finally {
        uploadResults(pipeline, cluster, jobUID, stageName)
        stage("Clean Up Slurm Resource") {
            // Workaround to handle the interruption during clean up SLURM resources
            retry(3) {
                try {
                    cleanUpSlurmResources(pipeline, cluster, jobUID)
                } catch (Exception e) {
                    error "Error during clean up SLURM resources: ${e.getMessage()} and retrying."
                }
            }
        }
    }
}

def runLLMTestlistOnSlurm(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, nodeCount=1, runWithSbatch=false, skipInstallWheel=false, cpver="cp312")
{
  echo "Run Slurm job with native sbatch: $runWithSbatch"
  if (nodeCount > 1 || runWithSbatch) {
    runLLMTestlistWithSbatch(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, gpuCount, nodeCount, skipInstallWheel, cpver)
  } else {
    runLLMTestlistWithAgent(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, gpuCount, skipInstallWheel, cpver)
  }
}

def trimForStageList(stageNameList)
{
    if (stageNameList == null) {
        return null
    }
    trimedList = []
    stageNameList.each { stageName ->
        trimedList.add(stageName.trim().replaceAll('\\\\', ''))
    }
    return trimedList
}

// Test filter flags
@Field
def REUSE_TEST = "reuse_test"
@Field
def REUSE_STAGE_LIST = "reuse_stage_list"
@Field
def ENABLE_SKIP_TEST = "skip_test"
@Field
def TEST_STAGE_LIST = "stage_list"
@Field
def GPU_TYPE_LIST = "gpu_type"
@Field
def TEST_BACKEND = "test_backend"
@Field
def IS_POST_MERGE = "post_merge"
@Field
def ADD_MULTI_GPU_TEST = "add_multi_gpu_test"
@Field
def ONLY_MULTI_GPU_TEST = "only_multi_gpu_test"
@Field
def DISABLE_MULTI_GPU_TEST = "disable_multi_gpu_test"
@Field
def EXTRA_STAGE_LIST = "extra_stage"
@Field
def MULTI_GPU_FILE_CHANGED = "multi_gpu_file_changed"
@Field
def ONLY_ONE_GROUP_CHANGED = "only_one_group_changed"
@Field
def AUTO_TRIGGER_TAG_LIST = "auto_trigger_tag_list"
@Field
def DEBUG_MODE = "debug"
@Field
def DETAILED_LOG = "detailed_log"
@Field
def testFilter = [
    (REUSE_TEST): null,
    (REUSE_STAGE_LIST): null,
    (ENABLE_SKIP_TEST): false,
    (TEST_STAGE_LIST): null,
    (GPU_TYPE_LIST): null,
    (TEST_BACKEND): null,
    (IS_POST_MERGE): false,
    (ADD_MULTI_GPU_TEST): false,
    (ONLY_MULTI_GPU_TEST): false,
    (DISABLE_MULTI_GPU_TEST): false,
    (EXTRA_STAGE_LIST): null,
    (MULTI_GPU_FILE_CHANGED): false,
    (ONLY_ONE_GROUP_CHANGED): "",
    (DEBUG_MODE): false,
    (AUTO_TRIGGER_TAG_LIST): [],
    (DETAILED_LOG): false,
]

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

class GlobalState {
    static def uploadResultStageNames = []

    // HOST_NODE_NAME to starting port section map
    // This map maintains the next available starting port for each host node
    // to avoid port conflicts when running parallel tests on the same node.
    // Key: HOST_NODE_NAME (e.g., "node-01.cluster.local")
    // Value: Next available starting port number for that node
    static def hostNodePortMap = [:]

    // Port allocation configuration
    static final int BASE_PORT = 10000           // Base starting port
    static final int PORT_SECTION_SIZE = 1000    // Number of ports per section/stage
    static final int MAX_PORT = 32000            // Maximum port number to avoid system ports
}

/**
 * Allocates and returns a starting port section for the given host node.
 * This function is thread-safe and ensures each stage running on the same
 * host node gets a unique port range to avoid conflicts.
 *
 * @param hostNodeName The HOST_NODE_NAME of the node running the stage
 * @param stageName Optional stage name for logging purposes
 * @return The starting port number for this stage's port section
 */
def getStartingPortForHost(String hostNodeName, String stageName = "") {
    lock(resource: 'globalstate-hostNodePortMap') {
        def startingPort = GlobalState.hostNodePortMap.get(hostNodeName, GlobalState.BASE_PORT)

        // Store the next available starting port for this host
        def nextPort = startingPort + GlobalState.PORT_SECTION_SIZE

        // Wrap around if we exceed MAX_PORT
        if (nextPort > GlobalState.MAX_PORT) {
            nextPort = GlobalState.BASE_PORT
        }

        GlobalState.hostNodePortMap[hostNodeName] = nextPort

        return startingPort
    }
}

/**
 * Gets the HOST_NODE_NAME from the current environment.
 * Falls back to hostname if HOST_NODE_NAME is not set.
 *
 * @return The host node name
 */
def getHostNodeName() {
    return sh(script: '''
        if [ -n "$HOST_NODE_NAME" ]; then
            echo "$HOST_NODE_NAME"
        else
            hostname -f || hostname
        fi
    ''', returnStdout: true).trim()
}

String getShortenedJobName(String path)
{
    static final nameMapping = [
        "L0_MergeRequest": "l0-mr",
        "L0_Custom": "l0-cus",
        "L0_PostMerge": "l0-pm",
        "L0_PostMergeDocker": "l0-pmd",
        "L1_Custom": "l1-cus",
        "L1_Nightly": "l1-nt",
        "L1_Stable": "l1-stb",
        "BuildDockerImageSanityTest": "img-check",
    ]
    def parts = path.split('/')
    // Apply nameMapping to the last part (jobName)
    def jobName = parts[-1]
    boolean replaced = false
    nameMapping.each { key, value ->
        if (jobName.contains(key)) {
            jobName = jobName.replace(key, value)
            replaced = true
        }
    }
    if (!replaced) {
        jobName = jobName.length() > 7 ? jobName.substring(0, 7) : jobName
    }
    // Replace the last part with the transformed jobName
    parts[-1] = jobName
    // Rejoin the parts with '-', convert to lowercase
    return parts.join('-').toLowerCase()
}

def cacheErrorAndUploadResult(stageName, taskRunner, finallyRunner, noResultIfSuccess=false, postTag="")
{
    checkStageName([stageName])
    def Boolean stageIsInterrupted = false
    def Boolean stageIsFailed = true
    try {
        taskRunner()
        stageIsFailed = false
    } catch (InterruptedException e) {
        stageIsInterrupted = true
        throw e
    } finally {
        ensureStageResultNotUploaded(stageName + postTag)
        if (stageIsInterrupted) {
            echo "Stage is interrupted, skip to upload test result."
        } else {
            // Temporarily disable to reduce the log size
            // sh 'if [ "$(id -u)" -eq 0 ]; then dmesg || true; fi'
            if (noResultIfSuccess && !stageIsFailed) {
                // Clean up the workspace
                sh """
                    env | sort
                    pwd && ls -alh
                    rm -rf ./*
                """

                echo "Finished test stage execution."
                return
            }
            echo "noResultIfSuccess: ${noResultIfSuccess}, stageIsFailed: ${stageIsFailed}"
            sh "mkdir -p ${stageName}"
            finallyRunner()
            if (stageIsFailed) {
                def timeoutTestXml = generateTimeoutTestResultXml(stageName, "unfinished_test.txt")
                if (timeoutTestXml != null) {
                    sh """
cat > ${stageName}/results-timeout.xml << 'EOF_TIMEOUT_XML'
${timeoutTestXml}
EOF_TIMEOUT_XML
                    """
                }
                def stageXml = generateStageFailTestResultXml(stageName, "Stage Failed", "Stage run failed without result", "results*.xml")
                if (stageXml != null) {
                    sh "echo '${stageXml}' > ${stageName}/results-stage.xml"
                }
            }
            sh "STAGE_NAME=${stageName}"
            sh "STAGE_NAME=${stageName} && env | sort > ${stageName}/debug_env.txt"
            echo "Upload test results."
            sh "tar -czvf results-${stageName}${postTag}.tar.gz ${stageName}/"
            trtllm_utils.uploadArtifacts(
                "results-${stageName}${postTag}.tar.gz",
                "${UPLOAD_PATH}/test-results/"
            )
            junit(testResults: "${stageName}/results*.xml")
        }

        // Clean up the workspace
        sh """
            env | sort
            pwd && ls -alh
            rm -rf ./*
        """

        echo "Finished test stage execution."
    }
}

def createKubernetesPodConfig(image, type, arch = "amd64", gpuCount = 1, perfMode = false)
{
    def targetCloud = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/arch: ${arch}
                  kubernetes.io/os: linux"""
    def containerConfig = ""
    def nodeLabelPrefix = ""
    def jobName = getShortenedJobName(env.JOB_NAME)
    def buildID = env.BUILD_ID
    def tolerations = ""
    def extraDeviceEnv = ""

    def archSuffix = arch == "arm64" ? "arm" : "amd"
    def jnlpImage = "urm.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_${archSuffix}_linux:jdk17"

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
        nodeLabelPrefix = "cpu"
        break
    case "slurm":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS_SLURM}]
                    tty: true
                    resources:
                      requests:
                        cpu: ${SLURM_CORES_REQUEST}
                        memory: ${SLURM_MEMORY_REQUEST}
                        ephemeral-storage: 100Gi
                      limits:
                        cpu: ${SLURM_CORES_LIMIT}
                        memory: ${SLURM_MEMORY_LIMIT}
                        ephemeral-storage: 100Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    case "build":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS_BUILD}]
                    volumeMounts:
                    - name: sw-tensorrt-pvc
                      mountPath: "/mnt/sw-tensorrt-pvc"
                      readOnly: false
                    tty: true
                    resources:
                      requests:
                        cpu: ${BUILD_CORES_REQUEST}
                        memory: ${BUILD_MEMORY_REQUEST}
                        ephemeral-storage: 200Gi
                      limits:
                        cpu: ${BUILD_CORES_LIMIT}
                        memory: ${BUILD_MEMORY_LIMIT}
                        ephemeral-storage: 200Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    default:
        def hasMultipleGPUs = (gpuCount > 1)
        def memorySize = "${TESTER_MEMORY}"
        def storageSize = "300Gi"
        def driverVersion = REQUIRED_OPEN_DRIVER_TYPES.any { type.contains(it) } ? Constants.DEFAULT_NVIDIA_OPEN_DRIVER_VERSION : Constants.DEFAULT_NVIDIA_DRIVER_VERSION
        def cpuCount = "${TESTER_CORES}"

        if (hasMultipleGPUs)
        {
            // Not a hard requirement, but based on empirical values.
            memorySize = "${gpuCount * 150}" + "Gi"
            storageSize = "${gpuCount * 150}" + "Gi"
            cpuCount = "${gpuCount * 12}"
        }

        def gpuType = KubernetesManager.selectGPU(type)
        nodeLabelPrefix = type

        targetCloud = "kubernetes"
        // DGX Spark requires a special setting for accessing the device.
        // It has 128GB unified memory as per spec. Use half of the memory at the CPU side.
        if (type.contains("gb10x")) {
            targetCloud = "nvks-sparks-cloud"
            memorySize = "64Gi"
            tolerations = """
                tolerations:
                - key: "node_for_blossom_trt"
                  operator: "Exists"
                  effect: "NoSchedule"
            """
            extraDeviceEnv = """
                    - name: NVIDIA_VISIBLE_DEVICES
                      value: "all"
                    - name: NVIDIA_DRIVER_CAPABILITIES
                      value: "compute,utility"
            """
        }

        // The following GPU types doesn't support dynamic driver flashing.
        if (REQUIRED_NO_DRIVER_TYPES.any { type.contains(it) }) {
            if (type.contains("gb10x")) {
                selectors = """
                    kubernetes.io/arch: ${arch}
                    kubernetes.io/os: linux
                    nvidia.com/gpu.machine: NVIDIA_DGX_Spark
                    nvidia.com/tenant: blossom_trt"""
            } else {
                selectors = """
                    kubernetes.io/arch: ${arch}
                    kubernetes.io/os: linux
                    nvidia.com/gpu_type: ${gpuType}"""
            }
        } else if (perfMode && !hasMultipleGPUs) {
        // Use single GPU machine with "tensorrt/test_type: perf" for stable perf testing.
        // H100 / A100 single GPU machine has this unique label in TensorRT Blossom pool.
            selectors = """
                    kubernetes.io/arch: ${arch}
                    kubernetes.io/os: linux
                    nvidia.com/gpu_type: ${gpuType}
                    nvidia.com/driver_version: '${driverVersion}'
                    tensorrt/test_type: perf"""
        }
        else
        {
            selectors = """
                    kubernetes.io/arch: ${arch}
                    kubernetes.io/os: linux
                    nvidia.com/gpu_type: ${gpuType}
                    nvidia.com/driver_version: '${driverVersion}'"""
        }

        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS_TEST}]
                    tty: true
                    resources:
                      requests:
                        cpu: ${cpuCount}
                        memory: ${memorySize}
                        nvidia.com/gpu: ${gpuCount}
                        ephemeral-storage: ${storageSize}
                      limits:
                        cpu: ${cpuCount}
                        memory: ${memorySize}
                        nvidia.com/gpu: ${gpuCount}
                        ephemeral-storage: ${storageSize}
                    imagePullPolicy: Always
                    volumeMounts:
                    - name: dshm
                      mountPath: /dev/shm
                    - name: scratch-trt-llm-data
                      mountPath: /scratch.trt_llm_data
                      readOnly: true
                    - name: sw-tensorrt-pvc
                      mountPath: "/mnt/sw-tensorrt-pvc"
                      readOnly: false
                    securityContext:
                      capabilities:
                        add:
                        - SYS_ADMIN"""
        break
    }
    def nodeLabel = trtllm_utils.appendRandomPostfix("${nodeLabelPrefix}---tensorrt-${jobName}-${buildID}")
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
    def llmModelVolume = """
                - name: scratch-trt-llm-data
                  nfs:
                    server: 10.117.145.14
                    path: /vol/scratch1/scratch.michaeln_blossom
    """

    // Austin FlexCache looks slow and unstable recently. Remove gh200 temporarily.
    // That means gh200 nodes will use the default Blossom data scratch.
    if (type.contains("6000d")) {
        // rtx-pro-6000d and gh200 nodes are located in Austin DC, we use the FlexCache to speed up the data access.
        llmModelVolume = """
                - name: scratch-trt-llm-data
                  nfs:
                    server: 10.20.162.212
                    path: /vol/scratch26/scratch.trt_llm_data
        """
    }

    def podConfig = [
        cloud: targetCloud,
        namespace: "sw-tensorrt",
        label: nodeLabel,
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                qosClass: Guaranteed
                affinity:
                    nodeAffinity:
                        requiredDuringSchedulingIgnoredDuringExecution:
                            nodeSelectorTerms:
                            - matchExpressions:
                              - key: "tensorrt/taints"
                                operator: DoesNotExist
                              - key: "tensorrt/affinity"
                                operator: NotIn
                                values:
                                - "core"
                nodeSelector: ${selectors}
                containers:
                  ${containerConfig}
                    env:
                    - name: HOST_NODE_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: spec.nodeName
                    ${extraDeviceEnv}
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
                qosClass: Guaranteed
                volumes:
                - name: dshm
                  emptyDir:
                    medium: Memory
                ${llmModelVolume}
                ${pvcVolume}
                ${tolerations}
        """.stripIndent(),
    ]

    return podConfig
}

def echoNodeAndGpuInfo(pipeline, stageName)
{
    String hostNodeName = sh(script: '''
        if [ -n "$HOST_NODE_NAME" ]; then
            echo "$HOST_NODE_NAME"
        else
            hostname -f || hostname
        fi
    ''', returnStdout: true).trim()

    String gpuUuids = pipeline.sh(script: "nvidia-smi -q | grep \"GPU UUID\" | awk '{print \$4}' | tr '\n' ',' || true", returnStdout: true)
    pipeline.echo "HOST_NODE_NAME = ${hostNodeName} ; GPU_UUIDS = ${gpuUuids} ; STAGE_NAME = ${stageName}"
}

def runLLMDocBuild(pipeline, config)
{
    // Step 1: cloning source code
    sh "pwd && ls -alh"
    sh "env | sort"
    // allow to checkout from forked repo, svc_tensorrt needs to have access to the repo, otherwise clone will fail
    trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
    sh "mkdir TensorRT-LLM"
    sh "cp -r ${LLM_ROOT}/ TensorRT-LLM/src/"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "git config --global --add safe.directory \"*\"")

    def llmPath = sh (script: "realpath .", returnStdout: true).trim()
    def llmSrc = "${llmPath}/TensorRT-LLM/src"

    // Step 2: download TRT-LLM tarfile
    def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${BUILD_CONFIGS[config][TARNAME]}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
    sh "cd ${llmPath} && tar -zxf ${BUILD_CONFIGS[config][TARNAME]}"
    // install python package
    if (env.alternativeTRT) {
        sh "cd ${llmSrc} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
    }
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmSrc} && pip3 install -r requirements-dev.txt")
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl")

    // Step 3: build doc
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y doxygen python3-pip graphviz")

    def containerPATH = sh(script: "echo \${PATH}", returnStdout: true).replaceAll("\\s", "")
    if (!containerPATH.contains("/usr/local/bin:")) {
        echo "Prepend /usr/local/bin into \${PATH}"
        containerPATH = "/usr/local/bin:${containerPATH}"
    }
    containerPATH = containerPATH.replaceAll(':+$', '')
    withEnv(["PATH=${containerPATH}"]) {
        sh "env | sort"
        sh "rm -rf ${LLM_ROOT}/docs/build"
        trtllm_utils.llmExecStepWithRetry(
            pipeline,
            script: """
                cd ${LLM_ROOT}/docs && \
                pip3 install -r requirements.txt && \
                pip3 install git+https://github.com/sphinx-doc/sphinx.git@v7.4.7 && \
                doxygen Doxygen && \
                make html && \
                cd build/html && \
                touch .nojekyll
            """
        )
    }

    echo "Upload built html."
    sh "tar -czvf doc-html-preview.tar.gz  ${LLM_ROOT}/docs/build/html"
    trtllm_utils.uploadArtifacts(
        "doc-html-preview.tar.gz",
        "${UPLOAD_PATH}/test-results/"
    )
}

def launchTestListCheck(pipeline)
{
    stageName = "Test List Check"
    trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(LLM_DOCKER_IMAGE, "a10"), "trt-llm", {
        try {
            echoNodeAndGpuInfo(pipeline, stageName)
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y libffi-dev")
            sh "nvidia-smi && nvidia-smi -q && nvidia-smi topo -m"
            // download TRT-LLM tarfile
            def tarName = BUILD_CONFIGS[VANILLA_CONFIG][TARNAME]
            def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pwd && wget -nv ${llmTarfile} && ls -alh")
            sh "tar -zxf ${tarName}"
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def llmSrc = "${llmPath}/TensorRT-LLM/src"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install -r ${llmSrc}/requirements-dev.txt")
            sh "NVIDIA_TRITON_SERVER_VERSION=25.12 LLM_ROOT=${llmSrc} LLM_BACKEND_ROOT=${llmSrc}/triton_backend python3 ${llmSrc}/scripts/check_test_list.py --l0 --qa --waive"
        } catch (InterruptedException e) {
            throw e
        } catch (Exception e) {
            throw e
        }
    })
}

def generateTimeoutTestResultXml(stageName, testFilePath) {
    if (!fileExists("${stageName}/${testFilePath}")) {
        echo "No ${testFilePath} found in ${stageName}, skipping timeout XML generation"
        return null
    }
    String timeoutTests = sh(script: "cd ${stageName} && cat ${testFilePath}", returnStdout: true).trim()
    echo "timeoutTests: ${timeoutTests}"

    if (timeoutTests == null || timeoutTests == "") {
        return null
    }
    def testList = timeoutTests.split("\n")
    String xmlContent = """<?xml version="1.0" encoding="UTF-8"?><testsuites>
        <testsuite name="${stageName}" errors="${testList.size()}" failures="0" skipped="0" tests="${testList.size()}" time="1.00">"""
    testList.each { test ->
        xmlContent += """<testcase name="${test}" classname="${stageName}" time="1.0">
        <error message="Test terminated unexpectedly"> Test terminated unexpectedly
        </error></testcase>"""
    }
    xmlContent += "</testsuite></testsuites>"
    return xmlContent
}

def generateStageFailTestResultXml(stageName, subName, failureLog, resultPath) {
    String resultFiles = sh(script: "cd ${stageName} && ls -l ${resultPath} | wc -l", returnStdout: true).trim()
    echo "resultFiles: ${resultFiles}"
    if (resultFiles != "0") {
        return null
    }
    return """<?xml version="1.0" encoding="UTF-8"?><testsuites>
        <testsuite name="${stageName}" errors="0" failures="1" skipped="0" tests="1" time="1.00">
        <testcase name="${subName}" classname="${stageName}" time="1.0">
        <failure message="${failureLog}"> ${failureLog}
        </failure></testcase></testsuite></testsuites>"""
}

def transformMakoArgsToJson(optList) {
    def makoOpts = [:]
    def startedMakoOpts = false
    def param = null
    def value = null
    optList.each { val ->
        if (startedMakoOpts) {
            // Handle case where value is missing
            param = null
            value = null
            try {
                (param, value) = val.split("=")
            } catch (ArrayIndexOutOfBoundsException ex) {
                param = val.split("=")[0]
                value = null
            }

            // Try to convert nulls, booleans, and floats into the correct type
            if (value != null) {
                if (value.toLowerCase() == "none") {
                    echo "Converted mako param '${param}' value '${value}' to 'null'"
                    value = null
                } else if (value.toLowerCase() in ["true", "false"]) {
                    echo "Converted mako param '${param}' value '${value}' to Boolean '${value.toBoolean()}'"
                    value = value.toBoolean()
                }
            }
            makoOpts[(param)] = value
        }
        if (val.equals("Mako options:")) {
            startedMakoOpts = true
        }
    }

    def makoOptsJson = JsonOutput.toJson(makoOpts)

    // Print and return the Test DB Query as a JSON string
    echo "Test DB Mako opts: ${makoOptsJson}"
    return makoOptsJson
}

def getMakoOpts(getMakoScript, makoArgs=[]) {
    // We want to save a map for the Mako opts
    def makoOutput = ""

    // Echo the command
    // NOTE: We redirect stderr to stdout so that we can capture
    //  both stderr and stdout streams with the 'returnStdout' flag
    //  in sh command.
    def listMakoCmd = [
        "python3",
        getMakoScript,
        "--device 0"].join(" ")

    if (makoArgs) {
        def makoOptArgs = makoArgs.collect { "--mako-opt " + it }
        listMakoCmd += " " + makoOptArgs.join(" ")
    }
    // Add the withCredentials step to access gpu-chip-mapping file
    withCredentials([file(credentialsId: 'gpu-chip-mapping', variable: 'GPU_CHIP_MAPPING')]) {
        listMakoCmd = [listMakoCmd, "--chip-mapping-file ${GPU_CHIP_MAPPING}"].join(" ")
        listMakoCmd = [listMakoCmd, "2>&1"].join(" ")

        echo "Scripts to get Mako list, cmd: ${listMakoCmd}"

        // Capture the mako output, add timeout in case any hang
        timeout(time: 30, unit: 'MINUTES'){
            makoOutput = sh(label: "Capture Mako Parameters", script: listMakoCmd, returnStdout: true)
        }
    }

    // Validate output
    assert makoOutput: "Mako opts not found - could not construct test db test list."

    // Split each line of mako output into a list
    def outputList = makoOutput.split("\n")

    def makoOptsJson = transformMakoArgsToJson(outputList)

    return makoOptsJson
}

def parseMultiNodeTaskConfigFromStageName(String stageName) {
    def taskConfig = null
    def matcher = (stageName =~ /([^-]+)-(\d+)_GPUs(?:-(\d+)_Nodes)?/)
    if (matcher.find()) {
        taskConfig = [
            gpu: "${matcher.group(1)}",
            system_gpu_count: "${matcher.group(2)}",
            node_count: matcher.group(3) ?: "1" // Default to 1 if _Nodes not present
        ]
    }
    return taskConfig
}

def getMakoArgsFromStageName(stageName, parseSysinfo=false) {
    def makoArgs = []
    def isPostMerge = stageName.contains("Post-Merge")
    makoArgs += [isPostMerge ? "stage=post_merge" : "stage=pre_merge"]
    // Determine the backend type based on keywords in stageName
    if (stageName.contains("-PyTorch-")) {
        // If stageName contains "-PyTorch-", add "backend=pytorch" to makoArgs
        // At this point, only tests with backend=pytorch or unspecified backend will be run
        makoArgs += ["backend=pytorch"]
    } else if (stageName.contains("-TensorRT-")) {
        // If stageName contains "-TensorRT-", add "backend=tensorrt" to makoArgs
        // At this point, only tests with backend=tensorrt or unspecified backend will be run
        makoArgs += ["backend=tensorrt"]
    } else if (stageName.contains("-CPP-")) {
        // If stageName contains "-CPP-", add "backend=cpp" to makoArgs
        // At this point, only tests with backend=cpp or unspecified backend will be run
        makoArgs += ["backend=cpp"]
    } else if (stageName.contains("-Triton-")) {
        // If stageName contains "-Triton-", add "backend=triton" to makoArgs
        // At this point, only tests with backend=triton or unspecified backend will be run
        makoArgs += ["backend=triton"]
    } else if (stageName.contains("-FMHA-")) {
        // If stageName contains "-FMHA-", add "backend=fmha" to makoArgs
        // At this point, only tests with backend=fmha or unspecified backend will be run
        makoArgs += ["backend=fmha"]
    } else if (stageName.contains("-AutoDeploy-")) {
        // If stageName contains "-AutoDeploy-", add "backend=autodeploy" to makoArgs
        // At this point, only tests with backend=autodeploy or unspecified backend will be run
        makoArgs += ["backend=autodeploy"]
    } else {
        // If stageName does not contain "-PyTorch-", "-TensorRT-", "-CPP-", "-Triton-", "-FMHA-", or "-AutoDeploy-", do not add any backend
        // At this point, all tests will be run
        // For cases where backend is not specified in makoArgs, we will match all types of backends and tests without specified backend
    }
    if (stageName.contains("-DeepSeek-")) {
        makoArgs += ["auto_trigger=deepseek"]
    } else if (stageName.contains("-GptOss-")) {
        makoArgs += ["auto_trigger=gpt_oss"]
    } else {
        makoArgs += ["auto_trigger=others"]
    }
    if (stageName.contains("-Ray-")) {
        // If stageName contains "-Ray-", add "orchestrator=ray" to makoArgs
        // At this point, only tests with orchestrator=ray or unspecified orchestrator will be run.
        // Mark tests with orchestrator=mpi to exclude them from Ray stage.
        makoArgs += ["orchestrator=ray"]
    } else {
        // Otherwise select tests with orchestrator=mpi or unspecified orchestrator
        makoArgs += ["orchestrator=mpi"]
    }

    if (parseSysinfo) {
        def taskConfig = parseMultiNodeTaskConfigFromStageName(stageName)
        if (taskConfig) {
            makoArgs += [
                "gpu=${taskConfig.gpu}",
                "system_gpu_count=${taskConfig.system_gpu_count}"
            ]
        }
    }

    return makoArgs
}

def renderTestDB(testContext, llmSrc, stageName, preDefinedMakoOpts=null) {
    def makoOpts = preDefinedMakoOpts

    if (!makoOpts) {
        def scriptPath = "${llmSrc}/tests/integration/defs/sysinfo/get_sysinfo.py"
        def makoArgs = getMakoArgsFromStageName(stageName)
        makoOpts = getMakoOpts(scriptPath, makoArgs)
    }

    sh "pip3 install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-tensorrt-pypi/simple --ignore-installed trt-test-db==1.8.5+bc6df7"
    def testDBPath = "${llmSrc}/tests/integration/test_lists/test-db"
    def testList = "${llmSrc}/${testContext}.txt"
    def testDBQueryCmd = [
        "trt-test-db",
        "-d",
        testDBPath,
        "--context",
        testContext,
        "--test-names",
        "--output",
        testList,
        "--match",
        "'${makoOpts}'"
    ].join(" ")

    sh(label: "Render test list from test-db", script: testDBQueryCmd)
    sh(script: "cat ${testList}")

    return testList
}

def getSSHConnectionPorts(portConfigFile, stageName)
{
    def type = stageName.split('-')[0]
    echo "The type is: ${type}"
    def fileContent = sh(script: "cat ${portConfigFile}", returnStdout: true).trim()

    // Get available VM port list from portConfigFile based on stage name (e.g. A10: [10022, 10023])
    def portList = []
    fileContent.split('\n').each { line ->
        def matcher = (line =~ /(.+?)=\[(.+?)\]/)
        if (matcher) {
            def key = matcher[0][1].replaceAll("\\s","")
            def values = matcher[0][2].replaceAll("\\s","").split(',').collect { it.replaceAll("\\s","") }
            if (key == type) {
                portList.addAll(values)
            }
        }
    }
    echo "Port List for ${type}: ${portList}"

    // Get current port usage status
    def portUsage = ""
    withCredentials([
        usernamePassword(credentialsId: 'tensorrt_llm_infra_debug_vm_01_credentials', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD'),
        string(credentialsId: 'DEBUG_HOST_NAME', variable: 'HOST_NAME')
        ]) {
        portUsage = sh(script: "ssh -v ${USERNAME}@${HOST_NAME} ${COMMON_SSH_OPTIONS} 'netstat -tuln'", returnStdout: true)
    }
    echo "Port Usage: ${portUsage}"

    // Get an available VM port
    def userPort = 0
    while (portList.size() > 0) {
        def randomIndex = (int)(Math.random() * portList.size())
        def curPort = portList[randomIndex].toInteger()
        if (!portUsage.contains(":${curPort}")) {
            userPort = curPort
            break
        }
        portList.remove(randomIndex)
    }

    if (userPort == 0) {
        echo "There is no available port for ${type}"
        return [0, 0]
    }

    echo "The chosen port is: ${userPort}"

    // Calculate autossh monitor port by subtracting 9000 from VM port (e.g. 10022 -> 1022)
    // If monitor port is already in use, randomly assign a value between 2000-3000
    def monitorPort = userPort - 9000
    while (portUsage.contains(":${monitorPort}")) {
        monitorPort = 2000 + (int)(Math.random() * 1000)
    }

    echo "The monitor port is: ${monitorPort}"

    return [userPort, monitorPort]
}

// Return true means the test rerun also fails. Return false otherwise.
def rerunFailedTests(stageName, llmSrc, testCmdLine, resultFileName="results.xml", testType="regular") {
    if (!fileExists("${WORKSPACE}/${stageName}/${resultFileName}")) {
        echo "There is no ${resultFileName} file, skip the rerun step"
        return true
    }

    // Create rerun directory structure to avoid conflicts
    def rerunDir = "${WORKSPACE}/${stageName}/rerun/${testType}"
    sh "mkdir -p ${rerunDir}"

    // Generate rerun test lists
    def failSignaturesList = trtllm_utils.getFailSignaturesList().join(",")
    sh """
        python3 ${llmSrc}/jenkins/scripts/test_rerun.py \
        generate_rerun_tests_list \
        --output-dir=${rerunDir}/ \
        --input-file=${WORKSPACE}/${stageName}/${resultFileName} \
        --fail-signatures='${failSignaturesList}'
    """

    // If there are some failed tests that cannot be rerun (e.g. test duration > 10 min and no known failure signatures),
    // fail the stage immediately without attempting any reruns
    def rerunTestList = "${rerunDir}/rerun_0.txt"
    if (fileExists(rerunTestList)) {
        sh "cat ${rerunTestList}"
        echo "There are some failed tests that cannot be rerun, skip the rerun step."
        return true
    }

    // If the stage has more than 5 failed tests, skip the rerun step
    def validLineCount = 0
    for (times in [1, 2]) {
        def currentRerunTestList = "${rerunDir}/rerun_${times}.txt"
        if (fileExists(currentRerunTestList)) {
            count = sh(
                script: "grep -v '^[[:space:]]*\$' ${currentRerunTestList} | wc -l",
                returnStdout: true
            ).trim().toInteger()
            echo "Found ${count} ${testType} tests to rerun ${times} time(s)"
            validLineCount += count
        }
    }
    if (validLineCount > 5) {
        echo "There are more than 5 failed ${testType} tests, skip the rerun step."
        return true
    } else if (validLineCount == 0) {
        echo "No failed ${testType} tests need to be rerun, skip the rerun step."
        return true
    }

    // Rerun tests
    def isRerunFailed = false
    for (times in [1, 2]) {
        def currentRerunTestList = "${rerunDir}/rerun_${times}.txt"
        if (!fileExists(currentRerunTestList)) {
            echo "No failed ${testType} tests need to be rerun ${times} time(s)"
            continue
        }
        sh "cat ${currentRerunTestList}"
        def xmlFile = "${rerunDir}/rerun_results_${times}.xml"
        // change the testCmdLine for rerun
        def noNeedLine = ["--splitting-algorithm", "--splits", "--group", "--cov"]
        def needToChangeLine = ["--test-list", "--csv", "--periodic-junit-xmlpath"]
        def newTestCmdLine = testCmdLine.findAll { cmd ->
            !noNeedLine.any { line -> cmd.contains(line) } && !needToChangeLine.any { line -> cmd.contains(line) }
        }
        newTestCmdLine += [
            "--test-list=${currentRerunTestList}",
            "--csv=${rerunDir}/rerun_report_${times}.csv",
            "--periodic-junit-xmlpath ${xmlFile}",
            "--reruns ${times - 1}"
        ]
        try {
            sh """
                cd ${llmSrc}/tests/integration/defs && \
                ${newTestCmdLine.join(" ")}
            """
        } catch(InterruptedException e) {
            throw e
        } catch (Exception e) {
            if (!fileExists(xmlFile)) {
                echo "The ${testType} tests crashed when rerun attempt."
                throw e
            }
            echo "The ${testType} tests still failed after rerun attempt."
            isRerunFailed = true
        }
    }

    echo "isRerunFailed for ${testType}: ${isRerunFailed}"
    return isRerunFailed
}

def generateRerunReport(stageName, llmSrc) {
    echo "Generating comprehensive rerun report for stage: ${stageName}"

    def rerunBaseDir = "${WORKSPACE}/${stageName}/rerun"
    def regularRerunDir = "${rerunBaseDir}/regular"

    // Check if regular rerun directory has rerun_results_*.xml files
    def hasRegularReruns = sh(script: "[ -d '${regularRerunDir}' ] && find '${regularRerunDir}' -name 'rerun_results_*.xml' | head -1 | grep -q . && echo 'true' || echo 'false'", returnStdout: true).trim() == 'true'

    // Check if any isolated rerun directories have rerun_results_*.xml files
    def hasIsolatedReruns = sh(script: "find ${rerunBaseDir} -type d -name 'isolated_*' -exec find {} -name 'rerun_results_*.xml' \\; 2>/dev/null | head -1 | grep -q . && echo 'true' || echo 'false'", returnStdout: true).trim() == 'true'

    // Find isolated tests that have actual rerun results and build mapping
    def isolatedTestsWithReruns = []
    if (hasIsolatedReruns) {
        def isolatedDirsOutput = sh(script: "find ${rerunBaseDir} -type d -name 'isolated_*' 2>/dev/null || true", returnStdout: true).trim()
        if (isolatedDirsOutput) {
            def isolatedDirs = isolatedDirsOutput.split('\n').findAll { it.trim() }
            isolatedDirs.each { isolatedDir ->
                // Extract the isolated number from directory name (e.g., isolated_0 -> 0)
                def isolatedNum = isolatedDir.split('/').last().replace('isolated_', '')

                // Check if this isolated directory has any rerun results
                def hasRerunResults = sh(script: "find '${isolatedDir}' -name 'rerun_results_*.xml' | head -1 | grep -q . && echo 'true' || echo 'false'", returnStdout: true).trim() == 'true'

                if (hasRerunResults) {
                    isolatedTestsWithReruns.add([
                        dir: isolatedDir,
                        num: isolatedNum,
                        originalResult: "${WORKSPACE}/${stageName}/results_isolated_${isolatedNum}.xml"
                    ])
                }
            }
        }
    }

    // Collect rerun result files and corresponding original result files
    def rerunResultFiles = []

    echo "Found regular reruns: ${hasRegularReruns}"
    echo "Found isolated tests with reruns: ${isolatedTestsWithReruns.collect { "isolated_${it.num}" }}"

    if (!hasRegularReruns && !hasIsolatedReruns) {
        echo "No rerun results found, skipping rerun report generation"
        return
    }

    // Specify the stage name correctly for all result xml files.
    sh "cd ${WORKSPACE}/${stageName} && find . -name '*.xml' -exec sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' {} + || true"

    // Collect all original and rerun result files
    def allInputFiles = []

    // Add original results
    if (fileExists("${WORKSPACE}/${stageName}/results.xml")) {
        allInputFiles.add("${WORKSPACE}/${stageName}/results.xml")
        // Add to rerunResultFiles only if it has reruns
        if (hasRegularReruns) {
            rerunResultFiles.add("${WORKSPACE}/${stageName}/results.xml")
        }
    }

    // Add ALL isolated test results to allInputFiles
    def isolatedResults = sh(script: "find ${WORKSPACE}/${stageName} -name 'results_isolated_*.xml' 2>/dev/null || true", returnStdout: true).trim()
    if (isolatedResults) {
        isolatedResults.split('\n').each { file ->
            if (file.trim()) {
                allInputFiles.add(file.trim())
            }
        }
        // Add isolated test results that have reruns to rerunResultFiles and add their rerun results to allInputFiles
        isolatedTestsWithReruns.each { isolatedTest ->
            if (fileExists(isolatedTest.originalResult)) {
                rerunResultFiles.add(isolatedTest.originalResult)
                echo "Added isolated result with reruns to rerunResultFiles: ${isolatedTest.originalResult}"
            }
            for (times in [1, 2]) {
                def rerunFile = "${isolatedTest.dir}/rerun_results_${times}.xml"
                if (fileExists(rerunFile)) {
                    allInputFiles.add(rerunFile)
                    rerunResultFiles.add(rerunFile)
                }
            }
        }
    }

    // Add regular rerun results
    if (hasRegularReruns) {
        for (times in [1, 2]) {
            def rerunFile = "${regularRerunDir}/rerun_results_${times}.xml"
            if (fileExists(rerunFile)) {
                allInputFiles.add(rerunFile)
                rerunResultFiles.add(rerunFile)
            }
        }
    }

    if (allInputFiles.isEmpty()) {
        echo "No valid input files found for rerun report generation"
        return
    }

    echo "Generating rerun report with input files: ${rerunResultFiles.join(',')}"

    // Generate comprehensive rerun report
    sh """
        python3 ${llmSrc}/jenkins/scripts/test_rerun.py \
        generate_rerun_report \
        --output-file=${WORKSPACE}/${stageName}/rerun_results.xml \
        --input-files=${rerunResultFiles.join(",")}
    """

    // Update original results xml file with all rerun results for junit
    sh """
        python3 ${llmSrc}/jenkins/scripts/test_rerun.py \
        merge_junit_xmls \
        --output-file=${WORKSPACE}/${stageName}/results.xml \
        --input-files=${allInputFiles.join(",")} \
        --deduplicate
    """

    // Upload rerun report
    if (fileExists("${WORKSPACE}/${stageName}/rerun_results.html")) {
        trtllm_utils.uploadArtifacts(
            "${WORKSPACE}/${stageName}/rerun_results.html",
            "${UPLOAD_PATH}/rerun_reports/${stageName}_rerun_results.html"
        )
        echo "Test rerun report: https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/rerun_reports/${stageName}_rerun_results.html"
    }

    // Remove isolation results since they are merged into results.xml
    sh "rm -rf ${WORKSPACE}/${stageName}/results_isolated_*.xml || true"

    echo "Rerun report generation completed for stage: ${stageName}"
}

def mergeWaivesTxt(pipeline, llmSrc, stageName) {
    def waivesTxt = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/waive_list/waives.txt"
    try {
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv ${waivesTxt}")
        if (!fileExists("waives.txt")) {
            error "There is no merged waives.txt file, use the default waives.txt."
        }
        sh "mv waives.txt ${llmSrc}/tests/integration/test_lists/waives.txt"
        echo "Download merged waives.txt successfully"
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        echo "Failed to download merged waives.txt, use the default waives.txt. Error: ${e.message}"
    }
}

def reusePassedTestResults(llmSrc, stageName, waivesTxt) {
    try {
        // Get passed test list from open search
        def passedTestListFile = "${WORKSPACE}/${stageName}/passed_test_list.txt"
        sh """
            python3 ${llmSrc}/jenkins/scripts/open_search_query.py \
            --commit-id ${env.gitlabCommit} \
            --stage-name ${stageName} \
            --output-file ${passedTestListFile}
        """

        def passedTestList = readFile(file: passedTestListFile).readLines()
        def reusedTests = passedTestList.collect { test -> test.trim() }

        // Append reused tests to waives.txt
        if (reusedTests.size() > 0) {
            // Build the content to append
            def reusedTestsContent = reusedTests.collect { test ->
                "${test} SKIP (Reused from previous pipeline)"
            }.join('\n')

            echo "Reused tests:\n${reusedTestsContent}"

            sh(label: "Append Reused Tests", script: """
cat >> ${waivesTxt} << 'REUSED_TESTS_EOF'
${reusedTestsContent}
REUSED_TESTS_EOF
""")
            echo "Appended ${reusedTests.size()} reused tests to ${waivesTxt}"
        } else {
            echo "No reused tests found"
        }
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        echo "Failed to add passed test list from previous pipeline run to the waives.txt. Error: ${e.message}"
    }
}

def runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312")
{
    // Step 1: create LLM_ROOT dir and clean up the workspace
    def llmRootConfig = "${LLM_ROOT}${config}"
    sh """
        env | sort
        pwd && ls -alh
        rm -rf ./*
        mkdir ${llmRootConfig}
        ls -alh ${env.WORKSPACE}
        ls -alh ${env.WORKSPACE_TMP}
    """

    def llmPath = sh (script: "realpath ${llmRootConfig}", returnStdout: true).trim()
    def llmSrc = "${llmPath}/TensorRT-LLM/src"
    echoNodeAndGpuInfo(pipeline, stageName)

    if (env.alternativeTRT && cpver) {
        stage("Replace TensorRT") {
            trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
        }
    }

    // Step 2: run tests
    stage ("Setup Environment")
    {
        // Random sleep to avoid resource contention
        sleep(10 * Math.random())
        sh "curl ifconfig.me || true"
        sh "nproc && free -g && hostname"
        echoNodeAndGpuInfo(pipeline, stageName)
        sh "cat ${MODEL_CACHE_DIR}/README"
        sh "nvidia-smi && nvidia-smi -q && nvidia-smi topo -m"
        sh "df -h"

        // setup HF_HOME to cache model and datasets
        // init the huggingface cache from nfs, since the nfs is read-only, and HF_HOME needs to be writable, otherwise it will fail at creating file lock
        sh "mkdir -p ${HF_HOME} && ls -alh ${HF_HOME}"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y rsync")
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "rsync -r ${MODEL_CACHE_DIR}/hugging-face-cache/ ${HF_HOME}/ && ls -lh ${HF_HOME}")
        sh "df -h"

        // install package
        sh "env | sort"
        sh "which python3"
        sh "python3 --version"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install -y libffi-dev")
        sh "rm -rf results-${stageName}.tar.gz ${stageName}/*"
        // download TRT-LLM tarfile
        def tarName = BUILD_CONFIGS[config][TARNAME]
        def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
        sh "cd ${llmPath} && tar -zxf ${tarName}"

        // install python package
        if (env.alternativeTRT) {
            sh "cd ${llmSrc} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
        }
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmSrc} && pip3 install -r requirements-dev.txt")
        if (stageName.contains("-Ray-")) {
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install ray[default]")
        }
        if (!skipInstallWheel) {
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl")
        }

        trtllm_utils.llmExecStepWithRetry(pipeline, script: "mkdir -p /opt/tritonserver/backends/tensorrtllm")
        def isAarch64 = config.contains("aarch64")
        if (!isAarch64) {
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && cp TensorRT-LLM/triton_backend/inflight_batcher_llm/libtriton_tensorrtllm.so /opt/tritonserver/backends/tensorrtllm/")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && cp TensorRT-LLM/triton_backend/inflight_batcher_llm/trtllmExecutorWorker /opt/tritonserver/backends/tensorrtllm/")
        }
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "git config --global --add safe.directory \"*\"")
    }

    if (testFilter[(DEBUG_MODE)]) {
        stage("Interactive Debug Session")
        {
            testFilter[(DEBUG_MODE)] = false

            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install openssh-server -y")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install autossh -y")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get install sshpass -y")

            sh """
                echo 'Port 22' >> /etc/ssh/sshd_config
                echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config
                echo 'PasswordAuthentication yes' >> /etc/ssh/sshd_config
                echo 'PubkeyAuthentication yes' >> /etc/ssh/sshd_config
                echo 'AllowTcpForwarding yes' >> /etc/ssh/sshd_config
                echo 'GatewayPorts yes' >> /etc/ssh/sshd_config
                cat /etc/ssh/sshd_config
            """

            sh "service ssh restart"
            sh "service ssh status"

            sh "ssh-keygen -t rsa -b 2048 -f ~/.ssh/id_rsa -N '' -q"

            sh """
                chmod 700 ~/.ssh
                chmod 400 ~/.ssh/id_rsa
                touch ~/.ssh/authorized_keys
                chmod 600 ~/.ssh/authorized_keys
            """

            // Write env variables to a file
            sh 'env | sort | sed -E \'s/^([^=]+)=(.*)$/export \\1="\\2"/\' > debug_env.sh'
            sh "cat debug_env.sh"

            // The portConfig file is in the VM
            def portConfigFilePath = "/root/.ssh/ports_config.txt"

            withCredentials([
                usernamePassword(credentialsId: 'tensorrt_llm_infra_debug_vm_01_credentials', usernameVariable: 'USERNAME', passwordVariable: 'PASSWORD'),
                string(credentialsId: 'DEBUG_HOST_NAME', variable: 'HOST_NAME')
                ]) {
                sh "sshpass -p ${PASSWORD} -v ssh ${USERNAME}@${HOST_NAME} ${COMMON_SSH_OPTIONS} 'cat >> ~/.ssh/authorized_keys' < ~/.ssh/id_rsa.pub"
                sh "ssh -v ${USERNAME}@${HOST_NAME} ${COMMON_SSH_OPTIONS} 'echo \"\" > ~/.ssh/known_hosts && cat ~/.ssh/id_rsa.pub' >> ~/.ssh/authorized_keys"
                sh "ssh -v ${USERNAME}@${HOST_NAME} ${COMMON_SSH_OPTIONS} 'cat ~/.ssh/ports_config.txt' >> ${portConfigFilePath}"

                def (int userPort, int monitorPort) = getSSHConnectionPorts(portConfigFilePath, stageName)
                if (userPort == 0) {
                    echo "Fail to setup an interactive debug session and exit the debug mode."
                    testFilter[(DEBUG_MODE)] = false
                    return
                }

                sh "ssh -f ${COMMON_SSH_OPTIONS} -L 1111:127.0.0.1:${monitorPort} -R ${monitorPort}:127.0.0.1:1112 -NR ${userPort}:localhost:22 ${USERNAME}@${HOST_NAME}"
                sh "autossh -fNR ${userPort}:localhost:22 ${USERNAME}@${HOST_NAME}"
                sh "ps aux | grep ssh"
                try {
                    timeout(time: 2, unit: 'HOURS') {
                        input message: "Pause 2 hours for Pre-Debug. Please type 'ssh root@${HOST_NAME} -p ${userPort}' on the CLI to create the connection. Please press the button to proceed when you finish debugging."
                    }
                } catch (InterruptedException e) {
                    echo "Pre-debug session was interrupted by user or timeout"
                    currentBuild.result = 'ABORTED'
                    error("Pipeline aborted during pre-debug session")
                } catch (Exception e) {
                    echo "An error occurred during pre-debug session: ${e.message}"
                    currentBuild.result = 'FAILURE'
                    error("Error in pre-debug session: ${e.message}")
                }
            }

            testFilter[(DEBUG_MODE)] = true
        }
    }

    stage ("[${stageName}] Run Pytest")
    {
        def noRegularTests = false
        def noIsolateTests = false
        def rerunFailed = false
        def testDBList = renderTestDB(testList, llmSrc, stageName)

        // Download and Merge waives.txt
        mergeWaivesTxt(pipeline, llmSrc, stageName)

        // Add passed test list from previous pipeline run to the waives.txt
        if (testFilter[(REUSE_TEST)] != false) {
            reusePassedTestResults(llmSrc, stageName, "${llmSrc}/tests/integration/test_lists/waives.txt")
        }

        // Process shard test list and create separate files for regular and isolate tests
        def preprocessedLists = processShardTestList(llmSrc, testDBList, splitId, splits, perfMode)

        // Test Coverage
        def TRTLLM_WHL_PATH = sh(returnStdout: true, script: "pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2").replaceAll("\\s","")
        sh "echo ${TRTLLM_WHL_PATH}"
        def coverageConfigFile = "${llmSrc}/${stageName}/.coveragerc"
        sh "mkdir -p ${llmSrc}/${stageName} && touch ${coverageConfigFile}"
        sh """
            echo '[run]' > ${coverageConfigFile}
            echo 'branch = True' >> ${coverageConfigFile}
            echo 'data_file = ${WORKSPACE}/${stageName}/.coverage.${stageName}' >> ${coverageConfigFile}
            echo '[paths]' >> ${coverageConfigFile}
            echo 'source =\n    ${llmSrc}/tensorrt_llm/\n    ${TRTLLM_WHL_PATH}/tensorrt_llm/' >> ${coverageConfigFile}
            cat ${coverageConfigFile}
        """
        echoNodeAndGpuInfo(pipeline, stageName)

        // Allocate a unique port section for this container to avoid port conflicts
        def hostNodeName = getHostNodeName()
        def containerPortStart = getStartingPortForHost(hostNodeName, stageName)
        def containerPortNum = GlobalState.PORT_SECTION_SIZE

        // Some clusters do not allow dmesg -C so we add || true
        // Temporarily disable to reduce the log size
        // sh 'if [ "$(id -u)" -eq 0 ]; then dmesg -C || true; fi'
        def pytestCommand = getPytestBaseCommandLine(
            llmSrc,
            stageName,
            "${llmSrc}/tests/integration/test_lists/waives.txt",
            perfMode,
            "${WORKSPACE}/${stageName}",
            TRTLLM_WHL_PATH,
            coverageConfigFile,
            "",  // pytestUtil
            [],  // extraArgs
            containerPortStart,
            containerPortNum
        )

        // Only add --test-list if there are regular tests to run
        if (preprocessedLists.regularCount > 0) {
            pytestCommand += ["--test-list=${preprocessedLists.regular}"]
        }

        def containerPIP_LLM_LIB_PATH = sh(script: "pip3 show tensorrt_llm | grep \"Location\" | awk -F\":\" '{ gsub(/ /, \"\", \$2); print \$2\"/tensorrt_llm/libs\"}'", returnStdout: true).replaceAll("\\s","")
        def containerLD_LIBRARY_PATH = sh(script: "echo \${LD_LIBRARY_PATH}", returnStdout: true).replaceAll("\\s","")
        if (!containerLD_LIBRARY_PATH.contains("${containerPIP_LLM_LIB_PATH}:")) {
            echo "Prepend ${containerPIP_LLM_LIB_PATH} into \${LD_LIBRARY_PATH}"
            containerLD_LIBRARY_PATH = "${containerPIP_LLM_LIB_PATH}:${containerLD_LIBRARY_PATH}"
        }
        containerLD_LIBRARY_PATH = containerLD_LIBRARY_PATH.replaceAll(':+$', '')
        withEnv(["LD_LIBRARY_PATH=${containerLD_LIBRARY_PATH}"]) {
            withCredentials([
                usernamePassword(
                    credentialsId: 'svc_tensorrt_gitlab_read_api_token',
                    usernameVariable: 'GITLAB_API_USER',
                    passwordVariable: 'GITLAB_API_TOKEN'
                ),
                string(credentialsId: 'llm_evaltool_repo_url', variable: 'EVALTOOL_REPO_URL')
            ]) {
                sh "env | sort"
                try {
                    if (preprocessedLists.regularCount > 0) {
                        sh """
                            rm -rf ${stageName}/ && \
                            cd ${llmSrc}/tests/integration/defs && \
                            ${pytestCommand.join(" ")}
                        """
                    } else {
                        echo "No regular tests to run for stage ${stageName}"
                        noRegularTests = true
                        sh "mkdir -p ${stageName}"
                        // Create an empty results.xml file for consistency
                        sh """
                            echo '<?xml version="1.0" encoding="UTF-8"?>' > ${stageName}/results.xml
                            echo '<testsuites>' >> ${stageName}/results.xml
                            echo '<testsuite name="${stageName}" errors="0" failures="0" skipped="0" tests="0" time="0.0">' >> ${stageName}/results.xml
                            echo '</testsuite>' >> ${stageName}/results.xml
                            echo '</testsuites>' >> ${stageName}/results.xml
                        """
                    }
                } catch (InterruptedException e) {
                    throw e
                } catch (Exception e) {
                    def isRerunFailed = rerunFailedTests(stageName, llmSrc, pytestCommand, "results.xml", "regular")
                    if (isRerunFailed) {
                        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                            error "Regular tests failed after rerun attempt"
                        }
                        rerunFailed = true
                    }
                }

                // Run the isolated tests if exists
                if (preprocessedLists.isolateCount > 0) {
                    stage ("[${stageName}] Run Pytest (Isolated)") {
                        echo "There are ${preprocessedLists.isolateCount} isolated tests to run"
                        rerunFailed = runIsolatedTests(preprocessedLists, pytestCommand, llmSrc, stageName) || rerunFailed
                    }
                } else {
                    echo "No isolated tests to run for stage ${stageName}"
                    noIsolateTests = true
                }

                if (noRegularTests && noIsolateTests) {
                    error "No tests were executed for stage ${stageName}, please check the test list and test-db rendering result."
                }
            }
        }

        // Generate comprehensive rerun report if any reruns occurred
        stage ("Generate Report") {
            generateRerunReport(stageName, llmSrc)
        }

        if (rerunFailed) {
            error "Some tests still failed after rerun attempts, please check the test report."
        }

        if (perfMode) {
            basePerfFilename = stageName.contains("PyTorch") ? "base_perf_pytorch.csv" : "base_perf.csv"
            basePerfPath = "${llmSrc}/tests/integration/defs/perf/${basePerfFilename}"
            stage("Check Perf Result") {
                def perfCheckResult = sh(
                    script: """
                    python3 ${llmSrc}/tests/integration/defs/perf/sanity_perf_check.py \
                        ${stageName}/perf_script_test_results.csv \
                        ${basePerfPath}
                    """,
                    returnStatus: true
                )
                if (perfCheckResult != 0) {
                    error "Performance regression detected and failing the build (exit code: ${perfCheckResult})"
                }
            }
            stage("Create Perf Report") {
                sh """
                    python3 ${llmSrc}/tests/integration/defs/perf/create_perf_comparison_report.py \
                    --output_path ${stageName}/report.pdf \
                    --files ${stageName}/perf_script_test_results.csv \
                    ${basePerfPath}
                """
            }
        }
    }
}


def runLLMTestlistOnPlatform(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312", postTag="")
{
    cacheErrorAndUploadResult(stageName, {
        runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver)
    }, {
        if (testFilter[(DEBUG_MODE)]) {
            try {
                timeout(time: 2, unit: 'HOURS') {
                    input message: "Pause 2 hours for Post-Debug. Please press the button to proceed when you finish debugging."
                }
            } catch (InterruptedException e) {
                echo "Post-debug session was interrupted by user or timeout"
                currentBuild.result = 'ABORTED'
                error("Pipeline aborted during post-debug session")
            } catch (Exception e) {
                echo "An error occurred during post-debug session: ${e.message}"
                currentBuild.result = 'FAILURE'
                error("Error in post-debug session: ${e.message}")
            }
        }
        // If the execution test list is null, remove the test result xml
        sh """
            ls -all ${stageName}/
            if ! grep -q '<testcase' ${stageName}/results.xml; then
                rm ${stageName}/results.xml || true
            fi
        """
        def llmPath = sh (script: "realpath .", returnStdout: true).trim()
        def llmSrc = "${llmPath}/${LLM_ROOT}${config}/TensorRT-LLM/src"
        // CPP tests will generate test result in ${llmSrc}/cpp/build_backup/, move these files to job result folder
        sh "ls -all ${llmSrc}/cpp/build_backup/ || true"
        sh "ls -all ${llmSrc}/cpp/build/ || true"
        // Sed for CPP test result
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/\" classname=\"/\" classname=\"${stageName}./g' *.xml || true"
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/testsuite name=\"[^\"]*\"/testsuite name=\"${stageName}\"/g' *.xml || true"
        // Sed for Pytest result
        sh "cd ${stageName} && sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' *.xml || true"
        // Copy CPP test result
        sh "cp ${llmSrc}/cpp/build_backup/*.xml ${stageName} || true"
        sh "ls ${stageName}/ -all"
    }, false, postTag)
}


def checkPipInstall(pipeline, wheel_path)
{
    def wheelArtifactLinks = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/${wheel_path}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT}/tests/unittest && python3 test_pip_install.py --wheel_path ${wheelArtifactLinks}")
}


def runLLMBuild(pipeline, cpu_arch, reinstall_dependencies=false, wheel_path="", cpver="cp312")
{
    sh "pwd && ls -alh"
    sh "env | sort"
    sh "ccache -sv"

    trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, "tensorrt_llm", true, true)
    if (env.alternativeTRT) {
        sh "cd tensorrt_llm/ && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
    }

    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"
    sh "cat ${CCACHE_DIR}/ccache.conf"
    sh "bash -c 'pip3 show tensorrt || true'"
    if (reinstall_dependencies == true) {
        sh "#!/bin/bash \n" + "pip3 uninstall -y torch"
        sh "#!/bin/bash \n" + "yum remove -y libcudnn*"
    }

    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && pip3 install -r requirements-dev.txt")
    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
    }
    buildArgs = "--clean --nixl_root /opt/nvidia/nvda_nixl"
    if (cpu_arch == AARCH64_TRIPLE) {
        buildArgs += " -a '90-real;100-real;103-real;120-real'"
    }

    withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'CONAN_LOGIN_USERNAME', passwordVariable: 'CONAN_PASSWORD')]) {
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && python3 scripts/build_wheel.py --use_ccache -G Ninja -j ${BUILD_JOBS} -D 'WARNING_IS_ERROR=ON' ${buildArgs}")
    }
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    def wheelName = sh(returnStdout: true, script: 'cd tensorrt_llm/build && ls -1 *.whl').trim()
    echo "uploading ${wheelName} to ${cpu_arch}/${wheel_path}"
    trtllm_utils.uploadArtifacts("tensorrt_llm/build/${wheelName}",  "${UPLOAD_PATH}/${cpu_arch}/${wheel_path}")

    if (reinstall_dependencies == true) {
        // Test installation in the new environment
        def pip_keep = "-e 'pip'"
        def remove_trt = "rm -rf /usr/local/tensorrt"
        if (env.alternativeTRT) {
            pip_keep += " -e tensorrt"
            remove_trt = "echo keep /usr/local/tensorrt"
        }
        sh "#!/bin/bash \n" + "pip3 list --format=freeze | egrep -v ${pip_keep} | xargs pip3 uninstall -y"
        sh "#!/bin/bash \n" + "yum remove -y libcudnn* libnccl* libcublas* && ${remove_trt}"
    }
    // Test preview installation
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && pip3 install pytest build/tensorrt_llm-*.whl")
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    return wheelName
}


def runPackageSanityCheck(pipeline, wheel_path, reinstall_dependencies=false, cpver="cp312")
{
    def whlUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/${wheel_path}"

    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"
    sh "bash -c 'pip3 show tensorrt || true'"
    sh "cat ${MODEL_CACHE_DIR}/README"
    sh "nvidia-smi && nvidia-smi -q && nvidia-smi topo -m"

    sh "pwd && ls -alh"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv ${whlUrl}")

    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
        sh "bash -c 'pip3 show tensorrt || true'"
    }
    if (reinstall_dependencies) {
        // Test installation in the new environment
        def pip_keep = "-e 'pip'"
        def remove_trt = "rm -rf /usr/local/tensorrt"
        if (env.alternativeTRT) {
            pip_keep += " -e tensorrt"
            remove_trt = "echo keep /usr/local/tensorrt"
        }
        sh "bash -c 'pip3 list --format=freeze | egrep -v ${pip_keep} | xargs pip3 uninstall -y'"
        sh "bash -c 'yum remove -y libcudnn* libnccl* libcublas* && ${remove_trt}'"
    }
    // Test preview installation
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'pip3 install pytest tensorrt_llm-*.whl'")
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    def pkgUrl = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${linuxPkgName}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv ${pkgUrl}")
    sh "tar -zvxf ${linuxPkgName}"

    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core/gpt && python3 ../../../generate_checkpoint_config.py --architecture GPTForCausalLM --dtype float16'")
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core//gpt && trtllm-build --model_config config.json --log_level verbose'")
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core/gpt && python3 ../../../run.py --max_output_len 4 --end_id -1'")
}

def checkStageNameSet(stageNames, jobKeys, paramName) {
    echo "Validate stage names for the passed GitLab bot params [${paramName}]."
    invalidStageName = stageNames.findAll { !(it in jobKeys) }
    if (invalidStageName) {
        def sortedJobKeys = jobKeys.sort()
        throw new Exception("Cannot find the stage names [${invalidStageName}] from the passed params [${paramName}]. Available stage names (${sortedJobKeys.size()} total):\n${sortedJobKeys.collect { "    ${it}" }.join('\n')}")
    }
}

def checkStageName(stageNames) {
    invalidStageName = stageNames.findAll { !(it ==~ /[-\+\w\[\]]+/) }
    if (invalidStageName) {
        throw new Exception("Invalid stage name: [${invalidStageName}], we only support chars '-+_[]0-9a-zA-Z' .")
    }
}

def ensureStageResultNotUploaded(stageName) {
    if(!GlobalState.uploadResultStageNames.contains(stageName)) {
        GlobalState.uploadResultStageNames.add(stageName)
    } else {
        stage('Upload Test Result') {
            catchError(buildResult: 'FAILURE', stageResult: 'FAILURE') {
                error "Upload test results for ${stageName} failed because it has already been uploaded."
            }
        }
    }
}

// TODO: Update existing functions to use runInDockerOnNodeMultiStage and get rid of runInDockerOnNode
def runInDockerOnNodeMultiStage(image, label, dockerArgs, partitionTimeout, needToDeleteDir=true)
{
    return {
        runner -> node(label) {
            try {
                if (needToDeleteDir) {
                    deleteDir()
                }
                stage('Pull Docker Image') {
                    docker.image(image).pull()
                }
                // We submit the Slurm job with the Slurm partition's time spec.
                // Minus 10 minutes to avoid the Slurm job being stopped earlier.
                timeout(time: partitionTimeout - 10, unit: 'MINUTES') {
                    docker.image(image).inside(dockerArgs) {
                        runner()
                    }
                }
            } catch (Exception e) {
                if (e.getMessage()?.contains("Failed to kill container")) {
                    echo "Known benign error ignored: ${e.getMessage()}"
                } else {
                    throw e // Re-throw if it's a different Exception
                }
            }
        }
    }
}

def runInEnrootOnNode(label, partitionTimeout)
{
    return {
        runner -> node(label) {
            // We submit the Slurm job with the Slurm partition's time spec.
            // Minus 10 minutes to avoid the Slurm job being stopped earlier.
            timeout(time: partitionTimeout - 10, unit: 'MINUTES') {
                runner()
            }
        }
    }
}

def runInKubernetes(pipeline, podSpec, containerName)
{
    return {
        stageName, runner -> stage(stageName) {
            trtllm_utils.launchKubernetesPod(pipeline, podSpec, containerName) {
                echoNodeAndGpuInfo(pipeline, stageName)
                runner()
            }
        }
    }
}

def buildStageConfigs(stageName, platform, testlist, testCount, gpuCount, nodeCount, runWithSbatch=false) {
    def configs = [:]
    for (int k = 1; k <= testCount; k++) {
        def key = "${stageName}-${k}"
        configs[key] = [platform, testlist, k, testCount, gpuCount, nodeCount, runWithSbatch]
    }
    return configs
}

def launchTestJobs(pipeline, testFilter)
{
    // IMPORTANT: Stage Configuration Syntax Requirement
    //
    // The test_to_stage_mapping.py script expects stage definitions in the following format:
    // "Stage-Name": ["platform", "yaml_file", splitId, split_count, gpu_count]
    //
    // Where:
    // - Stage-Name: Must be quoted string, used to identify the Jenkins stage
    // - platform: Hardware platform identifier (e.g., "a10", "h100-cr")
    // - yaml_file: Test database YAML filename without .yml extension (e.g., "l0_a10")
    // - splitId: Current split number (1-based)
    // - split_count: Total number of splits
    // - gpu_count: Number of GPUs required (optional, defaults to 1)
    //
    // This format is parsed by scripts/test_to_stage_mapping.py to provide bidirectional
    // mapping between test names and Jenkins stage names. Any changes to this syntax
    // may break the mapping functionality.

    x86TestConfigs = [
        "DGX_H100-4_GPUs-CPP-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "A10-PyTorch-1": ["a10", "l0_a10", 1, 2],
        "A10-PyTorch-2": ["a10", "l0_a10", 2, 2],
        "A10-CPP-1": ["a10", "l0_a10", 1, 1],
        "A10-TensorRT-1": ["a10", "l0_a10", 1, 5],
        "A10-TensorRT-2": ["a10", "l0_a10", 2, 5],
        "A10-TensorRT-3": ["a10", "l0_a10", 3, 5],
        "A10-TensorRT-4": ["a10", "l0_a10", 4, 5],
        "A10-TensorRT-5": ["a10", "l0_a10", 5, 5],
        "A30-Triton-1": ["a30", "l0_a30", 1, 1],
        "A30-PyTorch-1": ["a30", "l0_a30", 1, 2],
        "A30-PyTorch-2": ["a30", "l0_a30", 2, 2],
        "A30-AutoDeploy-1": ["a30", "l0_a30", 1, 1],
        "A30-CPP-1": ["a30", "l0_a30", 1, 3],
        "A30-CPP-2": ["a30", "l0_a30", 2, 3],
        "A30-CPP-3": ["a30", "l0_a30", 3, 3],
        "A100X-PyTorch-1": ["a100x", "l0_a100", 1, 1],
        "L40S-PyTorch-1": ["l40s", "l0_l40s", 1, 2],
        "L40S-PyTorch-2": ["l40s", "l0_l40s", 2, 2],
        "H100_PCIe-PyTorch-Ray-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-AutoDeploy-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-CPP-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-TensorRT-1": ["h100-cr", "l0_h100", 1, 1],
        "B200_PCIe-PyTorch-1": ["b100-ts2", "l0_b200", 1, 3],
        "B200_PCIe-PyTorch-2": ["b100-ts2", "l0_b200", 2, 3],
        "B200_PCIe-PyTorch-3": ["b100-ts2", "l0_b200", 3, 3],
        "B200_PCIe-AutoDeploy-1": ["b100-ts2", "l0_b200", 1, 1],
        "RTX5090-PyTorch-1": ["rtx-5090", "l0_gb202", 1, 1],
        "RTX5080-TensorRT-1": ["rtx-5080", "l0_gb203", 1, 2],
        "RTX5080-TensorRT-2": ["rtx-5080", "l0_gb203", 2, 2],
        // Currently post-merge test stages only run tests with "stage: post_merge" mako
        // in the test-db. This behavior may change in the future.
        "A10-PyTorch-Post-Merge-1": ["a10", "l0_a10", 1, 1],
        // "A10-TensorRT-Post-Merge-1": ["a10", "l0_a10", 1, 2],
        // "A10-TensorRT-Post-Merge-2": ["a10", "l0_a10", 2, 2],
        "A10-FMHA-Post-Merge-1": ["a10", "l0_a10", 1, 1],
        // "A30-TensorRT-Post-Merge-1": ["a30", "l0_a30", 1, 6],
        // "A30-TensorRT-Post-Merge-2": ["a30", "l0_a30", 2, 6],
        // "A30-TensorRT-Post-Merge-3": ["a30", "l0_a30", 3, 6],
        // "A30-TensorRT-Post-Merge-4": ["a30", "l0_a30", 4, 6],
        // "A30-TensorRT-Post-Merge-5": ["a30", "l0_a30", 5, 6],
        // "A30-TensorRT-Post-Merge-6": ["a30", "l0_a30", 6, 6],
        "A30-CPP-Post-Merge-1": ["a30", "l0_a30", 1, 1],
        "A30-Triton-Post-Merge-1": ["a30", "l0_a30", 1, 2],
        "A30-Triton-Post-Merge-2": ["a30", "l0_a30", 2, 2],
        // "A100X-TensorRT-Post-Merge-1": ["a100x", "l0_a100", 1, 6],
        // "A100X-TensorRT-Post-Merge-2": ["a100x", "l0_a100", 2, 6],
        // "A100X-TensorRT-Post-Merge-3": ["a100x", "l0_a100", 3, 6],
        // "A100X-TensorRT-Post-Merge-4": ["a100x", "l0_a100", 4, 6],
        // "A100X-TensorRT-Post-Merge-5": ["a100x", "l0_a100", 5, 6],
        // "A100X-TensorRT-Post-Merge-6": ["a100x", "l0_a100", 6, 6],
        "A100X-Triton-Post-Merge-1": ["a100x", "l0_a100", 1, 2],
        "A100X-Triton-Post-Merge-2": ["a100x", "l0_a100", 2, 2],
        "A100X-FMHA-Post-Merge-1": ["a100x", "l0_a100", 1, 1],
        // "L40S-TensorRT-Post-Merge-1": ["l40s", "l0_l40s", 1, 5],
        // "L40S-TensorRT-Post-Merge-2": ["l40s", "l0_l40s", 2, 5],
        // "L40S-TensorRT-Post-Merge-3": ["l40s", "l0_l40s", 3, 5],
        // "L40S-TensorRT-Post-Merge-4": ["l40s", "l0_l40s", 4, 5],
        // "L40S-TensorRT-Post-Merge-5": ["l40s", "l0_l40s", 5, 5],
        "L40S-FMHA-Post-Merge-1": ["l40s", "l0_l40s", 1, 1],
        "H100_PCIe-CPP-Post-Merge-1": ["h100-cr", "l0_h100", 1, 1],
        // "H100_PCIe-TensorRT-Post-Merge-1": ["h100-cr", "l0_h100", 1, 5],
        // "H100_PCIe-TensorRT-Post-Merge-2": ["h100-cr", "l0_h100", 2, 5],
        // "H100_PCIe-TensorRT-Post-Merge-3": ["h100-cr", "l0_h100", 3, 5],
        // "H100_PCIe-TensorRT-Post-Merge-4": ["h100-cr", "l0_h100", 4, 5],
        // "H100_PCIe-TensorRT-Post-Merge-5": ["h100-cr", "l0_h100", 5, 5],
        "H100_PCIe-FMHA-Post-Merge-1": ["h100-cr", "l0_h100", 1, 1],
        "B200_PCIe-Triton-Post-Merge-1": ["b100-ts2", "l0_b200", 1, 1],
        "B200_PCIe-PyTorch-Post-Merge-1": ["b100-ts2", "l0_b200", 1, 2],
        "B200_PCIe-PyTorch-Post-Merge-2": ["b100-ts2", "l0_b200", 2, 2],
        // "B200_PCIe-TensorRT-Post-Merge-1": ["b100-ts2", "l0_b200", 1, 2],
        // "B200_PCIe-TensorRT-Post-Merge-2": ["b100-ts2", "l0_b200", 2, 2],
        "H100_PCIe-TensorRT-Perf-1": ["h100-cr", "l0_perf", 1, 1],
        "H100_PCIe-PyTorch-Perf-1": ["h100-cr", "l0_perf", 1, 1],
        "DGX_H200-4_GPUs-Triton-Post-Merge-1": ["dgx-h200-x4", "l0_dgx_h200", 1, 1, 4],
        "DGX_H200-8_GPUs-PyTorch-Post-Merge-1": ["dgx-h200-x8", "l0_dgx_h200", 1, 1, 8],
        "DGX_H200-4_GPUs-PyTorch-Post-Merge-1": ["dgx-h200-x4", "l0_dgx_h200", 1, 1, 4],
        // "DGX_H200-4_GPUs-TensorRT-Post-Merge-1": ["dgx-h200-x4", "l0_dgx_h200", 1, 3, 4],
        // "DGX_H200-4_GPUs-TensorRT-Post-Merge-2": ["dgx-h200-x4", "l0_dgx_h200", 2, 3, 4],
        // "DGX_H200-4_GPUs-TensorRT-Post-Merge-3": ["dgx-h200-x4", "l0_dgx_h200", 3, 3, 4],
        // Disable RTXPro6000 stages due to nodes will be offline temporarily.
        // [TODO] Split tests between RTXPro6000 and RTXPro6000D and move reasonable mount of tests to pre-merge.
        // "RTXPro6000-PyTorch-Post-Merge-1": ["rtx-pro-6000", "l0_rtx_pro_6000", 1, 1],
        // "RTXPro6000-4_GPUs-PyTorch-Post-Merge-1": ["rtx-pro-6000-x4", "l0_rtx_pro_6000", 1, 2, 4],
        // "RTXPro6000-4_GPUs-PyTorch-Post-Merge-2": ["rtx-pro-6000-x4", "l0_rtx_pro_6000", 2, 2, 4],
        "RTXPro6000D-PyTorch-1": ["rtx-pro-6000d", "l0_rtx_pro_6000", 1, 2],
        "RTXPro6000D-PyTorch-2": ["rtx-pro-6000d", "l0_rtx_pro_6000", 2, 2],
        "RTXPro6000D-4_GPUs-PyTorch-Post-Merge-1": ["rtx-pro-6000d-x4", "l0_rtx_pro_6000", 1, 2, 4],
        "RTXPro6000D-4_GPUs-PyTorch-Post-Merge-2": ["rtx-pro-6000d-x4", "l0_rtx_pro_6000", 2, 2, 4],
    ]

    parallelJobs = x86TestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "amd64", values[4] ?: 1, key.contains("-Perf-")), {
        def config = VANILLA_CONFIG
        if (key.contains("single-device")) {
            config = SINGLE_DEVICE_CONFIG
        }
        if (key.contains("llvm")) {
            config = LLVM_CONFIG
        }
        runLLMTestlistOnPlatform(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3])
    }]]}
    fullSet = parallelJobs.keySet()

    x86SlurmTestConfigs = [
        "DGX_H100_PCIe-PyTorch-1": ["dgx-h100-oci", "l0_h100", 1, 4],
        "DGX_H100_PCIe-PyTorch-2": ["dgx-h100-oci", "l0_h100", 2, 4],
        "DGX_H100_PCIe-PyTorch-3": ["dgx-h100-oci", "l0_h100", 3, 4],
        "DGX_H100_PCIe-PyTorch-4": ["dgx-h100-oci", "l0_h100", 4, 4],
        "DGX_H100_PCIe-PyTorch-Post-Merge-1": ["dgx-h100-oci", "l0_h100", 1, 2],
        "DGX_H100_PCIe-PyTorch-Post-Merge-2": ["dgx-h100-oci", "l0_h100", 2, 2],
        "DGX_H100-2_GPUs-PyTorch-Others-1": ["dgx-h100-x2-oci", "l0_dgx_h100", 1, 2, 2],
        "DGX_H100-2_GPUs-PyTorch-Others-2": ["dgx-h100-x2-oci", "l0_dgx_h100", 2, 2, 2],
        "DGX_H100-2_GPUs-PyTorch-GptOss-1": ["dgx-h100-x2-oci", "l0_dgx_h100", 1, 1, 2],
        "DGX_H100-2_GPUs-PyTorch-Ray-1": ["dgx-h100-x2-oci", "l0_dgx_h100", 1, 1, 2],
        "DGX_H100-4_GPUs-PyTorch-DeepSeek-1": ["dgx-h100-x4-oci", "l0_dgx_h100", 1, 2, 4],
        "DGX_H100-4_GPUs-PyTorch-DeepSeek-2": ["dgx-h100-x4-oci", "l0_dgx_h100", 2, 2, 4],
        "DGX_H100-4_GPUs-PyTorch-GptOss-1": ["dgx-h100-x4-oci", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-PyTorch-Others-1": ["dgx-h100-x4-oci", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-PyTorch-Ray-1": ["dgx-h100-x4-oci", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-AutoDeploy-1": ["dgx-h100-x4-oci", "l0_dgx_h100", 1, 1, 4],
        "DGX_B200-4_GPUs-PyTorch-1": ["b200-x4-lbd", "l0_dgx_b200", 1, 1, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Ray-1": ["b200-x4-lbd", "l0_dgx_b200", 1, 1, 4, 1, true],
        "DGX_B200-4_GPUs-AutoDeploy-1": ["b200-x4-lbd", "l0_dgx_b200", 1, 1, 4, 1, true],
        "DGX_B200-8_GPUs-PyTorch-1": ["b200-x8-lbd", "l0_dgx_b200", 1, 1, 8, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Post-Merge-1": ["b200-x4-lbd", "l0_dgx_b200", 1, 2, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Post-Merge-2": ["b200-x4-lbd", "l0_dgx_b200", 2, 2, 4, 1, true],
        "B300-PyTorch-1": ["b300-single", "l0_b300", 1, 1],
        "DGX_B300-4_GPUs-PyTorch-1": ["b300-x4", "l0_dgx_b300", 1, 1, 4],
        "DGX_B300-4_GPUs-PyTorch-Post-Merge-1": ["b300-x4", "l0_dgx_b300", 1, 2, 4],
        "DGX_B300-4_GPUs-PyTorch-Post-Merge-2": ["b300-x4", "l0_dgx_b300", 2, 2, 4],
        // PerfSanity post-merge tests
        "DGX_B200-8_GPUs-PyTorch-PerfSanity-Post-Merge-1": ["b200-x8-lbd", "l0_dgx_b200_perf_sanity", 1, 3, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-PerfSanity-Post-Merge-2": ["b200-x8-lbd", "l0_dgx_b200_perf_sanity", 2, 3, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-PerfSanity-Post-Merge-3": ["b200-x8-lbd", "l0_dgx_b200_perf_sanity", 3, 3, 8, 1, true],
    ]
    fullSet += x86SlurmTestConfigs.keySet()

    parallelSlurmJobs = x86SlurmTestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, "slurm", "amd64"), {
        def config = VANILLA_CONFIG
        if (key.contains("single-device")) {
            config = SINGLE_DEVICE_CONFIG
        }
        if (key.contains("llvm")) {
            config = LLVM_CONFIG
        }
        runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3], values[4] ?: 1, values[5] ?: 1, values[6] ?: false)
    }]]}

    parallelJobs += parallelSlurmJobs

    // SBSA machines from the Blossom machine pool
    SBSATestConfigs = [
        "GH200-TensorRT-Post-Merge-1": ["gh200", "l0_gh200", 1, 1],
        // DGX Spark is also named as GB10 Grace Blackwell Superchip.
        "GB10-PyTorch-1": ["gb10x", "l0_gb10", 1, 1],
    ]
    fullSet += SBSATestConfigs.keySet()

    SBSASlurmTestConfigs = [
        "GB200-4_GPUs-PyTorch-1": ["auto:gb200-x4", "l0_gb200_multi_gpus", 1, 2, 4],
        "GB200-4_GPUs-PyTorch-2": ["auto:gb200-x4", "l0_gb200_multi_gpus", 2, 2, 4],
        "GB200-4_GPUs-PyTorch-Post-Merge-1": ["auto:gb200-x4", "l0_gb200_multi_gpus", 1, 1, 4],
        "GB10-PyTorch-Post-Merge-1": ["gb10x-single", "l0_gb10", 1, 1],
        // Disable GB300 stages due to nodes will be offline temporarily.
        // "GB300-PyTorch-1": ["gb300-single", "l0_gb300", 1, 1],
        // "GB300-4_GPUs-PyTorch-Post-Merge-1": ["gb300-x4", "l0_gb300_multi_gpus", 1, 1, 4],
        // PerfSanity pre-merge tests
        "GB200-4_GPUs-PyTorch-PerfSanity-1": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 1, 1, 4],
        // PerfSanity post-merge tests
        "GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-1": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 1, 3, 4],
        "GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-2": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 2, 3, 4],
        "GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-3": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 3, 3, 4],
    ]
    fullSet += SBSASlurmTestConfigs.keySet()

    multiNodesSBSAConfigs = [
        // Each testcase uses 8 GPUs and 2 nodes.
        // https://nvbugs/5598863 (uncorrectable NVLink error detected during the execution) may not exist in OCI machines.
        "GB200-8_GPUs-2_Nodes-PyTorch-1": ["auto:gb200-flex", "l0_gb200_multi_nodes", 1, 2, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-2": ["auto:gb200-flex", "l0_gb200_multi_nodes", 2, 2, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-1": ["auto:gb200-flex", "l0_gb200_multi_nodes", 1, 3, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-2": ["auto:gb200-flex", "l0_gb200_multi_nodes", 2, 3, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-3": ["auto:gb200-flex", "l0_gb200_multi_nodes", 3, 3, 8, 2],
    ]
    // PerfSanity post-merge aggr tests
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-8_GPUs-2_Nodes-PyTorch-Aggr-PerfSanity-Node2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_aggr_perf_sanity_node2_gpu8",
        5,
        8,
        2
    )
    // PerfSanity post-merge disagg tests
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU1-GEN1-NODE1-GPU4-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_disagg_perf_sanity_ctx1_node1_gpu1_gen1_node1_gpu4",
        1,
        8,
        2
    )
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE1-GPU4-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_disagg_perf_sanity_ctx1_node1_gpu4_gen1_node1_gpu4",
        3,
        8,
        2
    )
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-12_GPUs-3_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU1-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_disagg_perf_sanity_ctx1_node1_gpu1_gen1_node2_gpu8",
        1,
        12,
        3
    )
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-12_GPUs-3_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_disagg_perf_sanity_ctx1_node1_gpu4_gen1_node2_gpu8",
        5,
        12,
        3
    )
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-16_GPUs-4_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE2-GPU8-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_disagg_perf_sanity_ctx1_node2_gpu8_gen1_node2_gpu8",
        1,
        16,
        4
    )
    fullSet += multiNodesSBSAConfigs.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        parallelJobs = SBSATestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "arm64"), {
            runLLMTestlistOnPlatform(pipeline, values[0], values[1], LINUX_AARCH64_CONFIG, false, key, values[2], values[3])
        }]]}

        // Add SBSA Slurm jobs
        parallelSlurmJobs = SBSASlurmTestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, "slurm", "arm64"), {
            def config = LINUX_AARCH64_CONFIG
            if (key.contains("single-device")) {
                config = SINGLE_DEVICE_CONFIG
            }
            if (key.contains("llvm")) {
                config = LLVM_CONFIG
            }
            runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3], values[4] ?: 1, values[5] ?: 1, values[6] ?: false)
        }]]}
        parallelJobs += parallelSlurmJobs

        // Add SBSA multi node Slurm jobs
        parallelMultiNodesSBSAJobs = multiNodesSBSAConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, "slurm", "arm64"), {
            def config = LINUX_AARCH64_CONFIG
            if (key.contains("single-device")) {
                config = SINGLE_DEVICE_CONFIG
            }
            if (key.contains("llvm")) {
                config = LLVM_CONFIG
            }
            runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3], values[4] ?: 1, values[5] ?: 2, values[6] ?: false)
        }]]}

        parallelJobs += parallelMultiNodesSBSAJobs
    }

    docBuildSpec = createKubernetesPodConfig(LLM_DOCKER_IMAGE, "a10")
    docBuildConfigs = [
        "A10-Build_Docs": [docBuildSpec, {
            sh "rm -rf **/*.xml *.tar.gz"
            runLLMDocBuild(pipeline, config=VANILLA_CONFIG)
        }],
    ]

    fullSet += docBuildConfigs.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        docBuildConfigs = [:]
    }

    docBuildJobs = docBuildConfigs.collectEntries{key, values -> [key, [values[0], {
        stage("[${key}] Run") {
            cacheErrorAndUploadResult("${key}", values[1], {}, true)
        }
    }]]}

    // Python version and OS for sanity check
    x86SanityCheckConfigs = [
        "PY312-DLFW": [
            LLM_DOCKER_IMAGE,  // Workaround ABI incompatibilities between PyTorch 2.9.1 and 2.10.0a0
            "B200_PCIe",
            X86_64_TRIPLE,
            false,
            "dlfw/",
            DLFW_IMAGE,
            false,
        ],
        "PY310-UB2204": [
            LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE,
            "A10",
            X86_64_TRIPLE,
            true,
            "",
            UBUNTU_22_04_IMAGE,
            true, // Extra install PyTorch CUDA 13.0 package to align with the CUDA version used for building TensorRT LLM wheels.
        ],
        "PY312-UB2404": [
            LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE,
            "RTX5090",
            X86_64_TRIPLE,
            true,
            "",
            UBUNTU_24_04_IMAGE,
            true, // Extra PyTorch CUDA 13.0 install
        ],
    ]

    aarch64SanityCheckConfigs = [
        // Workaround PyTorch 2.9.1 vs. 2.10.0a0 incompatibility issue. Once resolved, change back to:
        // 1. DLFW_IMAGE -> UBUNTU_24_04_IMAGE
        // 2. Extra PyTorch CUDA install: false -> true
        "PY312-UB2404": [
            LLM_DOCKER_IMAGE,
            "GH200",
            AARCH64_TRIPLE,
            false,
            "",
            DLFW_IMAGE,
            false, // Extra PyTorch CUDA 13.0 install
        ],
        "PY312-DLFW": [
            LLM_DOCKER_IMAGE,
            "GH200",
            AARCH64_TRIPLE,
            false,
            "dlfw/",
            DLFW_IMAGE,
            false,
        ],
    ]

    def toStageName = { gpuType, key -> "${gpuType}-PackageSanityCheck-${key}".toString() }
    fullSet += x86SanityCheckConfigs.collectEntries{ key, values -> [toStageName(values[1], key), null] }.keySet()
    fullSet += aarch64SanityCheckConfigs.collectEntries{ key, values -> [toStageName(values[1], key), null] }.keySet()

    sanityCheckConfigs = x86SanityCheckConfigs
    if (env.targetArch == AARCH64_TRIPLE) {
        sanityCheckConfigs = aarch64SanityCheckConfigs
    }

    sanityCheckJobs = sanityCheckConfigs.collectEntries {key, values -> [toStageName(values[1], key), {
        cacheErrorAndUploadResult(toStageName(values[1], key), {
            def cpu_arch = values[2]
            def gpu_type = values[1].toLowerCase()
            if (values[1] == "B200_PCIe") {
                gpu_type = "b100-ts2"
            }
            if (values[1] == "RTX5090") {
                gpu_type = "rtx-5090"
            }

            def k8s_arch = "amd64"
            if (cpu_arch == AARCH64_TRIPLE) {
                k8s_arch = "arm64"
            }

            def buildSpec = createKubernetesPodConfig(values[0], "build", k8s_arch)
            def buildRunner = runInKubernetes(pipeline, buildSpec, "trt-llm")
            def sanityRunner = null


            def sanitySpec = createKubernetesPodConfig(values[0], gpu_type, k8s_arch)
            sanityRunner = runInKubernetes(pipeline, sanitySpec, "trt-llm")

            def wheelPath = "${values[4]}"
            def wheelName = ""
            def cpver = "cp312"
            def pyver = "3.12"
            if (key.contains("PY310")) {
                cpver = "cp310"
                pyver = "3.10"
            }

            buildRunner("[${toStageName(values[1], key)}] Build") {
                def env = []
                if (key.contains("manylinux")) {
                    env = ["LD_LIBRARY_PATH+=:/usr/local/cuda/compat"]
                }
                withEnv(env) {
                    wheelName = runLLMBuild(pipeline, cpu_arch, values[3], wheelPath, cpver)
                }
            }

            def fullWheelPath = "${cpu_arch}/${wheelPath}${wheelName}"

            // TODO: Re-enable the sanity check after updating GPU testers' driver version.
            // sanityRunner("Sanity check") {
            //     runPackageSanityCheck(pipeline, fullWheelPath, values[3], cpver)
            // }

            def checkPipStage = false
            if (cpu_arch == X86_64_TRIPLE) {
                checkPipStage = true
            } else if (cpu_arch == AARCH64_TRIPLE) {
                checkPipStage = true
            }

            if (checkPipStage) {
                stage("Run LLMAPI Test") {
                    pipInstallSanitySpec = createKubernetesPodConfig(values[5], gpu_type, k8s_arch)
                    trtllm_utils.launchKubernetesPod(pipeline, pipInstallSanitySpec, "trt-llm", {
                        echo "###### Prerequisites Start ######"
                        echoNodeAndGpuInfo(pipeline, toStageName(values[1], key))
                        // Clean up the pip constraint file from the base NGC PyTorch image.
                        if (values[5] == DLFW_IMAGE) {
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt || true")
                        }
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y python3-pip git rsync curl wget")
                        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 config set global.break-system-packages true")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install requests")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 uninstall -y tensorrt")
                        if (values[5] != DLFW_IMAGE) {
                            def ubuntu_version = key.contains("UB2404") ? "ubuntu2404" : "ubuntu2204"
                            def platform = cpu_arch == X86_64_TRIPLE ? "x86_64" : "sbsa"
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_version}/${platform}/cuda-keyring_1.1-1_all.deb")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "dpkg -i cuda-keyring_1.1-1_all.deb")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y cuda-toolkit-13-1")
                        }
                        // Extra PyTorch CUDA 13.0 install for all bare-metal environments (Default PyTorch is for CUDA 12.8)
                        if (values[6]) {
                            echo "###### Extra PyTorch CUDA 13.0 install Start ######"
                            // Use internal mirror instead of https://download.pytorch.org/whl/cu130 for better network stability.
                            // PyTorch CUDA 13.0 package and torchvision package can be installed as expected.
                            if (k8s_arch == "amd64") {
                                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install torch==2.9.1+cu130 torchvision==0.24.1+cu130 --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/pytorch-cu128-remote/simple")
                            } else {
                                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install torch==2.9.1+cu130 torchvision==0.24.1 --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/pytorch-cu128-remote/simple")
                            }
                        }

                        def libEnv = []
                        if (env.alternativeTRT) {
                            stage("Replace TensorRT") {
                                trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
                            }
                            libEnv += ["LD_LIBRARY_PATH+tensorrt=/usr/local/tensorrt/lib"]
                            libEnv += ["LD_LIBRARY_PATH+nvrtc=/usr/local/lib/python${pyver}/dist-packages/nvidia/cuda_nvrtc/lib"]
                        }
                        echo "###### Check pip install Start ######"
                        withEnv(libEnv) {
                            // Retry 2 times if timeout occurs.
                            sh "env | sort"
                            trtllm_utils.llmRetry(1, "checkPipInstall", {
                                timeout(time: 30, unit: 'MINUTES') {
                                    checkPipInstall(pipeline, "${cpu_arch}/${wheelPath}")
                                }
                            })
                        }
                        echo "###### Run LLMAPI tests Start ######"

                        def config = VANILLA_CONFIG
                        if (cpu_arch == AARCH64_TRIPLE) {
                            config = LINUX_AARCH64_CONFIG
                        }
                        withEnv(libEnv) {
                            sh "env | sort"
                            runLLMTestlistOnPlatform(pipeline, gpu_type, "l0_sanity_check", config, false, toStageName(values[1], key), 1, 1, true, null, "-SubJob-RunTest")
                        }
                    })
                }
            }
        }, {}, true)
    }]}

    multiGpuJobs = parallelJobs.findAll{(it.key =~ /\d+_GPUs/) && !it.key.contains("Post-Merge")}
    println multiGpuJobs.keySet()
    multiGpuJobsPostMerge = parallelJobs.findAll{(it.key =~ /\d+_GPUs/) && it.key.contains("Post-Merge")}

    parallelJobs += docBuildJobs
    parallelJobs += sanityCheckJobs

    postMergeJobs = parallelJobs.findAll {it.key.contains("Post-Merge")}

    // Start as a normal pre-merge job
    parallelJobsFiltered = parallelJobs - multiGpuJobs - postMergeJobs

    // Check if the multi GPU related file has changed or not. If changed, add multi GPU test stages.
    if (testFilter[(MULTI_GPU_FILE_CHANGED)]) {
        parallelJobsFiltered += multiGpuJobs
    }

    if (testFilter[(AUTO_TRIGGER_TAG_LIST)]) {
        echo "AUTO_TRIGGER_TAG_LIST mode is true. Auto trigger tags: ${testFilter[(AUTO_TRIGGER_TAG_LIST)].join(', ')}."
        def autoTriggerTagStages = [:]
        for (tag in testFilter[(AUTO_TRIGGER_TAG_LIST)]) {
            autoTriggerTagStages += parallelJobs.findAll { it.key.contains(tag) }
        }
        parallelJobsFiltered += autoTriggerTagStages
        if (autoTriggerTagStages.size() > 0) {
            echo "Auto trigger will force run stages: ${autoTriggerTagStages.keySet().join(', ')}."
        }
        println parallelJobsFiltered.keySet()
    }

    // Check --post-merge, post-merge or TRT dependency testing pipelines.
    // If true, add post-merge only test stages and multi-GPU test stages.
    if (env.alternativeTRT || testFilter[(IS_POST_MERGE)]) {
        parallelJobsFiltered += multiGpuJobs
        parallelJobsFiltered += postMergeJobs
    }

    // Check --skip-test, only run doc build and sanity check stages.
    if (testFilter[(ENABLE_SKIP_TEST)]) {
        echo "All test stages are skipped."
        parallelJobsFiltered = docBuildJobs + sanityCheckJobs
    }

    // Check --add-multi-gpu-test, if true, add multi-GPU test stages back.
    if (testFilter[(ADD_MULTI_GPU_TEST)]) {
        parallelJobsFiltered += multiGpuJobs
    }

    // Check --only-multi-gpu-test, if true, only run multi-GPU test stages.
    if (testFilter[(ONLY_MULTI_GPU_TEST)]) {
        if (testFilter[(IS_POST_MERGE)]) {
            parallelJobsFiltered = multiGpuJobsPostMerge
        } else {
            parallelJobsFiltered = multiGpuJobs
        }
    }

    // Check --disable-multi-gpu-test, if true, remove multi-GPU test stages.
    if (testFilter[(DISABLE_MULTI_GPU_TEST)]) {
        parallelJobsFiltered -= multiGpuJobs
    }

    // Check --gpu-type, filter test stages.
    if (testFilter[(GPU_TYPE_LIST)] != null) {
        echo "Use GPU_TYPE_LIST for filtering. GPU types: ${testFilter[(GPU_TYPE_LIST)]}."
        parallelJobsFiltered = parallelJobsFiltered.findAll {it.key.tokenize('-')[0] in testFilter[(GPU_TYPE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    // Check --backend-mode, filter test stages.
    if (testFilter[(TEST_BACKEND)] != null) {
        echo "Use TEST_BACKEND for filtering. Backend mode: ${testFilter[(TEST_BACKEND)]}."
        def backendMode = testFilter[(TEST_BACKEND)].collect { it.toLowerCase() }
        def changeMap = [
            "pytorch": "-PyTorch-",
            "tensorrt": "-TensorRT-",
            "cpp": "-CPP-",
            "triton": "-Triton-",
            "fmha": "-FMHA-",
        ]
        def backendModeList = backendMode.collect { changeMap.get(it) }.flatten()
        def parallelJobsNoBackend = parallelJobsFiltered.findAll { key, _ ->
            !changeMap.values().any { backend -> key.contains(backend) }
        }
        def parallelJobsBackendMode = parallelJobsFiltered.findAll { key, _ ->
            backendModeList.any { backend -> key.contains(backend) }
        }
        parallelJobsFiltered = parallelJobsNoBackend + parallelJobsBackendMode
        echo "parallelJobsBackendMode: ${parallelJobsBackendMode.keySet()}"
        println parallelJobsFiltered.keySet()
    }

    if (testFilter[(ONLY_ONE_GROUP_CHANGED)] == "Docs") {
        echo "Only docs files are changed, run doc build stage only."
        parallelJobsFiltered = docBuildJobs
        println parallelJobsFiltered.keySet()
    } else if (testFilter[(ONLY_ONE_GROUP_CHANGED)] != "") {
        if (testFilter[(TEST_BACKEND)] != null) {
            echo "Force disable ONLY_ONE_GROUP_CHANGED mode. Backend mode set by flag: ${testFilter[(TEST_BACKEND)]}."
        } else {
            echo "ONLY_ONE_GROUP_CHANGED mode is true. The group is: ${testFilter[(ONLY_ONE_GROUP_CHANGED)]}."
            def excludedBackends = new HashMap()
            excludedBackends["PyTorch"] = ["-CPP-", "-TensorRT-", "-FMHA-"]     // Only pytorch file change also need to run triton tests
            excludedBackends["Triton"] = ["-PyTorch-", "-CPP-", "-TensorRT-", "-FMHA-"]
            excludedBackends["FMHA"] = ["-PyTorch-", "-CPP-", "-TensorRT-", "-Triton-"]
            def group = testFilter[(ONLY_ONE_GROUP_CHANGED)]
            if (excludedBackends.containsKey(group)) {
                parallelJobsFiltered = parallelJobsFiltered.findAll { key, value ->
                    !excludedBackends[group].any { backend -> key.contains(backend) }
                }
            }
            println parallelJobsFiltered.keySet()
        }
    }

    // Check --stage-list, only run the stages in stage-list.
    if (testFilter[TEST_STAGE_LIST] != null) {
        echo "Use TEST_STAGE_LIST for filtering. Stages: ${testFilter[(TEST_STAGE_LIST)]}."
        parallelJobsFiltered = parallelJobs.findAll {it.key in testFilter[(TEST_STAGE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    // Check --extra-stage, add the stages in extra-stage.
    if (testFilter[EXTRA_STAGE_LIST] != null) {
        echo "Use EXTRA_STAGE_LIST for filtering. Stages: ${testFilter[(EXTRA_STAGE_LIST)]}."
        parallelJobsFiltered += parallelJobs.findAll {it.key in testFilter[(EXTRA_STAGE_LIST)]}
        println parallelJobsFiltered.keySet()
    }

    checkStageName(fullSet)

    if (testFilter[(TEST_STAGE_LIST)] != null) {
        checkStageNameSet(testFilter[(TEST_STAGE_LIST)], fullSet, TEST_STAGE_LIST)
    }
    if (testFilter[(EXTRA_STAGE_LIST)] != null) {
        checkStageNameSet(testFilter[(EXTRA_STAGE_LIST)], fullSet, EXTRA_STAGE_LIST)
    }

    echo "Check the passed GitLab bot testFilter parameters."
    def keysStr = parallelJobsFiltered.keySet().join(",\n")
    pipeline.echo "Now we will run stages: [\n${keysStr}\n]"

    parallelJobsFiltered = parallelJobsFiltered.collectEntries { key, values -> [key, {
        stage(key) {
            if (key in testFilter[REUSE_STAGE_LIST]) {
                stage("Skip - Reused") {
                    echo "Skip - Passed in the previous pipelines."
                }
            } else if (values instanceof List) {
                trtllm_utils.launchKubernetesPod(pipeline, values[0], "trt-llm", {
                    values[1]()
                })
            } else {
                values()
            }
        }
    }]}

    return parallelJobsFiltered
}



def launchTestJobsForImagesSanityCheck(pipeline, globalVars) {
    def testConfigs = [
        "NGC Devel Image amd64": [
            name: "NGC-Devel-Image-amd64-Sanity-Test",
            k8sArch: "amd64",
            wheelInstalled: false,
            config: VANILLA_CONFIG,
        ],
        "NGC Devel Image arm64": [
            name: "NGC-Devel-Image-arm64-Sanity-Test",
            k8sArch: "arm64",
            wheelInstalled: false,
            config: LINUX_AARCH64_CONFIG,
        ],
        "NGC Release Image amd64": [
            name: "NGC-Release-Image-amd64-Sanity-Test-A10",
            gpuType: "a10",
            k8sArch: "amd64",
            wheelInstalled: true,
            config: VANILLA_CONFIG,
        ],
        "NGC Release Image arm64": [
            name: "NGC-Release-Image-arm64-Sanity-Test-GH200",
            gpuType: "gh200",
            k8sArch: "arm64",
            wheelInstalled: true,
            config: LINUX_AARCH64_CONFIG,
        ],
    ]
    if (!ENABLE_NGC_DEVEL_IMAGE_TEST) {
        ["NGC Devel Image amd64", "NGC Devel Image arm64"].each { key ->
            testConfigs.remove(key)
        }
        echo "NGC Devel Image test is disabled."
    }
    if (!ENABLE_NGC_RELEASE_IMAGE_TEST) {
        ["NGC Release Image amd64", "NGC Release Image arm64"].each { key ->
            testConfigs.remove(key)
        }
        echo "NGC Release Image test is disabled."
    }
    // Update testConfigs image field using the map from globalVars
    testConfigs.each { key, config ->
        if (globalVars[IMAGE_KEY_TO_TAG] && globalVars[IMAGE_KEY_TO_TAG][key]) {
            config.image = globalVars[IMAGE_KEY_TO_TAG][key]
        }
    }
    // Filter out all configs that don't have image set
    testConfigs = testConfigs.findAll { key, config ->
        return config.image != null
    }

    echo "Filtered test configs with images:"
    println testConfigs

    def testJobs = testConfigs.collectEntries { key, values -> [values.name, {
        if (values.wheelInstalled) {
            stage(values.name) {
                echo "Run ${values.name} sanity test."
                imageSanitySpec = createKubernetesPodConfig(values.image, values.gpuType, values.k8sArch)
                trtllm_utils.launchKubernetesPod(pipeline, imageSanitySpec, "trt-llm", {
                    sh "env | sort"
                    trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y git rsync curl")
                    runLLMTestlistOnPlatform(pipeline, values.gpuType, "l0_sanity_check", values.config, false, values.name, 1, 1, true, null, "-SubJob-TestImage")
                })
            }
        } else {
            stage(values.name) {
                imageSanitySpec = createKubernetesPodConfig(values.image, "build", values.k8sArch)
                trtllm_utils.launchKubernetesPod(pipeline, imageSanitySpec, "trt-llm", {
                    sh "env | sort"
                    def cpuArch = values.k8sArch == "amd64" ? X86_64_TRIPLE : AARCH64_TRIPLE
                    runLLMBuild(pipeline, cpuArch, false, "imageTest/")
                })
            }
        }
    }]}

    return testJobs
}


pipeline {
    agent {
        kubernetes createKubernetesPodConfig("", "agent")
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
        //Workspace normally is: /home/jenkins/agent/workspace/LLM/L0_MergeRequest@tmp/
        HF_HOME="${env.WORKSPACE_TMP}/.cache/huggingface"
        CCACHE_DIR="${CCACHE_DIR}"
        GITHUB_MIRROR="https://urm.nvidia.com/artifactory/github-go-remote"
        PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
        // force datasets to be offline mode, to prevent CI jobs are downloading HF dataset causing test failures
        HF_DATASETS_OFFLINE=1
        CMAKE_POLICY_VERSION_MINIMUM="3.5"
        OPEN_SEARCH_DB_BASE_URL=credentials("open_search_db_base_url")
        OPEN_SEARCH_DB_CREDENTIALS=credentials("open_search_db_credentials")
    }
    stages {
        stage("Setup Environment")
        {
            steps
            {
                script {
                    echo "enableFailFast is: ${params.enableFailFast}"
                    echo "env.testFilter is: ${env.testFilter}"
                    testFilter = trtllm_utils.updateMapWithJson(this, testFilter, env.testFilter, "testFilter")
                    println testFilter
                    echo "env.globalVars is: ${env.globalVars}"
                    globalVars = trtllm_utils.updateMapWithJson(this, globalVars, env.globalVars, "globalVars")
                    globalVars[ACTION_INFO] = trtllm_utils.setupPipelineDescription(this, globalVars[ACTION_INFO])
                }
            }
        }
        stage("Check Test List")
        {
            when {
                expression {
                    // Only run the test list validation when necessary
                    env.targetArch == X86_64_TRIPLE &&
                    testFilter[ONLY_ONE_GROUP_CHANGED] != "Docs" &&
                    !(env.JOB_NAME ==~ /.*Multi-GPU.*/) &&
                    !(env.JOB_NAME ==~ /.*BuildDockerImageSanityTest.*/)
                }
            }
            steps
            {
                script {
                    launchTestListCheck(this)
                }
            }
        }
        stage("Test") {
            steps {
                script {
                    if (env.JOB_NAME ==~ /.*BuildDockerImageSanityTest.*/) {
                        parallelJobs = launchTestJobsForImagesSanityCheck(this, globalVars)
                    } else {
                        parallelJobs = launchTestJobs(this, testFilter)
                    }

                    singleGpuJobs = parallelJobs
                    dgxJobs = [:]

                    def testPhase2StageName = env.testPhase2StageName
                    if (testPhase2StageName) {
                        def multiGpuPattern = /\d+_GPUs/
                        singleGpuJobs = parallelJobs.findAll{!(it.key =~ multiGpuPattern)}
                        dgxJobs = parallelJobs.findAll{it.key =~ multiGpuPattern}
                    }

                    if (env.JOB_NAME ==~ /.*Single-GPU.*/) {
                        echo "Only run single-GPU tests."
                        if (dgxJobs.size() > 0) {
                            if (globalVars[ACTION_INFO]['parents'].size() > 0) {
                                // We add a special marker to the parent job's description.
                                // This will be used to decide whether to run multi-GPU test stage.
                                def parentJob = globalVars[ACTION_INFO]['parents'][-2]
                                def archStr = (env.targetArch == X86_64_TRIPLE) ? "x86_64" : (env.targetArch == AARCH64_TRIPLE ? "SBSA" : "Unknown")
                                trtllm_utils.appendBuildDescription(this, parentJob['name'], parentJob['build_number'], "====Require ${archStr} Multi-GPU Testing====<br/>")
                            } else {
                                echo "No parent job found to add the special marker for executing multi-GPU test stage."
                            }
                        } else {
                            echo "Skip multi-GPU testing. No test to run."
                        }
                        if (singleGpuJobs.size() > 0) {
                            singleGpuJobs.failFast = params.enableFailFast
                            parallel singleGpuJobs
                        } else {
                            echo "Skip single-GPU testing. No test to run."
                        }
                    } else if (env.JOB_NAME ==~ /.*Multi-GPU.*/) {
                        echo "Only run multi-GPU tests."
                        if (dgxJobs.size() > 0) {
                            dgxJobs.failFast = params.enableFailFast
                            parallel dgxJobs
                        } else {
                            error "Skip multi-GPU testing. No test to run."
                        }
                    } else {
                        if (singleGpuJobs.size() > 0) {
                            singleGpuJobs.failFast = params.enableFailFast
                            parallel singleGpuJobs
                        } else {
                            echo "Skip single-GPU testing. No test to run."
                        }

                        if (dgxJobs.size() > 0) {
                            stage(testPhase2StageName) {
                                dgxJobs.failFast = params.enableFailFast
                                parallel dgxJobs
                            }
                        }
                    }
                }
            }
        } // Test stage
    } // stages
} // pipeline
