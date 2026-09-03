/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.InterruptedException
import groovy.transform.Field
import groovy.json.JsonOutput
import com.nvidia.bloom.KubernetesManager
import com.nvidia.bloom.Constants
import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.SlurmCluster
import com.nvidia.bloom.SlurmPartition
import com.nvidia.bloom.Utils
import com.nvidia.bloom.ContainerRuntime
import com.nvidia.bloom.SshAuthMethod
import org.jenkinsci.plugins.workflow.cps.CpsThread
import org.jsoup.Jsoup
import org.jenkinsci.plugins.pipeline.modeldefinition.Utils as jUtils
import trtllm.FailureClassifier
import trtllm.ContextDeath
import trtllm.exceptions.InfraFailure
import trtllm.exceptions.PipelineInterruption
import trtllm.exceptions.TrtllmCiException
import trtllm.exceptions.UserFailure

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}
LLM_ROOT = "llm"

ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
UPLOAD_PATH = env.uploadPath ? env.uploadPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
URM_ARTIFACTORY_BASE = "https://urm.nvidia.com/artifactory"
ENABLE_UPLOAD_TEST_RESULTS = params.enableUploadTestResults != null ? params.enableUploadTestResults : true

X86_64_TRIPLE = "x86_64-linux-gnu"
AARCH64_TRIPLE = "aarch64-linux-gnu"

// default package name
linuxPkgName = ( env.targetArch == AARCH64_TRIPLE ? "tensorrt-llm-sbsa-release-src-" : "tensorrt-llm-release-src-" ) + (env.artifactCommit ? env.artifactCommit : env.gitlabCommit) + ".tar.gz"

// Container configuration
// available tags can be found in: https://artifactory.nvidia.com/artifactory/sw-tensorrt-llm-docker-local/tensorrt-llm/
// [base_image_name]-[arch]-[os](-[python_version])-[trt_version]-[torch_install_type]-[stage]-[date]-[mr_id]
LLM_DOCKER_IMAGE = env.dockerImage
X86_64_DOCKER_IMAGE = LLM_DOCKER_IMAGE.replace("aarch64", "x86_64").replace("sbsa", "x86_64")
LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE = env.wheelDockerImagePy310
LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE = env.wheelDockerImagePy312
LLM_WHEEL_DOCKER_IMAGE = env.wheelDockerImage

// K8s secret in namespace sw-tensorrt for pulling from artifactory.nvidia.com
ARTIFACTORY_IMAGE_PULL_SECRET = "trtllm-artifactory"
ARTIFACTORY_DOCKER_HOST = "artifactory.nvidia.com"
// Read-only artifactory pull credentials
ARTIFACTORY_CREDENTIALS_ID = "trtllm-artifactory-credentials"

// DLFW torch image
DLFW_IMAGE = "urm.nvidia.com/docker/nvidia/pytorch:26.05-py3"

MODEL_EXPRESS_VERSION = "0.4.1"
MODEL_EXPRESS_NIXL_VERSION = "1.4.0"
MODEL_EXPRESS_SERVER_IMAGE = "urm.nvidia.com/docker/nvidia/ai-dynamo/modelexpress-server:${MODEL_EXPRESS_VERSION}"
MODEL_EXPRESS_REDIS_IMAGE = "urm.nvidia.com/docker/redis:7-alpine"

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
def INFRA_DRY_RUN_TEST_CONTEXT = "infra_dry_run"

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

TESTER_CPU_ONLY_CORES = "2"
TESTER_CPU_ONLY_MEMORY = "12Gi"

CCACHE_DIR="/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"
MODEL_CACHE_DIR="/scratch.trt_llm_data/llm-models"

KITMAKER_CREDENTIALS_ID = env.kitmakerCredentialsId ? env.kitmakerCredentialsId : "svc_tensorrt_kitmaker_api_token"
KITMAKER_DRY_RUN_PIC_EMAIL = env.kitmakerDryRunPicEmail ? env.kitmakerDryRunPicEmail : "kefu@nvidia.com"
KITMAKER_PUBLISH_TO = env.kitmakerPublishTo ? env.kitmakerPublishTo : "both_devzone_pypi"
RELEASE_SCRIPT_REPO = env.releaseScriptRepo ? env.releaseScriptRepo.trim() : ""
RELEASE_SCRIPT_COMMIT = env.releaseScriptCommit ? env.releaseScriptCommit.trim() : ""

// GPU types that require open driver
REQUIRED_OPEN_DRIVER_TYPES = ["b100-ts2", "rtx-5080", "rtx-5090", "rtx-pro-6000", "rtx-pro-6000d"]

// GPU types that don't support dynamic driver flashing
REQUIRED_NO_DRIVER_TYPES = ["dgx-h100", "dgx-h200", "gh200", "gb10x"]

// Maximum SLURM infra-failure retries (total attempts = SLURM_INFRA_RETRY_MAX + 1).
// Recognised failure patterns are tagged with scope=SLURM or BOTH in the
// shared-lib PATTERN_CATALOG.
//
// Capped at 1 (2 attempts) because each SLURM attempt that times out at the
// partition walltime burns ~4 hours, and the dominant retryable failure
// (AgentOfflineException after test ran for hours) is a test-timeout
// proxy — not a transient blip worth multi-retry. Cross-layer total is also
// capped at 2: SLURM dispatcher pods pass singleAttempt:true to the outer
// K8s pod retry, so 1 pod × 2 SLURM attempts = 2 total.
SLURM_INFRA_RETRY_MAX = 1

// Maximum K8s infra-failure retries (total attempts = K8S_INFRA_RETRY_MAX + 1).
// Kept distinct from SLURM_INFRA_RETRY_MAX so the two paths can be tuned
// independently as production telemetry comes in. Patterns tagged scope=K8S
// or BOTH in the shared-lib PATTERN_CATALOG.
//
// Capped at 1 (2 attempts) so a non-SLURM K8s stage runs at most 2 pods.
// For SLURM dispatcher pods this budget is bypassed (singleAttempt:true)
// to avoid nesting with the inner SLURM retry.
K8S_INFRA_RETRY_MAX = 1

// Infra-scoped fail-fast master switch. When true, a branch whose post-retry
// failure classifies as a positive infra abort (via
// FailureClassifier.isDeferrableInfra) is recorded and swallowed -- its sibling
// branches keep running instead of being SIGTERMed by failFast -- and a sub-job
// that saw only infra aborts (no genuine failure) resolves to UNSTABLE. When
// false, every failure rethrows and the original bare-boolean fail-fast is fully
// restored. Kept separate from params.enableFailFast so the scoped behavior can
// be disabled pipeline-wide without turning fail-fast itself off. Both K8s-scoped
// and SLURM-scoped aborts are deferred: runBranchesWithInfraDefer classifies each
// branch under its real execution scope (SLURM dispatcher stages under both K8S
// and SLURM, every other stage under K8S).
//
// Overridable without a code change by setting the ENABLE_INFRA_SCOPED_FAILFAST
// env var on the job. Env values are strings ("false" is truthy in Groovy), so
// the override goes through toBoolean() rather than the bare elvis.
ENABLE_INFRA_SCOPED_FAILFAST = env.ENABLE_INFRA_SCOPED_FAILFAST ? env.ENABLE_INFRA_SCOPED_FAILFAST.toBoolean() : true

// Per-stage override of the above: set `infraRetryMax` in a stage's opts map (the
// 3rd element of its parallel-jobs config tuple, alongside singleAttempt) to cap
// or disable stage-level infra retries for resource-scarce hardware pools --
// `infraRetryMax: 0` disables retries entirely (1 attempt). It may only reduce the
// budget: values above the scope global are clamped down to it (resolveInfraRetryMax),
// so it can never increase retries past these caps. It applies to whichever
// stage-level retry the stage uses: the SLURM retry (runLLMTestlistOnSlurm) for
// dispatcher pods, or the K8s pod retry (runKubernetesPodWithInfraRetry) for regular
// test pods. It does NOT touch the dispatcher-pod launch-retry (relaunching a cheap
// Blossom pod doesn't tax the scarce hardware). Null/absent = use the globals above.

// Fallback discriminator for SLURM timeouts.
// If we can't reach the SLURM node for an authoritative reason,
// we apply a heuristic: if the job needed more than this of its budget
// to fail, we treat it as a timeout.
SLURM_TIMEOUT_RETRY_FRACTION = 0.9

// SLURM states in which the job is still alive (no terminal verdict yet). When a
// monitor/agent exception surfaces while the job is in one of these, the failure
// is a transient infra blip (lost SSH/agent) -- not a test result -- so the stage
// should retry rather than defer an opaque exception to the classifier (which
// could mistake it for a test failure and not retry). Mirrors the active-state
// set the sbatch resubmit guard reuses an existing job on.
SLURM_NON_TERMINAL_STATES = [
    "RUNNING", "PENDING", "CONFIGURING", "COMPLETING",
    "REQUEUED", "RESIZING", "SUSPENDED", "SIGNALING", "STOPPED",
]

// How often the background shell watcher snapshots the in-progress
// ${stageName}/ directory (containing the PeriodicJUnitXML reporter's
// results.xml) up to Artifactory. 10 minutes balances Artifactory traffic
// against staleness for multi-hour test stages.
PROGRESS_UPLOAD_INTERVAL_SEC = 600

// Typed-exception hierarchy and FailureClassifier (PATTERN_CATALOG, classify(),
// flattenThrowable) live in trtllm-jenkins-shared-lib under src/trtllm/. They
// were originally inline here, but the Jenkins script-security sandbox
// requires per-instance signature approval for `RuntimeException(String,
// Throwable)` and `Throwable.getSuppressed()` when those operations execute
// inside an inline pipeline script. Code under src/ in a shared library runs
// outside the sandbox, so the same operations work without approval.
//
// See trtllm-jenkins-shared-lib for the implementation; the imports at the
// top of this file pull them in.

// ENABLE_NGC_DEVEL_IMAGE_TEST is currently disabled in the Jenkins BuildDockerImageSanityTest job config
ENABLE_NGC_DEVEL_IMAGE_TEST = params.enableNgcDevelImageTest ?: false
ENABLE_NGC_RELEASE_IMAGE_TEST = params.enableNgcReleaseImageTest ?: false

COMMON_SSH_OPTIONS = Utils.DEFAULT_CUSTOM_SSH_OPTIONS

// Per-stage CBTS coverage exclusions applied on top of the upstream eligibility decision.
CBTS_EXCLUDE_STAGES = [] as Set

def isInfraDryRun() {
    return testFilter[(INFRA_DRY_RUN)] ?: false
}

def isCbtsStage(String stageName) {
    // Pipeline-level eligibility (post-merge gate + kill switch) is decided in L0_MergeRequest.groovy and propagated via testFilter.
    if (!(testFilter[(CBTS_COVERAGE)] ?: false)) {
        return false
    }
    // Perf stages skip coverage to avoid skewing measurements (perfMode == stageName.contains("Perf")).
    if (stageName.contains("Perf")) {
        return false
    }
    // Skip stages with no product Python coverage: TensorRT (legacy), CPP (gtest), AutoDeploy (leaving L0).
    if (stageName.contains("TensorRT") || stageName.contains("CPP") || stageName.contains("AutoDeploy")) {
        return false
    }
    // Phase 1: single-GPU only; multi-GPU / multi-node stages carry the "_GPUs" / "_Nodes" token.
    if (stageName.contains("_GPUs") || stageName.contains("_Nodes")) {
        return false
    }
    return !CBTS_EXCLUDE_STAGES.contains(stageName)
}

def freezeCbtsCoverage(String stageDir) {
    // Stop .cbtscov writes and wait for the dir to settle before the results tar reads it.
    sh(
        returnStatus: true,
        script: """
            mkdir -p ${stageDir}
            touch ${stageDir}/${CBTS_STOP_FILE_NAME}
            prev=\$(stat -c %y ${stageDir} 2>/dev/null || echo unknown)
            for i in \$(seq 1 15); do
                sleep 1
                cur=\$(stat -c %y ${stageDir} 2>/dev/null || echo unknown)
                [ "\$cur" = "\$prev" ] && break
                prev=\$cur
            done
        """
    )
}

def scpFromRemoteCmd(Map remote, String remotePath, String localPath) {
    String portOpt = remote.port ? "-P ${remote.port} " : ""
    if (remote.privateKeyPath) {
        return "scp -i ${remote.privateKeyPath} ${portOpt}-r -p ${COMMON_SSH_OPTIONS} ${remote.user}@${remote.host}:${remotePath} ${localPath}"
    }
    return "sshpass -p '${remote.passwd}' scp ${portOpt}-r -p ${COMMON_SSH_OPTIONS} ${remote.user}@${remote.host}:${remotePath} ${localPath}"
}

// Print the last `lines` lines of the remote Slurm job log file to the console when the Slurm job fails.
// If the log file does not exist, print a message to the console.
def echoRemoteLogTail(def pipeline, Map remote, String remotePath, int lines = 200) {
    pipeline.echo("===== Last ${lines} lines of ${remotePath} on ${remote.host} =====")
    try {
        def tailOut = Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(remote,
                "\"bash -c 'if [ -f \\\"${remotePath}\\\" ]; then tail -n ${lines} -- \\\"${remotePath}\\\"; " +
                "else echo \\\"[log not found: ${remotePath}]\\\"; fi'\""),
            returnStdout: true,
            numRetries: 1,
        )?.trim()
        pipeline.echo(tailOut ?: "")
    } catch (InterruptedException e) {
        throw e
    } catch (Exception tailEx) {
        pipeline.echo("Ignorable warning: could not tail ${remotePath} on ${remote.host}: ${tailEx.message}")
    }
}

// Scrape the SLURM job output log for a device / driver / interconnect fault
// signature and return the matched signature itself, or "" for no match.
//
// Device faults (CUDA/NVLink/ECC/driver) print into job-output.log but never
// reach the stage exception chain -- the tracker squashes a failed job to
// `exit 1` -- so classify() otherwise sees only a generic failure and cannot
// steer the retry off the bad node. This is a GATE only: the returned signature
// is folded into a fresh exception so FailureClassifier.PATTERN_CATALOG (the
// authoritative list) makes the real retry/severity decision. A signature the
// catalog does not recognize simply falls through to a normal rethrow.
// App-induced CUDA errors (illegal memory access, unspecified launch failure,
// OOM) are deliberately excluded -- the OpenSearch stage data shows those are
// overwhelmingly code regressions, not node faults, and must not trigger a
// node-avoiding retry.
//
// grep -o returns only the matched signature (not the whole line), so a long
// log line cannot truncate the signature out of the result before it reaches
// classify(). Each alternative must therefore be catalog-exact: it must match
// (via `.` wildcards for shell-hostile chars) the full catalog substring, so
// grep -o emits text that still contains the catalog pattern.
def scrapeSlurmLogForDeviceFault(def pipeline, Map remote, String remoteLogPath) {
    def deviceFaultRegex = "cudaErrorMapBufferObjectFailed|mapping of buffer object failed|" +
        "uncorrectable NVLink error|cudaErrorNvlinkUncorrectable|CUDA_ERROR_SYSTEM_NOT_READY|" +
        "uncorrectable ECC error|CUDA_ERROR_ECC_UNCORRECTABLE|has fallen off the bus|GPU is lost|" +
        "Unable to determine the device handle for GPU|RmInitAdapter failed|Failed to initialize NVML|" +
        "could... communicate with the NVIDIA driver|CUDA_ERROR_DEVICE_UNAVAILABLE|" +
        "no CUDA-capable device is detected|CUDA_ERROR_UNKNOWN: 999|CUDA unknown error|" +
        "CUDA-capable device.s. is/are busy or unavailable"
    try {
        // Wrap the body in `bash -c` so it is shell-agnostic: cluster login shells
        // are often csh/tcsh, which can't parse this bash test/pipe/redirection
        // syntax. The login shell only has to run `bash -c '<single-quoted body>'`.
        return Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(remote,
                "\"bash -c 'if [ -f \\\"${remoteLogPath}\\\" ]; then grep -aioE \\\"${deviceFaultRegex}\\\" \\\"${remoteLogPath}\\\" 2>/dev/null | tail -n 1 | cut -c1-500; fi'\""),
            returnStdout: true,
            numRetries: 1,
        )?.trim()
    } catch (InterruptedException e) {
        throw e
    } catch (Exception scrapeEx) {
        pipeline.echo("Ignorable warning: could not scrape ${remoteLogPath} for device faults on ${remote.host}: ${scrapeEx.message}")
        return ""
    }
}

// Read back the sbatch output the submit script captured to sbatch_output.txt
// in the job workspace. A failed sh step surfaces only the exit code ("script
// returned exit code 1"), so sbatch's stderr -- e.g. "Slurm backup controller
// in standby mode" during a slurmctld failover -- never reaches
// FailureClassifier.classify(), which matches the exception chain only. Like
// scrapeSlurmLogForDeviceFault above, this is a GATE only: the caller folds
// the returned text into a fresh exception and the shared-lib PATTERN_CATALOG
// (the authoritative list) makes the real retry/severity decision. Returns ""
// when the file is missing or unreadable, so the caller can rethrow the
// original exception unchanged.
def readSlurmSubmitOutput(def pipeline, Map remote, String jobWorkspace, String stageName) {
    def outputPath = "${jobWorkspace}/sbatch_output.txt"
    try {
        return Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(remote,
                "\"bash -c 'if [ -f \\\"${outputPath}\\\" ]; then head -c 1000 -- \\\"${outputPath}\\\"; fi'\""),
            returnStdout: true,
            numRetries: 1,
        )?.trim()
    } catch (InterruptedException e) {
        throw e
    } catch (Exception readEx) {
        // A dead frontend must propagate so the enclosing withSlurmFrontendFailover
        // fails over to another remote; swallowing it as "" would return an empty
        // read to the caller, which then rethrows the original (generic) submit
        // exception and strands the stage on the unreachable frontend. Any other
        // read failure (missing file, transient) is non-fatal -- the caller
        // rethrows the original exception unchanged.
        if (CloudManager.isSlurmFrontendConnectionFailure(readEx)) {
            throw readEx
        }
        pipeline.echo("Ignorable warning: could not read ${outputPath} on ${remote.host}: ${readEx.message}")
        return ""
    }
}

// `postTag` uniquifies the uploaded tar filename, the Artifactory guard key and
// the locally-staged result XMLs when the same stageName is uploaded more than
// once in a build (e.g. SLURM infra-failure retries). First attempt passes "".
def uploadResults(def pipeline, SlurmCluster cluster, String clusterName, String nodeName, String stageName, String postTag="", boolean suppressTestReporting=false) {
    pipeline.stage('Submit Test Result') {
        sh "ls -al ${stageName}/ || true"

        if (suppressTestReporting) {
            // This attempt is superseded by a planned retry. Rename its XMLs so
            // Collect Test Result does not ingest them after extracting the tar.
            // The progress snapshot still contains the old names, so it must not
            // be promoted and the modified directory must be uploaded instead.
            sh """
                cd ${stageName} && for f in results*.xml; do
                    [ -e "\$f" ] && mv "\$f" "superseded-\$f"
                done || true
            """
        }

        // Promote progress tar to final path, or fall back to direct upload.
        // progress_upload_snapshot.sh writes the sentinel on each successful PUT.
        ensureStageResultNotUploaded("${stageName}${postTag}")
        if (suppressTestReporting || !promoteProgressTar(stageName, postTag)) {
            if (suppressTestReporting) {
                echo "[PROGRESS-UPLOAD] ${stageName}: results*.xml changed on disk, re-uploading instead of promoting progress tar"
            } else {
                // Progress upload never succeeded (Artifactory unreachable, watcher not started, etc.).
                echo "[PROGRESS-UPLOAD] ${stageName}: no successful progress upload recorded, falling back to direct upload"
            }
            // Fall back to the original approach: tar the local stage directory
            // and upload it directly. Use --transform so tar contents carry the
            // postTag filename without touching on-disk results*.xml files.
            def xmlCount = sh(script: "ls ${stageName}/results*.xml 2>/dev/null | wc -l", returnStdout: true).trim().toInteger()
            if (suppressTestReporting || xmlCount > 0) {
                def transformOpt = postTag ? "--transform 's|^\\(${stageName}/results[^/]*\\)\\.xml\$|\\1${postTag}.xml|'" : ""
                sh "tar -czvf results-${stageName}${postTag}.tar.gz ${transformOpt} ${stageName}/"
                trtllm_utils.uploadArtifacts(
                    "results-${stageName}${postTag}.tar.gz",
                    "${UPLOAD_PATH}/test-results/"
                )
            } else {
                println("No results xml to submit")
            }
        }

        // Pull this stage's per-process .cbtscov files as one archive into ${stageName}/cbts/; bounded and non-fatal.
        if (isCbtsStage(stageName)) {
            CloudManager.withSlurmSshCredentials(pipeline, clusterName, cluster) { remote ->
                def remoteWs = "/home/svc_tensorrt/bloom/scripts/${nodeName}"
                def cbtsLocalDir = "${stageName}/cbts"
                def cbtsArchive = "cbts_coverage_${stageName}.tar.gz"
                sh "mkdir -p ${cbtsLocalDir}"
                try {
                    timeout(time: 5, unit: 'MINUTES') {
                        // Freeze writers, then pack on the login node; an archive with no member is dropped.
                        Utils.exec(
                            pipeline,
                            script: Utils.sshUserCmd(
                                remote,
                                "\"cd '${remoteWs}' && touch '${CBTS_STOP_FILE_NAME}' && sleep 2 && " +
                                "tar czf '${cbtsArchive}' .cbtscov.${stageName}* 2>/dev/null; " +
                                "tar tzf '${cbtsArchive}' 2>/dev/null | grep -q . || rm -f '${cbtsArchive}'\""
                            ),
                            returnStatus: true,
                            numRetries: 3,
                        )
                        def gotArchive = Utils.exec(
                            pipeline,
                            script: scpFromRemoteCmd(remote, "${remoteWs}/${cbtsArchive}", "${cbtsLocalDir}/"),
                            returnStatus: true,
                            numRetries: 3,
                        ) == 0
                        if (gotArchive) {
                            sh "tar xzf ${cbtsLocalDir}/${cbtsArchive} -C ${cbtsLocalDir}/ && rm -f ${cbtsLocalDir}/${cbtsArchive}"
                        } else {
                            echo "CBTS: no coverage archive retrieved for ${stageName} (no coverage data or transfer skipped)."
                        }
                    }
                } catch (Exception e) {
                    echo "CBTS: coverage pull for ${stageName} skipped (${e.message}); continuing."
                }
            }
        }
    }

    // For a reportable attempt, rename local results*.xml with postTag so
    // junit() reports each attempt separately. Superseded attempts were already
    // renamed above and therefore do not match this loop.
    if (postTag) {
        sh """
            cd ${stageName}
            for f in results*.xml; do
                [ -f "\$f" ] || continue
                case "\$f" in *${postTag}.xml) continue ;; esac
                mv "\$f" "\${f%.xml}${postTag}.xml" || true
            done
        """
    }

    // junit() uses local XML files already downloaded by the final progress snapshot.
    def hasLocalResults = sh(script: "ls ${stageName}/results*.xml 2>/dev/null | wc -l", returnStdout: true).trim().toInteger() > 0
    if (hasLocalResults && !suppressTestReporting) {
        junit(allowEmptyResults: true, testResults: "${stageName}/results*.xml")
    } else if (suppressTestReporting) {
        echo "[INFRA-RETRY] ${stageName}${postTag}: suppressing junit() because a retry is still planned"
    }
}

def runIsolatedTests(pipeline, preprocessedLists, testCmdLine, llmSrc, stageName, postTag="") {
    // Run the isolated tests one by one to avoid any potential conflicts
    def isolateTestList = preprocessedLists.isolate
    def isolateTestLines = readFile(file: isolateTestList).readLines()
    def rerunFailed = false
    def hasUnrerunFailure = false

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

        try {
            sh """
                cd ${llmSrc}/tests/integration/defs && \
                ${isolateTestCmdLine.join(" ")}
            """
        } catch (InterruptedException e) {
            throw e
        } catch (Exception e) {
            def isRerunFailed = rerunFailedTests(
                stageName, llmSrc, isolateTestCmdLine, "results_isolated_${i}.xml", "isolated_${i}", postTag)
            if (isRerunFailed) {
                catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                    error "Isolated test ${i} (${isolateTestName}) failed after rerun attempt"
                }
                // Mark that at least one isolated test failed, but continue processing other tests
                rerunFailed = true
            } else {
                // unfinished_test.txt is shared across the whole stage, so match
                // by this test's own name instead of just checking file presence.
                def unfinishedTestFile = "${WORKSPACE}/${stageName}/unfinished_test.txt"
                def isTestUnfinished = fileExists(unfinishedTestFile) &&
                    sh(script: "grep -qF -- '${isolateTestName}' ${unfinishedTestFile}", returnStatus: true) == 0
                if (isTestUnfinished) {
                    // Record this crash as a JUnit <testcase> like the regular-test
                    // path does. hasUnrerunFailure stays untouched here: it drives
                    // the duration/no-signature message below, which doesn't apply.
                    generateTimeoutTestResultXml(pipeline, stageName)
                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                        error "Isolated test ${i} (${isolateTestName}) terminated unexpectedly, please check the test report."
                    }
                } else if (fileExists("${WORKSPACE}/${stageName}/rerun/isolated_${i}/rerun_0.txt")) {
                    // Same duration/no-signature gap as the regular-test path: this
                    // finished but failed, and was never actually rerun, so
                    // results_isolated_${i}.xml still carries the original
                    // <failure> with nothing here to flag it.
                    catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                        error "Isolated test ${i} (${isolateTestName}) failed and was not eligible for rerun (duration > 10 min, no matching failure signature)"
                    }
                    hasUnrerunFailure = true
                }
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
    if (hasUnrerunFailure) {
        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
            error "One or more isolated tests failed and were not eligible for rerun, please check the test report."
        }
    }

    return [rerunFailed: rerunFailed, hasUnrerunFailure: hasUnrerunFailure]
}

def getInfraDryRunPytestTargets(testListPath) {
    if (!isInfraDryRun()) {
        return []
    }

    // --test-list filters after collection, so also pass the exact rendered
    // nodeid positionally to avoid importing unrelated product tests.
    def targets = readFile(file: testListPath).readLines()
        .collect { it.trim().split(/\s+/, 2)[0] }
        .findAll { it.contains("::") }
    def expectedTarget =
        "test_infra_dry_run_benchmark.py::test_infra_dry_run_benchmark"
    if (targets != [expectedTarget]) {
        error "Unexpected pytest targets in infrastructure dry-run list ${testListPath}: ${targets}"
    }
    return targets
}

def processShardTestList(llmSrc, testDBList, splitId, splits, perfMode=false, durationsPath="") {
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
            "--group ${splitId}",
        ]
        if (durationsPath) {
            testListCmd += ["--durations-path ${durationsPath}"]
        }
        testListCmd += getInfraDryRunPytestTargets(cleanedTestDBList)

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

// SLURM job IDs are digit-only. Not String.isNumber(): that is a
// BigDecimal-style parse, so values like "1.5" pass and later feed
// invalid IDs into scancel/sacct/scontrol.
def isValidSlurmJobId(def slurmJobID) {
    return slurmJobID && slurmJobID.toString() ==~ /\d+/
}

def cleanUpSlurmResources(def pipeline, SlurmCluster cluster, String clusterName, String jobUID){
    CloudManager.withSlurmFrontendFailover(pipeline, clusterName, cluster) { remote ->
        def jobWorkspace = "/home/svc_tensorrt/bloom/scripts/${jobUID}"
        def s3SpoolRoot = "/home/svc_tensorrt/bloom/scripts/.s3-spool-${jobUID}"

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

        if (!isValidSlurmJobId(slurmJobID)) {
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
            // .sqsh is shared across jobs (named by image digest), so age-prune
            // instead of deleting per job; reused images keep a refreshed mtime.
            "find ${cluster.scratchPath}/users/svc_tensorrt/containers -maxdepth 1 -name 'container-*.sqsh' -mtime +3 -delete 2>/dev/null || true",
            "find ${cluster.scratchPath}/users/svc_tensorrt/containers -maxdepth 1 \\( -name 'container-*.tmp' -o -name 'container-*.lock' \\) -mtime +1 -delete 2>/dev/null || true",
            "rm -rf ${jobWorkspace} ${s3SpoolRoot} || true",
        ].join(" ; ")
        Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(
                remote,
                Utils.bashWrappedRemoteCmd(cleanupCommands)
            )
        )

        Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID} cleaned up")
    }
}

// Methods to run Slurm job with Jenkins Agent
def cleanUpNodeResources(def pipeline, SlurmCluster cluster, String clusterName, String nodeName, String slurmJobID) {
    Utils.exec(pipeline, script: "echo Sleeping to allow docker stop; sleep 30")

    CloudManager.destroyNode(nodeName)

    Utils.exec(pipeline, script: "echo Sleeping to allow node destruction; sleep 30")

    CloudManager.withSlurmFrontendFailover(pipeline, clusterName, cluster) { remote ->
        // A missing/non-numeric ID means the job was never submitted (or its ID
        // never captured); running the dump anyway executes `scancel null` /
        // `scontrol show job null`, whose "Invalid job id specified" output then
        // shows up in failure analysis as a bogus error signature.
        if (!isValidSlurmJobId(slurmJobID)) {
            Utils.exec(pipeline, script: "echo \"No SLURM job ID captured for node ${nodeName}; skipping job cleanup dump\"")
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

        def entrypoint = SlurmConfig.containerRuntimeToEntrypoint[cluster.containerRuntime]
        def cleanupCommands = [
            "rm -rf /home/svc_tensorrt/bloom/scripts/agent-${nodeName}.jar /home/svc_tensorrt/bloom/scripts/${nodeName}-${entrypoint} || true",
            "rm -rf ${cluster.scratchPath}/users/svc_tensorrt/enroot-config-${nodeName} || true",
            // .sqsh is shared across jobs (named by image digest), so age-prune
            // instead of deleting per job; reused images keep a refreshed mtime.
            "find ${cluster.scratchPath}/users/svc_tensorrt/containers -maxdepth 1 -name 'container-*.sqsh' -mtime +3 -delete 2>/dev/null || true",
            "find ${cluster.scratchPath}/users/svc_tensorrt/containers -maxdepth 1 \\( -name 'container-*.tmp' -o -name 'container-*.lock' \\) -mtime +1 -delete 2>/dev/null || true",
        ].join(" ; ")
        Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(
                remote,
                Utils.bashWrappedRemoteCmd(cleanupCommands)
            )
        )

        Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID} cleaned up")
    }
}

// ---- Off-pod SLURM resource reconciliation --------------------------------
// A SLURM stage runs inside a K8s dispatcher pod that ssh-drives the job on the
// login node. If that pod dies mid-run (eviction, container error, agent
// offline), the in-pod cleanup can no longer reach the controller, so the SLURM
// job and any Jenkins agent node leak. State + reconciliation live in the shared
// `resourceLedger` (trtllm-jenkins-shared-lib). Each SLURM stage registers two
// sibling entries -- "<stage>/dispatcher-pod" (the pod spec, from the pod
// wrapper) and "<stage>/slurm" (the per-attempt SLURM job / Jenkins node, from
// the stage body) -- so freeing the job/node after a successful in-pod cleanup
// (markReclaimed) never disturbs the pod spec a *later* attempt's off-pod
// reconciliation needs. The ledger stores only serializable primitives (it
// rejects anything else) so pipeline persistence is unaffected; the SlurmCluster
// is rebuilt from clusterName at reconcile time. The functions below are thin
// SLURM-specific adapters over that generic ledger.
//
// Shared-library contract used below (trtllm-jenkins-shared-lib), documented here
// for equivalence with the pre-shared inlined finalizer this replaced:
//   * ContextDeath.isContextDeath(e) -- true when the flattened cause + suppressed
//     chain contains a dispatcher-pod-death signal: "pod failed (reason:", "pod
//     just failed", "pod failed because container terminated", "unable to create
//     live filepath". Same pattern set the inlined isDispatcherPodFailure matched,
//     now also traversing suppressed exceptions' own cause chains.
//   * register(id, type, fields, ownerBuildTag) -- merges non-null fields into the
//     id's entry; stores serializable primitives only (rejects anything else); a
//     `fields` map may not overwrite the reserved id/type/ownerBuildTag keys.
//   * markReclaimed(id) -- drops the entry (idempotent).
//   * get(id) -- a deep-copied snapshot of the entry, or null.
//   * reconcile(pipeline, select, reclaim) -- for each live entry matching
//     select(entry), runs reclaim(pipeline, entry): deregisters entries whose
//     reclaim returns truthy; leaves live those returning falsy or throwing (a
//     throw is logged, never propagated). This is the equivalence backbone --
//     "reconcile off-pod, deregister on success, leave for the sweep on failure."

// Register a live SLURM resource. The pod wrapper passes a podSpec (recorded
// under "<stage>/dispatcher-pod"); the stage body passes the SLURM job / Jenkins
// node identity (recorded under "<stage>/slurm"). Keyed separately so they are
// reclaimed independently. Absent fields are skipped by the ledger.
void registerSlurmResource(String stageName, Map fields) {
    if (!stageName) {
        return
    }
    if (fields.containsKey('podSpec')) {
        resourceLedger.register(id: "${stageName}/dispatcher-pod", type: 'k8sDispatcherPod',
            fields: [podSpec: fields.podSpec, containerName: fields.containerName],
            ownerBuildTag: env.BUILD_TAG)
    } else {
        resourceLedger.register(id: "${stageName}/slurm",
            type: fields.usedSbatch ? 'slurmJob' : 'slurmNode',
            fields: [clusterName: fields.clusterName, nodeName: fields.nodeName,
                     jobUID: fields.jobUID, slurmJobId: fields.slurmJobId, usedSbatch: fields.usedSbatch],
            ownerBuildTag: env.BUILD_TAG)
    }
}

// Called once an attempt's resources are actually torn down: drop the per-attempt
// SLURM job/node entry. The stage's dispatcher-pod entry is a separate id and is
// left intact (a later attempt's off-pod reconciliation still needs its pod spec).
void deregisterSlurmResource(String stageName) {
    if (!stageName) {
        return
    }
    resourceLedger.markReclaimed("${stageName}/slurm")
}

// Reclaim one orphaned SLURM entry off the (dead) dispatcher pod: launch a fresh
// short-lived pod and run the normal cleanup from there (scancel the job, clean
// the workspace, and -- agent path -- drop the leaked Jenkins node). Returns true
// when the entry is reconciled (so resourceLedger deregisters it) and false when
// it cannot be (left for the post-build sweep / manual cleanup). The pod spec and
// container come from the sibling "<stage>/dispatcher-pod" entry; an in-catch
// caller may pass podSpecOverride to use the failing attempt's spec directly.
// Best-effort: a launch/cleanup failure is logged and the entry left live; it
// never masks the stage's own failure. Used as the reclaim body for both the
// in-catch finalize and the post-build sweep.
def reconcileSlurmResource(pipeline, def entry, def podSpecOverride = null) {
    // The ledger is shared build-wide; only SLURM job/node entries are
    // reconcilable here. Never touch (or deregister) another subsystem's entry a
    // selector might pass in -- return false to leave it live and untouched.
    if (entry.type != 'slurmJob' && entry.type != 'slurmNode') {
        return false
    }
    // Pod died before any job/node was provisioned: nothing to reconcile.
    if (!entry.jobUID && !entry.nodeName) {
        return true
    }
    def podEntry = resourceLedger.get(((entry.id ?: "") as String).replaceFirst(/\/slurm$/, "/dispatcher-pod"))
    def podSpec = podSpecOverride ?: podEntry?.podSpec
    def containerName = podEntry?.containerName ?: "trt-llm"
    def cluster = entry.clusterName ? SlurmConfig.clusterConfig[entry.clusterName] : null
    if (!podSpec || !cluster) {
        echo "[SLURM-FINALIZER] ${entry.id}: cannot reconcile off-pod (missing pod spec or unknown cluster " +
             "'${entry.clusterName}'); SLURM job=${entry.slurmJobId ?: entry.jobUID ?: 'unknown'} " +
             "node=${entry.nodeName ?: 'n/a'} may need manual cleanup."
        return false
    }
    try {
        echo "[SLURM-FINALIZER] ${entry.id}: reconciling orphaned SLURM resources off-pod " +
             "(job=${entry.slurmJobId ?: entry.jobUID}, node=${entry.nodeName ?: 'n/a'})."
        trtllm_utils.launchKubernetesPod(pipeline, podSpec, containerName, {
            if (entry.usedSbatch) {
                cleanUpSlurmResources(pipeline, cluster, entry.clusterName, entry.jobUID)
            } else {
                cleanUpNodeResources(pipeline, cluster, entry.clusterName, entry.nodeName, entry.slurmJobId)
            }
        })
        echo "[SLURM-FINALIZER] ${entry.id}: off-pod reconciliation complete."
        return true
    } catch (Exception e) {
        echo "[SLURM-FINALIZER] ${entry.id}: off-pod reconciliation failed (${e.toString()}); leaving entry for post-build sweep."
        return false
    }
}

// Reconcile this stage's orphaned SLURM job/node off-pod (in-catch, on a detected
// dispatcher-pod death). podSpecOverride is the failing attempt's pod spec so
// reconciliation never depends on the sibling entry surviving.
def finalizeOrphanedSlurmResource(pipeline, String stageName, def podSpecOverride = null) {
    if (!stageName) {
        return
    }
    resourceLedger.reconcile(pipeline,
        { it.id == "${stageName}/slurm" },
        { p, entry -> reconcileSlurmResource(p, entry, podSpecOverride) })
}

// Post-build backstop: reconcile any SLURM job/node still registered at the end of
// the build (a dispatcher-pod death whose in-catch finalize also failed, or a
// failure mode the catch never saw). Runs off-pod from a fresh cleanup pod; inert
// dispatcher-pod-only entries are skipped by the selector.
def sweepOrphanedSlurmResources(pipeline) {
    // Scope to this subsystem's SLURM job/node entries: the ledger is shared
    // build-wide, so filter by type as well as a live job/node handle rather than
    // passing any entry that merely happens to carry a jobUID/nodeName field.
    resourceLedger.reconcile(pipeline,
        { (it.type == 'slurmJob' || it.type == 'slurmNode') && (it.jobUID || it.nodeName) },
        { p, entry -> reconcileSlurmResource(p, entry) })
}

// Authoritative timeout signal: ask the SLURM controller (via sacct on the
// cluster login node) for a job's terminal state. Returns the uppercased
// primary state token -- e.g. "TIMEOUT", "COMPLETED", "FAILED", "NODE_FAIL",
// "OUT_OF_MEMORY", "CANCELLED" -- or null when it can't be determined.
//
// Best-effort: any SSH/sacct error returns null so callers fall back to the
// duration heuristic rather than failing the stage.
def querySlurmJobState(def pipeline, SlurmCluster cluster, String clusterName, String slurmJobID) {
    if (!isValidSlurmJobId(slurmJobID)) {
        return null
    }
    String state = null
    try {
        CloudManager.withSlurmFrontendFailover(pipeline, clusterName, cluster) { remote ->
            // -X: allocation row only (skip .batch/.extern steps). -Pn:
            // parsable, no header. First line's first token is the job state;
            // SLURM renders cancellations as "CANCELLED by <uid>", so we keep
            // only the leading token.
            def out = Utils.exec(
                pipeline,
                script: Utils.sshUserCmd(remote, "\"sacct -j ${slurmJobID} --format=State -Pn -X || true\""),
                returnStdout: true,
            )?.trim()
            if (out) {
                state = out.readLines()[0]?.trim()?.tokenize(' ')?.getAt(0)?.toUpperCase(java.util.Locale.ROOT)
            }
        }
    } catch (Exception e) {
        pipeline.echo("[INFRA-RETRY] Could not query SLURM job ${slurmJobID} state via sacct: ${e.message}")
    }
    return state
}

boolean isNonTerminalSlurmState(String state) {
    return state != null && SLURM_NON_TERMINAL_STATES.contains(state.toUpperCase(java.util.Locale.ROOT))
}

def runLLMTestlistWithAgent(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, skipInstallWheel=false, cpver="cp312", String postTag="", boolean useClusterDurations=false, Map placementContext=null, Map retryContext=null)
{
    SlurmPartition partition = SlurmConfig.resolvePlatform(platform)
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]

    // Record which cluster this attempt ran on so a failed node is remembered
    // against its own cluster (auto: platforms pick a cluster at random per attempt).
    if (placementContext != null) {
        placementContext.lastSlurmClusterName = partition.clusterName
    }

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
        CloudManager.withSlurmFrontendFailover(pipeline, partition.clusterName, cluster) { remote ->
            stage('Request Node Via Slurm') {
                println("Selected Cluster: ${cluster.name}")

                def jenkinsSetupPath = Utils.copyLibraryResource(pipeline, entrypoint)

                Utils.exec(pipeline, script: "cat ${jenkinsSetupPath}")

                Utils.copyFileToRemoteHost(pipeline, remote, jenkinsSetupPath, "/home/svc_tensorrt/bloom/scripts/${nodeName}-${entrypoint}", true)

                Utils.exec(pipeline, script: "echo Sleeping before Slurm job submission; sleep \$((RANDOM % 29 + 1))")

                // Enroot needs artifactory auth. Use a per-job config dir (nodeName is
                // unique) — never the shared ~/.config/enroot, which races across jobs.
                // ENROOT_CONFIG_PATH is exported in the sbatch submit shell so Slurm
                // propagates it into the agent setup script (--export=ALL by default).
                def enrootConfigDir = null
                if (cluster.containerRuntime.toString() == "ENROOT") {
                    enrootConfigDir = "${cluster.scratchPath}/users/svc_tensorrt/enroot-config-${nodeName}"
                    withCredentials([usernamePassword(
                        credentialsId: ARTIFACTORY_CREDENTIALS_ID,
                        usernameVariable: 'ARTIFACTORY_USER',
                        passwordVariable: 'ARTIFACTORY_PASSWORD'
                    )]) {
                        def credsLocal = Utils.createTempLocation(pipeline, "./enroot_credentials_slurm-${nodeName}")
                        withEnv([
                            "ENROOT_CREDS_PATH=${credsLocal}",
                            "ARTIFACTORY_DOCKER_HOST=${ARTIFACTORY_DOCKER_HOST}",
                        ]) {
                            Utils.exec(pipeline, script: '''
                                set +x
                                umask 077
                                cat > "$ENROOT_CREDS_PATH" <<EOF
                                machine ${ARTIFACTORY_DOCKER_HOST} login ${ARTIFACTORY_USER} password ${ARTIFACTORY_PASSWORD}
                                EOF
                            '''.replaceAll("\\n\\s*", "\n"))
                        }
                        Utils.exec(pipeline, script: Utils.sshUserCmd(remote, Utils.bashWrappedRemoteCmd("mkdir -p '${enrootConfigDir}'")))
                        Utils.copyFileToRemoteHost(
                            pipeline,
                            remote,
                            credsLocal,
                            "${enrootConfigDir}/.credentials"
                        )
                    }
                }

                def mounts = getMountListForSlurmTest(cluster, false).join(",")
                def imageForSlurm = LLM_DOCKER_IMAGE
                if (cluster.containerRuntime.toString() == "ENROOT") {
                    imageForSlurm = LLM_DOCKER_IMAGE
                        .replace("${ARTIFACTORY_DOCKER_HOST}/", "${ARTIFACTORY_DOCKER_HOST}#")
                }
                def slurmCommand = SlurmConfig.generateCommand(cluster, partition, nodeSecret, nodeName, Jenkins.instance.rootUrl, imageForSlurm, mounts)
                def clusterExcludes = placementContext?.excludedSlurmNodeListsByCluster?.get(partition.clusterName)
                def slurmCommandWithExclusion = trtllm_utils.addSlurmExcludeToCommand(slurmCommand, clusterExcludes)
                def slurmExcludeArg = trtllm_utils.buildSlurmExcludeArg(clusterExcludes)
                if (slurmExcludeArg) {
                    if (slurmCommandWithExclusion != slurmCommand) {
                        echo "[INFRA-RETRY] ${stageName}: requesting SLURM retry placement exclusion: ${slurmExcludeArg}"
                    } else {
                        echo "[INFRA-RETRY] ${stageName}: could not inject ${slurmExcludeArg} into generated SLURM agent command; submitting without node exclusion"
                    }
                }

                def slurmSubmitCommand = slurmCommandWithExclusion
                if (enrootConfigDir) {
                    slurmSubmitCommand = "export ENROOT_CONFIG_PATH='${enrootConfigDir}'; ${slurmCommandWithExclusion}"
                }

                def slurmSubmitOutput = Utils.exec(
                    pipeline,
                    timeout: false,
                    script: Utils.sshUserCmd(
                        remote,
                        Utils.bashWrappedRemoteCmd(slurmSubmitCommand)
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

                // Record the live SLURM job + Jenkins node so a dispatcher-pod death
                // can be reconciled off-pod (the in-pod cleanup can't reach the login
                // node once the pod is gone). Deregistered when cleanup actually runs.
                registerSlurmResource(stageName, [clusterName: partition.clusterName, nodeName: nodeName, slurmJobId: slurmJobID, usedSbatch: false])

                if (!isValidSlurmJobId(slurmJobID)) {
                    echo "Slurm job did not submit successfully. No job ID found.\nSubmission output:\n${slurmSubmitOutput}"
                    // The job never entered the SLURM queue, so nothing ran on any
                    // node: retryable infra by construction. Failing fast here also
                    // keeps the node-wait loop below (up to 15h) from polling a
                    // bogus job ID.
                    throw new InfraFailure(
                        "SLURM agent submission for ${stageName} produced no usable job ID " +
                        "(none or a non-numeric value in the submission output); " +
                        "the job never entered the queue.",
                        null, InfraFailure.TRANSIENT, InfraFailure.SLURM, "<typed:slurm-submit-no-jobid>")
                }
                Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobID}")
                Utils.exec(pipeline, script: "echo Sleeping to allow agent initialization; sleep 30")
            }
        }

        // Wall-clock at which the SLURM job was first observed RUNNING.
        // Captured locally so it remains usable even when the controller
        // is later unreachable (the case the fallback exists for). Null until
        // Phase 1 confirms RUNNING; callers fall back to executeStartMs.
        def jobRunningStartMs = null

        stage('Check If Node Is Online') {
            CloudManager.withSlurmSshCredentialRemotes(pipeline, partition.clusterName, cluster) { remotes ->
                // Check the SLURM job once; if it is no longer active, raise a typed
                // InfraFailure(SLURM) so the retry layer routes it via instanceof (scope=SLURM).
                def checkSlurmJobActive = {
                    try {
                        CloudManager.withSlurmFrontendFailover(pipeline, remotes) { statusRemote ->
                            SlurmConfig.checkJobStatus(pipeline, cluster, slurmJobID, statusRemote)
                        }
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception e) {
                        if (e.message?.contains("is no longer active")) {
                            def slurmLogPath = "/home/svc_tensorrt/slurm-logs/slurm-${slurmJobID}-${nodeName}.out"
                            CloudManager.withSlurmFrontendFailover(pipeline, remotes) { logRemote ->
                                echoRemoteLogTail(pipeline, logRemote, slurmLogPath)
                            }
                            throw new InfraFailure(
                                "${e.message}. Check SLURM logs at ${slurmLogPath} on ${cluster.host}",
                                e, InfraFailure.TRANSIENT, InfraFailure.SLURM, "<typed:slurm-job-inactive>"
                            )
                        }
                        // Otherwise, log the error but continue (SSH might be temporarily unavailable)
                        pipeline.echo("Warning: Could not check SLURM job status: ${e.message}")
                    }
                }

                // Phase 1: wait for the job to leave the queue (PENDING -> RUNNING), polling
                // every 3 min. The whole loop runs in a SINGLE shell step so a long queue wait
                // only adds one flow-node to the Blue Ocean graph (instead of one per iteration,
                // which overflowed the per-stage step cap). Release the held job every 10
                // iterations (~30 min). 300 iterations * 3 min = 15h budget.
                // Exit codes: 0 = job RUNNING, 3 = job no longer active, 4 = timed out.
                def sacctStateCmd = CloudManager.sshUserCmdWithSlurmFrontendFailover(remotes, "\"sacct -j ${slurmJobID} --format=State -Pn --allocations\"")
                def releaseCmd = CloudManager.sshUserCmdWithSlurmFrontendFailover(remotes, "\"scontrol release ${slurmJobID} || true\"")
                def waitRc = pipeline.sh(returnStatus: true, script: """
                    set +e
                    counter=0
                    while [ \$counter -lt 300 ]; do
                        # Avoid the job being stuck in the held state. Release every 10 iterations (~30 min).
                        if [ \$(( counter % 10 )) -eq 0 ]; then
                            ${releaseCmd} || true
                        fi
                        STATE=\$(${sacctStateCmd} | head -1 | cut -d'|' -f1 | awk '{print \$1}')
                        echo "[node-wait] iteration \$counter: SLURM job ${slurmJobID} state='\$STATE'"
                        case "\$STATE" in
                            RUNNING|COMPLETING)
                                echo "[node-wait] SLURM job ${slurmJobID} is running."
                                exit 0
                                ;;
                            PENDING|CONFIGURING|REQUEUED|RESIZING|SUSPENDED|SIGNALING|STOPPED|"")
                                # Still queued, or a transient sacct/ssh hiccup (empty state): keep waiting.
                                ;;
                            *)
                                echo "[node-wait] SLURM job ${slurmJobID} is no longer active (state='\$STATE')."
                                exit 3
                                ;;
                        esac
                        counter=\$(( counter + 1 ))
                        # Wait 3 minutes before checking the job state again.
                        sleep 180
                    done
                    echo "[node-wait] Timed out waiting for SLURM job ${slurmJobID} to start."
                    exit 4
                """)

                // If the job reached a terminal state while queued, confirm via the canonical
                // status check so the exact typed InfraFailure(SLURM) is raised.
                if (waitRc == 3) {
                    checkSlurmJobActive()
                }
                if (waitRc != 0) {
                    error "SLURM job ${slurmJobID} did not reach RUNNING during the queue wait. Terminating the job."
                }

                // Phase 2: job is RUNNING; wait for the Jenkins agent to come online. isNodeOnline()
                // and Thread.sleep() emit no flow-nodes, so poll every 30s without bloating Blue
                // Ocean, and probe job status every ~3 min (every 6th iter) to fail fast if the
                // job dies during bring-up. 60 * 30s = 30 min.
                if (waitRc == 0) {
                    // Job is RUNNING: stamp the walltime-budget origin for the
                    // timeout duration fallback (within Phase 1's ~3min poll
                    // granularity of the true RUNNING transition).
                    jobRunningStartMs = System.currentTimeMillis()
                    def onlineCounter = 0
                    while (!CloudManager.isNodeOnline(nodeName) && onlineCounter < 60) {
                        Thread.sleep(30L * 1000L)
                        if (onlineCounter % 6 == 0) {
                            checkSlurmJobActive()
                        }
                        onlineCounter++
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
                        if (placementContext != null) {
                            placementContext.lastSlurmNodeList = sh(script: "hostname -f || hostname", returnStdout: true).trim()
                            echo "[INFRA-RETRY] ${stageName}: SLURM agent is running on ${placementContext.lastSlurmNodeList}"
                        }
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

                    if (cluster.containerRuntime.toString() == "ENROOT" && slurmJobID) {
                        def setupLogPath = "/home/svc_tensorrt/slurm-logs/slurm-${slurmJobID}-${nodeName}.out"
                        def enrootLog = Utils.exec(
                            pipeline,
                            script: CloudManager.sshUserCmdWithSlurmFrontendFailover(remotes, Utils.bashWrappedRemoteCmd("grep '\\[ENROOT\\]' ${setupLogPath} 2>/dev/null || true")),
                            returnStdout: true,
                            numRetries: 3
                        ).trim()
                        if (enrootLog) {
                            echo "[Enroot setup log]\n${enrootLog}"
                        }
                    }
                } else {
                    def setupLogPath = "/home/svc_tensorrt/slurm-logs/slurm-${slurmJobID}-${nodeName}.out"
                    try {
                        CloudManager.withSlurmFrontendFailover(pipeline, remotes) { logRemote ->
                            echoRemoteLogTail(pipeline, logRemote, setupLogPath)
                        }
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception logEx) {
                        echo "Ignorable warning: could not retrieve ${setupLogPath}: ${logEx.message}"
                    }
                    throw new InfraFailure(
                        "SLURM agent ${nodeName} for job ${slurmJobID} did not come online within 30 minutes " +
                        "after the job started. Check SLURM logs at ${setupLogPath} on ${cluster.host}.",
                        null, InfraFailure.TRANSIENT, InfraFailure.SLURM, "<typed:slurm-agent-online-timeout>"
                    )
                }
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
        long executeStartMs = System.currentTimeMillis()
        // Reclassify a raw test-execution failure against the SLURM job's terminal
        // state: a walltime kill is a UserFailure (not retryable), while a job still
        // alive when the monitor lost contact is transient infra (retry on a fresh
        // node). Kept as a closure so executeLLMTestOnSlurm can apply it INSIDE the
        // task runner -- that way cacheErrorAndUploadResult suppresses this attempt's
        // junit for the SAME typed failure the retry loop acts on. Deciding it only
        // here (after junit already ran) is what left retried-and-passed agent stages
        // UNSTABLE from an intermediate attempt's results.
        // Classify each SLURM test-execution failure exactly once. The closure sets
        // this when it runs (inside the task runner), so the outer catch reclassifies
        // only failures that never reached it (image pull, node/agent bring-up) --
        // avoiding a second querySlurmJobState round trip that could reach a different
        // verdict as elapsed time and the job's state move on.
        boolean slurmFailureClassified = false
        def classifySlurmFailure = { Throwable err ->
            slurmFailureClassified = true
            // A completed-pytest deterministic failure (tests ran, were re-run via
            // --reruns, and still failed) is a real test failure, not lost-contact
            // infra. On the agent path the SLURM allocation outlives pytest, so the
            // job is still RUNNING here even though pytest already finished -- do NOT
            // let the job state below relabel it as slurm-job-still-running and retry
            // it (which masks the failure whenever the retry happens to pass). Key
            // off the propagated rerun-failure error: it is raised only when reruns
            // genuinely failed and takes precedence over the timeout path. Do not key
            // off failed_results.xml -- generateRerunReport writes it from the first
            // run whenever any rerun occurred, even when the rerun passed or the job
            // timed out, so it would wrongly suppress a legitimate infra retry. Defer
            // so the base classifier treats it as a UserFailure (no retry); a monitor-
            // lost-contact cut raises "terminated unexpectedly" instead and retries.
            if ((err?.toString() ?: "").contains("still failed after rerun attempts")) {
                echo "[INFRA-RETRY] ${stageName}: pytest reported deterministic test failure(s); not an infra retry."
                return err
            }
            // Measure elapsed from when the job was first observed RUNNING; fall back
            // to executeStartMs if the RUNNING stamp was never set.
            long timeoutBaselineMs = (jobRunningStartMs ?: executeStartMs) as long
            long elapsedMin = Math.floorDiv(System.currentTimeMillis() - timeoutBaselineMs, 60000L)
            Integer walltimeMin = (partition?.time ?: SlurmConfig.DEFAULT_TIMEOUT_SHORT) as Integer
            def slurmState = querySlurmJobState(pipeline, cluster, partition.clusterName, slurmJobID)

            if (slurmState == "TIMEOUT") {
                return new UserFailure(
                    "SLURM job ${slurmJobID} for ${stageName} ended in state TIMEOUT " +
                    "(hit partition walltime ${walltimeMin}min); treating as a test timeout, not retrying. " +
                    "Original failure: ${err.message}",
                    err)
            }

            if (slurmState == null && walltimeMin != null
                    && elapsedMin >= (long)(SLURM_TIMEOUT_RETRY_FRACTION * walltimeMin)) {
                return new UserFailure(
                    "SLURM job ${slurmJobID} for ${stageName} ran ${elapsedMin}min " +
                    "(>= ${(int)(SLURM_TIMEOUT_RETRY_FRACTION * 100)}% of the ${walltimeMin}min walltime) and " +
                    "the SLURM controller was unreachable for an authoritative state; treating as a likely " +
                    "timeout, not retrying. Original failure: ${err.message}",
                    err)
            }

            if (isNonTerminalSlurmState(slurmState)) {
                return new InfraFailure(
                    "SLURM job ${slurmJobID} for ${stageName} is still in non-terminal state ${slurmState} " +
                    "(${elapsedMin}min of ${walltimeMin}min walltime); the monitor lost contact while the job was " +
                    "alive (transient infra), so this is not a test failure. Original failure: ${err.message}",
                    err, InfraFailure.TRANSIENT, InfraFailure.SLURM, "<typed:slurm-job-still-running>")
            }

            echo "[INFRA-RETRY] ${stageName}: SLURM job ${slurmJobID} terminal state=${slurmState ?: 'unknown'}, " +
                 "ran ${elapsedMin}min of ${walltimeMin}min walltime; deferring to failure classifier."
            return err
        }
        try {
            executeLLMTestOnSlurm(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver, slurmRunner, postTag, useClusterDurations, retryContext, classifySlurmFailure)
        } catch (InterruptedException e) {
            throw e
        } catch (Exception e) {
            // A test-execution failure was already labeled inside the task runner, so
            // this attempt's junit suppression and the retry loop see the same typed
            // failure -- just propagate it. Failures raised outside the task runner
            // (image pull, node/agent bring-up, etc.) were never labeled, so classify
            // them here on the agent.
            throw (slurmFailureClassified || e instanceof TrtllmCiException ? e : classifySlurmFailure(e))
        }
    } finally {
        // Resource cleanup must run even if SLURM metadata capture is interrupted.
        try {
            captureSlurmJobNodeList(pipeline, cluster, partition.clusterName, slurmJobID, placementContext, stageName)
        } finally {
            stage("Clean Up Slurm Resource") {
                // Workaround to handle the interruption during clean up SLURM resources
                retry(3) {
                    try {
                        cleanUpNodeResources(pipeline, cluster, partition.clusterName, nodeName, slurmJobID)
                    } catch (Exception e) {
                        error "Error during clean up SLURM resources: ${e.getMessage()} and retrying."
                    }
                }
            }
            // Cleanup ran on the live pod; drop the registry entry so the off-pod
            // finalizer/sweep does not reconcile already-freed resources.
            deregisterSlurmResource(stageName)
        }
    }
}

def executeLLMTestOnSlurm(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312", runner, String postTag="", boolean useClusterDurations=false, Map retryContext=null, Closure classifySlurmFailure=null)
{
    runner {
        // TODO: refactor the finallyRunner to reuse within slurm or nonslurm job.
        cacheErrorAndUploadResult(stageName, {
            try {
                runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver, postTag, useClusterDurations)
            } catch (InterruptedException e) {
                throw e
            } catch (Exception e) {
                // Label the failure against the SLURM job's terminal state before
                // cacheErrorAndUploadResult decides junit suppression, so a monitor-
                // detected infra failure (e.g. slurm-job-still-running) suppresses this
                // attempt's results and a passing retry stays green. Pipeline aborts
                // (FlowInterruptedException) must not be relabeled.
                if (classifySlurmFailure == null || e.getClass().name.contains("FlowInterruptedException")) {
                    throw e
                }
                throw classifySlurmFailure(e)
            }
        }, {
            // If the execution test list is null, remove the test result xml
            sh """
                ls -al ${stageName}/
                if ! grep -q '<testcase' ${stageName}/results.xml; then
                    rm ${stageName}/results.xml || true
                fi
            """
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def llmSrc = "${llmPath}/${LLM_ROOT}${config}/TensorRT-LLM/src"
            // CPP tests will generate test result in ${llmSrc}/cpp/build_backup/, move these files to job result folder
            sh "ls -al ${llmSrc}/cpp/build_backup/ || true"
            sh "ls -al ${llmSrc}/cpp/build/ || true"
            // Sed for CPP test result
            sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/\" classname=\"/\" classname=\"${stageName}./g' *.xml || true"
            sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/testsuite name=\"[^\"]*\"/testsuite name=\"${stageName}\"/g' *.xml || true"
            // Sed for Pytest result
            sh "cd ${stageName} && sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' *.xml || true"
            // Copy CPP test result
            sh "cp ${llmSrc}/cpp/build_backup/*.xml ${stageName} || true"
            sh "ls -al ${stageName}/"
        }, false, postTag, true, retryContext)
    }
}
// End of Methods to run Slurm job with Jenkins Agent

def getNodeArgs(int nodeCount, int gpuCount, boolean setSegment = false) {
    int gpusPerNode = ((gpuCount / nodeCount) as BigDecimal).setScale(0, BigDecimal.ROUND_CEILING).intValue()
    def args = nodeCount == 1 ? [
        "--nodes=${nodeCount}",
        "--gpus-per-node=${gpuCount}"
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
    String coverageConfigFile,
    String pytestUtil = "",
    List<String> extraArgs = [],
    int containerPortStart = 0,
    int containerPortNum = 0
) {
    def extraInternalEnv = ""
    def pytestTestTimeout = "3600"
    def cbtsMode = isCbtsStage(stageName)

    // TRT uses half of the host logic cores for engine building which is bad for multi-GPU machines.
    extraInternalEnv = "__LUNOWUD=\"-thread_pool_size=${TESTER_CORES}\""
    // CPP test execution is timing out easily, so we always override its internal timeout to the same value as pytest
    extraInternalEnv += " CPP_TEST_TIMEOUT_OVERRIDDEN=${pytestTestTimeout}"
    // Enable NCCL debug information for multi-GPU tests
    extraInternalEnv += " NCCL_DEBUG=INFO"
    // Pass stage name to perf sanity tests for OpenSearch tracking
    extraInternalEnv += " stageName=${stageName}"
    // Persist the AutoTuner profiling cache to a CONTAINER-LOCAL, volatile path so
    // that repeated tactic profiling is reused across testcases within one stage.
    // /tmp lives on the container overlay (srun --no-container-mount-home / fresh
    // pod), so the cache is never written to the host and vanishes when the stage
    // container is destroyed. Never point this at a bind-mounted / shared path:
    // the AutoTuner cache uses fcntl.lockf, which is unreliable over NFS.
    extraInternalEnv += " TLLM_AUTOTUNER_CACHE_PATH=/tmp/trtllm_autotuner_cache/autotuner_cache.json"
    // CBTS stages put cbts_plugin on PYTHONPATH (via ${VAR:-} for set -u safety) plus the marker/config env vars sitecustomize.py reads in subprocesses.
    if (cbtsMode) {
        def cbtsScriptDir = "${llmSrc}/jenkins/scripts/cbts/coverage_utils"
        extraInternalEnv += " PYTHONPATH=${cbtsScriptDir}:\${PYTHONPATH:-}"
        extraInternalEnv += " CBTS_COVERAGE_CONFIG=${coverageConfigFile}"
        extraInternalEnv += " CBTS_MARKER_FILE=${outputPath}/cbts_current_test.txt"
        extraInternalEnv += " CBTS_STOP_FILE=${outputPath}/${CBTS_STOP_FILE_NAME}"
    }

    // Container port allocation environment variables for avoiding port conflicts
    def portEnvVars = ""
    if (containerPortStart > 0 && containerPortNum > 0) {
        portEnvVars = "CONTAINER_PORT_START=${containerPortStart} CONTAINER_PORT_NUM=${containerPortNum}"
    }

    def jUnitLogging = "out-err"
    if (ENABLE_UPLOAD_TEST_RESULTS) {
        jUnitLogging = "all"
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
        "-o junit_logging=${jUnitLogging}",
        // Coverage capture: only CBTS (post-merge) stages instrument, via cbts_plugin / sitecustomize.
        cbtsMode ? "-p cbts_plugin" : "",
        "--periodic-junit",
        "--periodic-junit-xmlpath ${outputPath}/results.xml",
        "--periodic-batch-size=1",
        "--periodic-save-unfinished-test",
        "--periodic-hang-traceback",
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
    def unittestMarkExpr = (stageName.startsWith("CPU-")) ? "cpu_only" : "not cpu_only"
    testCmdLine += ["--unittest-markexpr='${unittestMarkExpr}'"]
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

    // TODO: Add mounts for different cache directories like pip, triton, etc.

    return mounts
}

def runLLMTestlistWithSbatch(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, nodeCount=1, skipInstallWheel=false, cpver="cp312", String postTag="", boolean useClusterDurations=false, Map placementContext=null, Map retryContext=null)
{
    SlurmPartition partition = SlurmConfig.resolvePlatform(platform)
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]

    // Record which cluster this attempt ran on so a failed node is remembered
    // against its own cluster (auto: platforms pick a cluster at random per attempt).
    if (placementContext != null) {
        placementContext.lastSlurmClusterName = partition.clusterName
    }

    // Create a unique suffix for the job name
    String customSuffix = "${env.BUILD_TAG}-${UUID.randomUUID().toString().replaceAll("-", "").substring(0, 6)}".toLowerCase()
    def jobUID = "${cluster.host}-multi_node_test-${customSuffix}"
    def jobWorkspace = "/home/svc_tensorrt/bloom/scripts/${jobUID}"
    def disaggMultiNodeMode = stageName.contains("Disagg-PerfSanity")
    def aggMultiNodeMode = !disaggMultiNodeMode && nodeCount > 1 && stageName.contains("PerfSanity")
    def singleNvlinkDomainMode = stageName.contains("SingleNvlinkDomain")
    def infraDryRun = isInfraDryRun()
    if (infraDryRun) {
        testList = INFRA_DRY_RUN_TEST_CONTEXT
        splitId = 1
        splits = 1
        perfMode = false
    }

    Utils.exec(pipeline, script: "env | sort && pwd && ls -alh")

    def stageIsInterrupted = false
    // Captured so the finally can suppress this attempt's junit when the failure is
    // a retryable infra failure (a retry follows) -- otherwise a stage that fails
    // an intermediate attempt and passes on retry leaves the build UNSTABLE.
    def caughtStageError = null

    try {
        // Run ssh command to start node in desired cluster via SLURM
        withCredentials([
            string(credentialsId: 'TRTLLM_HF_TOKEN', variable: 'HF_TOKEN'),
            string(credentialsId: 'svc_tensorrt-swift-stack-key', variable: 'S3_SECRET_KEY'),
        ]) {
            CloudManager.withSlurmFrontendFailover(pipeline, partition.clusterName, cluster) { remote ->
            def tarName = BUILD_CONFIGS[config][TARNAME]
            def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
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
            def waivesListPathLocal = infraDryRun
                ? "${llmPath}/infra_dry_run_waives.txt"
                : "${llmSrcLocal}/tests/integration/test_lists/waives.txt"
            def slurmJobLogPath = "${jobWorkspace}/job-output.log"
            def scriptLaunchPathLocal = Utils.createTempLocation(pipeline, "./slurm_launch.sh")
            def scriptLaunchPathNode = "${jobWorkspace}/${jobUID}-slurm_launch.sh"
            def scriptSubmitPathLocal = Utils.createTempLocation(pipeline, "./slurm_submit.sh")
            def scriptSubmitPathNode = "${jobWorkspace}/${jobUID}-slurm_submit.sh"
            def scriptTrackPathLocal = Utils.createTempLocation(pipeline, "./slurm_track.sh")
            def scriptTrackPathNode = "${jobWorkspace}/${jobUID}-slurm_track.sh"
            def s3SecretKeyPathLocal = Utils.createTempLocation(pipeline, "./s3_secret_key")
            def s3SecretKeyPathNode = "${jobWorkspace}/s3_secret_key"
            // Per-job enroot config (never shared ~/.config/enroot — that races across jobs).
            def enrootConfigDirNode = "${jobWorkspace}/enroot-config"
            def coverageConfigFile = "${jobWorkspace}/.coveragerc"

            stage("Initialize Test") {
                println("Selected Cluster: ${cluster.name}")
                // Create Job Workspace folder in Frontend Node
                Utils.exec(pipeline, script: Utils.sshUserCmd(remote, "\"mkdir -p ${jobWorkspace}\""), numRetries: 3)

                // Download and Unzip Tar File
                timeout(time: 30, unit: 'MINUTES') {
                    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
                }
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
                String clusterNameForDurations = useClusterDurations ? partition.clusterName.replaceAll('[^a-zA-Z0-9]', '_') : null
                def testListPathLocal = renderTestDB(pipeline, testList, llmSrcLocal, stageName, makoOptsJson, clusterNameForDurations)
                // Copy the test list atomically. A retry that reuses a still-active job
                // re-copies over ${testListPathNode} while that job may be reading it via
                // --test-list; scp truncates-then-streams, so a concurrent read could see a
                // partial list and silently run a subset. Stage to a temp path and mv into
                // place (same-dir rename is atomic) so a reader sees the whole old or new file.
                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    testListPathLocal,
                    "${testListPathNode}.tmp"
                )
                Utils.exec(
                    pipeline,
                    script: Utils.sshUserCmd(remote, "\"mv -f ${testListPathNode}.tmp ${testListPathNode}\"")
                )

                if (infraDryRun) {
                    sh "mkdir -p ${llmPath} && : > ${waivesListPathLocal}"
                } else {
                    // Download and Merge waives.txt
                    mergeWaivesTxt(pipeline, llmSrcLocal, stageName)

                    // Add passed test list from previous pipeline run to the waives.txt
                    if (testFilter[(REUSE_TEST)] != false) {
                        reusePassedTestResults(llmSrcLocal, stageName, waivesListPathLocal, postTag)
                    }
                }

                Utils.copyFileToRemoteHost(
                    pipeline,
                    remote,
                    waivesListPathLocal,
                    waivesListPathNode
                )

                if (ENABLE_UPLOAD_TEST_RESULTS) {
                    pipeline.writeFile(file: s3SecretKeyPathLocal, text: S3_SECRET_KEY)
                    // Preserve the secret file mode through scp -p instead of issuing
                    // an extra remote chmod, which adds another flaky SSH round trip.
                    Utils.exec(pipeline, script: "chmod 600 ${s3SecretKeyPathLocal}")
                    Utils.copyFileToRemoteHost(
                        pipeline,
                        remote,
                        s3SecretKeyPathLocal,
                        s3SecretKeyPathNode
                    )
                }

                if (cluster.containerRuntime.toString() == "ENROOT") {
                    withCredentials([usernamePassword(
                        credentialsId: ARTIFACTORY_CREDENTIALS_ID,
                        usernameVariable: 'ARTIFACTORY_USER',
                        passwordVariable: 'ARTIFACTORY_PASSWORD'
                    )]) {
                        // Unique per stage so parallel stages do not share one @tmp path.
                        def credsLocal = Utils.createTempLocation(pipeline, "./enroot_credentials-${stageName}")
                        withEnv([
                            "ENROOT_CREDS_PATH=${credsLocal}",
                            "ARTIFACTORY_DOCKER_HOST=${ARTIFACTORY_DOCKER_HOST}",
                        ]) {
                            Utils.exec(pipeline, script: '''
                                set +x
                                umask 077
                                cat > "$ENROOT_CREDS_PATH" <<EOF
                                machine ${ARTIFACTORY_DOCKER_HOST} login ${ARTIFACTORY_USER} password ${ARTIFACTORY_PASSWORD}
                                EOF
                            '''.replaceAll("\\n\\s*", "\n"))
                        }
                        Utils.exec(pipeline, script: Utils.sshUserCmd(remote, "\"mkdir -p ${enrootConfigDirNode}\""), numRetries: 3)
                        Utils.copyFileToRemoteHost(
                            pipeline,
                            remote,
                            credsLocal,
                            "${enrootConfigDirNode}/.credentials"
                        )
                    }
                }

                // Generate .coveragerc: CBTS stages render coveragerc.template; all other stages get an empty rcfile (no coverage).
                if (isCbtsStage(stageName)) {
                    // @TRTLLM_WHEEL_PATH@ stays a placeholder; slurm_run.sh substitutes it on the worker.
                    sh """
                        cp ${llmSrcLocal}/jenkins/scripts/cbts/coverage_utils/coveragerc.template ./.coveragerc
                        sed -i \\
                            -e 's|@JOB_WORKSPACE@|${jobWorkspace}|g' \\
                            -e 's|@STAGE_NAME@|${stageName}|g' \\
                            ./.coveragerc
                        cat ./.coveragerc
                    """
                } else {
                    sh "touch ./.coveragerc"
                }

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
                def uploadPath = "${env.JOB_NAME}/${env.BUILD_NUMBER}"

                def clusterDurationsArgsNode = []
                if (useClusterDurations) {
                    def clusterKey = partition.clusterName.replaceAll('[^a-zA-Z0-9]', '_')
                    def clusterDurationsPathNode = "${llmSrcNode}/tests/integration/defs/.test_durations_${clusterKey}"
                    clusterDurationsArgsNode = ["--durations-path ${clusterDurationsPathNode}"]
                }
                def extraArgs = [
                    "--test-list=$testListPathNode",
                    "--splitting-algorithm least_duration",
                    "--splits $splits",
                    "--group $splitId",
                    *clusterDurationsArgsNode,
                ]
                if (ENABLE_UPLOAD_TEST_RESULTS && !testFilter[(DETAILED_LOG)]) {
                    extraArgs += [
                        "--capture=fd",
                        "--s3-upload-path=${uploadPath}/${stageName}",
                        "--s3-upload-mode=deferred",
                    ]
                }
                def pytestCommandParts = getPytestBaseCommandLine(
                    llmSrcNode,
                    stageName,
                    waivesListPathNode,
                    perfMode,
                    jobWorkspace,
                    "$jobWorkspace/.coveragerc",
                    pytestUtil,
                    extraArgs,
                )
                pytestCommandParts += getInfraDryRunPytestTargets(testListPathLocal)
                def pytestCommand = pytestCommandParts.join(" ")

                // Generate Job Launch Script
                def container = LLM_DOCKER_IMAGE
                if (cluster.containerRuntime.toString() == "ENROOT") {
                    container = LLM_DOCKER_IMAGE
                        .replace("${ARTIFACTORY_DOCKER_HOST}/", "${ARTIFACTORY_DOCKER_HOST}#")
                }
                def mounts = getMountListForSlurmTest(cluster, true).join(",")
                String[] taskArgs = getNodeArgs(nodeCount, gpuCount, disaggMultiNodeMode || singleNvlinkDomainMode)
                if (taskArgs == null) {
                    error "Invalid Slurm test stage name is set"
                }
                taskArgs = [
                    *taskArgs,
                ]

                def containerImageArg = container
                def srunPrologue = ""
                if (cluster.containerRuntime.toString() == "ENROOT") {
                    def containerDir = "${cluster.scratchPath}/users/svc_tensorrt/containers"
                    // Name the .sqsh by image digest (not job ID) so jobs sharing
                    // an image reuse one .sqsh instead of re-running `enroot import`
                    // per job. Path is resolved at runtime into ${enrootImagePath}.
                    containerImageArg = "\${enrootImagePath}"

                    srunPrologue = """
                    export ENROOT_CACHE_PATH='/home/svc_tensorrt/.cache/enroot'

                    # Job-private enroot config (under jobWorkspace). Avoids racing on
                    # the shared ~/.config/enroot/.credentials across concurrent jobs.
                    export ENROOT_CONFIG_PATH="${enrootConfigDirNode}"
                    cleanup_enroot_config() { rm -rf "\${ENROOT_CONFIG_PATH}"; }
                    trap cleanup_enroot_config EXIT

                    containerDir="$containerDir"
                    mkdir -p "\$containerDir"
                    # If the image URI already contains a manifest digest (@sha256:…) use
                    # it directly for content-addressed caching; otherwise hash the tag
                    # string, which is stable for per-build images but will not detect a
                    # re-pushed mutable tag within the 3-day TTL window.
                    if printf '%s' "$container" | grep -q '@sha256:'
                    then
                        imageDigest=\$(printf '%s' "$container" | grep -oP '(?<=@sha256:)[a-f0-9]+')
                    else
                        imageDigest=\$(printf '%s' "$container" | sha256sum | cut -d' ' -f1)
                    fi
                    export enrootImagePath="\$containerDir/container-\${imageDigest}.sqsh"

                    importContainerWithRetries() {
                        local docker_uri=\$1
                        local output_path=\$2
                        local max_attempts=\${3:-3}
                        local delay=\${4:-60}
                        local attempt=1
                        local tmp_path

                        # Best-effort lock so racing jobs don't all import the same
                        # image. flock may be a no-op on some shared filesystems;
                        # correctness still holds since the import publishes atomically.
                        exec 9>"\${output_path}.lock" || true
                        flock 9 || true

                        if [ -f "\$output_path" ]
                        then
                            echo "Reusing cached container image: \$output_path"
                            # Refresh mtime so reused images survive age-based pruning.
                            touch "\$output_path" || true
                            flock -u 9 || true
                            return 0
                        fi

                        # Import to a temp path, then mv to publish atomically so
                        # other jobs never see a partial .sqsh.
                        tmp_path="\${output_path}.\${SLURM_JOB_ID}.tmp"
                        rm -f "\$tmp_path"

                        until enroot import -o "\$tmp_path" -- "docker://\$docker_uri"
                        do
                            if ((attempt >= max_attempts))
                            then
                                echo "enroot import failed after \$max_attempts attempts"
                                rm -f "\$tmp_path"
                                flock -u 9 || true
                                return 1
                            fi

                            echo "enroot import failed (attempt \$attempt of \$max_attempts). Retrying in \${delay}s..."
                            rm -f "\$tmp_path"
                            sleep \$delay
                            attempt=\$((attempt + 1))
                        done

                        mv -f "\$tmp_path" "\$output_path"
                        flock -u 9 || true
                    }

                    importContainerWithRetries "$container" "\$enrootImagePath"
                    cleanup_enroot_config
                    trap - EXIT
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

                def srunArgs = [
                    "--container-name=multi_node_test-\${SLURM_JOB_ID}",
                    "--container-image=$containerImageArg",
                    "--container-workdir=$jobWorkspace",
                    "--container-mounts=$mounts",
                    "--no-container-mount-home",
                    "--container-env=NVIDIA_IMEX_CHANNELS"
                ]
                if (ENABLE_UPLOAD_TEST_RESULTS) {
                    srunArgs.add("--container-env=S3_SECRET_KEY")
                }
                envVarsToExport.each { varName, varValue ->
                    srunArgs.add("--container-env=${varName}")
                }

                def exemptionComment = ""
                if (SlurmConfig.needsIdleGpuExemption(cluster)) {
                    exemptionComment = "--comment='${SlurmConfig.IDLE_GPU_EXEMPTION_PAYLOAD}'"
                }
                def slurmExcludeArg = trtllm_utils.buildSlurmExcludeArg(placementContext?.excludedSlurmNodeListsByCluster?.get(partition.clusterName))
                def slurmExcludeDirective = slurmExcludeArg ? "#SBATCH ${slurmExcludeArg}" : ""
                if (slurmExcludeArg) {
                    echo "[INFRA-RETRY] ${stageName}: requesting SLURM retry placement exclusion: ${slurmExcludeArg}"
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
                    ${slurmExcludeDirective}
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
                    ${infraDryRun ? "export infraDryRun=true" : ""}
                    export resourcePathNode=$resourcePathNode
                    export pytestCommand="$pytestCommand"
                    export coverageConfigFile="$coverageConfigFile"
                    export HF_TOKEN=$HF_TOKEN
                    if [ -f "${s3SecretKeyPathNode}" ]; then
                        set +x
                        export S3_SECRET_KEY="\$(cat "${s3SecretKeyPathNode}")"
                        set -x
                    fi
                    export NVIDIA_IMEX_CHANNELS=\${NVIDIA_IMEX_CHANNELS:-0}
                    export NVIDIA_VISIBLE_DEVICES=\${NVIDIA_VISIBLE_DEVICES:-\$(seq -s, 0 \$((\$(nvidia-smi --query-gpu=count -i 0 --format=csv,noheader)-1)))}
                    ${envExportStatements}

                    echo "Env NVIDIA_IMEX_CHANNELS: \$NVIDIA_IMEX_CHANNELS"
                    echo "Env NVIDIA_VISIBLE_DEVICES: \$NVIDIA_VISIBLE_DEVICES"

                    ${srunPrologue}
                """.replaceAll("(?m)^\\s*", "")

                if (!isInfraDryRun() && (disaggMultiNodeMode || aggMultiNodeMode)) {
                    def scriptLaunchPrefixPathLocal = Utils.createTempLocation(pipeline, "./slurm_launch_prefix.sh")
                    def scriptLaunchSrunArgsPathLocal = Utils.createTempLocation(pipeline, "./slurm_srun_args.txt")
                    // The unified submit.py handles both agg and disagg; only the
                    // draft launch script differs between the two paths.
                    def scriptLaunchDraftPathLocal = disaggMultiNodeMode
                        ? "${llmSrcLocal}/jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh"
                        : "${llmSrcLocal}/jenkins/scripts/perf/aggregated/slurm_launch_draft.sh"
                    def scriptSubmitLocalPath = "${llmSrcLocal}/jenkins/scripts/perf/submit.py"

                    srunArgs.removeAll { it == "--mpi=pmi2" || it == "--mpi=pmix" }

                    pipeline.writeFile(file: scriptLaunchPrefixPathLocal, text: scriptLaunchPrefix)
                    pipeline.writeFile(file: scriptLaunchSrunArgsPathLocal, text: srunArgs.join(" "))

                    sh """
                        pip3 install 'pyyaml>=6.0.1,<6.0.3' && \\
                        python3 ${scriptSubmitLocalPath} \\
                        --llm-src ${llmSrcLocal} \\
                        --test-list ${testListPathLocal} \\
                        --draft-launch-sh ${scriptLaunchDraftPathLocal} \\
                        --launch-sh ${scriptLaunchPathLocal} \\
                        --run-sh ${scriptRunPathNode} \\
                        --install-sh ${scriptInstallPathNode} \\
                        --script-prefix ${scriptLaunchPrefixPathLocal} \\
                        --srun-args ${scriptLaunchSrunArgsPathLocal} \\
                        --split-group ${splitId} \\
                        --stage-name ${stageName} \\
                        --cluster-name ${partition.clusterName}
                    """
                } else {
                    if(nodeCount > 1) {
                        srunArgs.add("--mpi=pmix")
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
                if (ENABLE_UPLOAD_TEST_RESULTS) {
                    filesToKeepWhenRetry += [
                        s3SecretKeyPathNode
                    ]
                }
                if (cluster.containerRuntime.toString() == "ENROOT") {
                    filesToKeepWhenRetry += [
                        enrootConfigDirNode
                    ]
                }
                def findKeepWhenRetryArgs = filesToKeepWhenRetry.collect { " ! -name \"\$(basename \"${it}\")\"" }.join("")

                def scriptSubmit = """#!/bin/bash
                    set -xEeuo pipefail
                    trap 'rc=\$?; echo "Error in file \${BASH_SOURCE[0]} on line \$LINENO: \$BASH_COMMAND (exit \$rc)"; exit \$rc' ERR

                    # Reuse an already-active job after an ambiguous frontend disconnect.
                    if [ -f "${jobWorkspace}/slurm_job_id.txt" ]; then
                        previous_job_id=\$(cat "${jobWorkspace}/slurm_job_id.txt")
                        echo "Found previous Slurm job ID: \${previous_job_id}"
                        previous_state=\$(sacct -j "\${previous_job_id}" --format=State -Pn --allocations 2>/dev/null | head -1 | cut -d'|' -f1 | awk '{print \$1}' || true)
                        if [ -z "\${previous_state}" ]; then
                            previous_state=\$(scontrol show job "\${previous_job_id}" 2>/dev/null | tr ' ' '\\n' | sed -n 's/^JobState=//p' | head -1 || true)
                        fi
                        case "\${previous_state}" in
                            RUNNING|PENDING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED|SIGNALING|STOPPED)
                                echo "Reusing active Slurm job \${previous_job_id} in state \${previous_state}"
                                exit 0
                                ;;
                            *)
                                echo "Previous Slurm job \${previous_job_id} is not active (state='\${previous_state:-UNKNOWN}'). Cleaning it up before resubmission."
                                scancel "\${previous_job_id}" || true
                                # Wait for 120 seconds to ensure the previous job is canceled
                                sleep 120
                                ;;
                        esac
                    fi

                    # Clean up workspace: remove all files/dirs not in the keep list
                    find "${jobWorkspace}" -maxdepth 1 -mindepth 1 ${findKeepWhenRetryArgs} -exec rm -rf {} +

                    touch ${slurmJobLogPath}
                    # Capture sbatch's combined output and persist it before acting on
                    # the exit code: a failed sh step carries only the exit code back
                    # to Jenkins, so a submission rejection's stderr (e.g. "Slurm
                    # backup controller in standby mode") would otherwise never reach
                    # the failure classifier. The pipeline reads sbatch_output.txt
                    # back on failure (readSlurmSubmitOutput) and folds it into the
                    # exception it throws.
                    #
                    # Control-plane failover backoff: during a slurmctld failover the
                    # backup controller rejects submissions with "Slurm backup
                    # controller in standby mode" until it promotes, which takes up to
                    # SlurmctldTimeout (upstream default 120s) plus takeover time. The
                    # rejection is instant and burns no partition walltime, so retrying
                    # it here -- every 30s, up to ~3min, inside this one stage attempt
                    # -- rides out a failover without touching the expensive
                    # stage-level SLURM retry budget. This window is sized to a
                    # failover only, NOT to a cluster-wide controller outage or
                    # maintenance: a standby condition that outlasts it falls through
                    # to the failure classifier, whose PATTERN_CATALOG marks
                    # "standby mode" PERSISTENT so the stage does not keep re-running
                    # against a down controller. Any other sbatch failure still fails
                    # immediately.
                    sbatch_max_attempts=6
                    sbatch_attempt=1
                    while true; do
                        sbatch_rc=0
                        sbatch_output=\$(sbatch ${scriptLaunchPathNode} 2>&1) || sbatch_rc=\$?
                        printf '%s\\n' "\${sbatch_output}"
                        printf '%s\\n' "\${sbatch_output}" > "${jobWorkspace}/sbatch_output.txt"
                        if [ "\${sbatch_rc}" -eq 0 ]; then
                            break
                        fi
                        if [ "\${sbatch_attempt}" -lt "\${sbatch_max_attempts}" ] && printf '%s' "\${sbatch_output}" | grep -qi 'Slurm backup controller in standby mode'; then
                            echo "sbatch rejected by a standby controller (attempt \${sbatch_attempt} of \${sbatch_max_attempts}); retrying in 30s."
                            sbatch_attempt=\$((sbatch_attempt + 1))
                            sleep 30
                            continue
                        fi
                        echo "Error: Slurm job submission failed with exit code \${sbatch_rc}."
                        exit "\${sbatch_rc}"
                    done
                    jobId=\$(printf '%s\\n' "\${sbatch_output}" | awk '/Submitted batch job/ {print \$4; exit}')
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
                // Submit the Slurm job. Submit/metadata/track all run on the one
                // frontend the enclosing withSlurmFrontendFailover pinned, so they
                // share the job workspace (slurm_job_id.txt, scripts) on that login
                // node; a frontend disconnect fails the whole closure over to a fresh
                // frontend as a unit (the submit script reuses an active job).
                try {
                    Utils.exec(
                        pipeline,
                        timeout: false,
                        script: Utils.sshUserCmd(remote, scriptSubmitPathNode),
                        numRetries: 3
                    )
                } catch (InterruptedException e) {
                    throw e
                } catch (Exception submitEx) {
                    // Only enrich failures the classifier cannot already act on. An
                    // interrupt (user abort / pipeline timeout) or an exception that
                    // already classifies as infra must propagate unchanged --
                    // wrapping them in a fresh cause-less Exception would erase the
                    // FlowInterruptedException / typed-infra signal from the chain.
                    def preClassified = FailureClassifier.classify(submitEx, InfraFailure.SLURM)
                    if (preClassified instanceof PipelineInterruption || preClassified instanceof InfraFailure) {
                        throw submitEx
                    }
                    // The sh step's exception carries only the exit code, so fold the
                    // sbatch output captured by the submit script (stderr included)
                    // into a fresh exception for FailureClassifier.classify() at the
                    // runLLMTestlistOnSlurm caller; the shared-lib PATTERN_CATALOG
                    // (e.g. its "Slurm backup controller in standby mode" row) then
                    // drives the retry/severity decision. An empty read (file
                    // missing or unreadable) preserves today's behavior by
                    // rethrowing the original exception unchanged.
                    def sbatchOutput = readSlurmSubmitOutput(pipeline, remote, jobWorkspace, stageName)
                    if (sbatchOutput) {
                        throw new Exception("SLURM job submission failed for ${stageName}: ${sbatchOutput}")
                    }
                    throw submitEx
                }

                def slurmMetadata = captureSlurmWorkspaceMetadata(pipeline, remote, jobWorkspace, placementContext, stageName)
                def slurmJobId = slurmMetadata.slurmJobId
                if (!slurmJobId) {
                    // `|| true` so a missing file (submission never wrote it) yields
                    // an empty string and reaches the typed throw below, instead of a
                    // generic "script returned exit code 1" the classifier can't act on.
                    slurmJobId = Utils.exec(
                        pipeline,
                        script: Utils.sshUserCmd(remote, "\"cat ${jobWorkspace}/slurm_job_id.txt 2>/dev/null || true\""),
                        returnStdout: true,
                        numRetries: 3
                    ).trim()
                    recordSlurmPlacementContext(placementContext, slurmJobId, null, stageName)
                }
                if (!isValidSlurmJobId(slurmJobId)) {
                    // The job never entered the SLURM queue (sbatch failed or the ID
                    // was never captured), so nothing ran on any node: retryable
                    // infra by construction, and node-avoidance has nothing to avoid.
                    throw new InfraFailure(
                        "SLURM job submission for ${stageName} produced no usable job ID " +
                        "(slurm_job_id.txt missing, empty, or non-numeric in ${jobWorkspace}); " +
                        "the job never entered the queue.",
                        null, InfraFailure.TRANSIENT, InfraFailure.SLURM, "<typed:slurm-submit-no-jobid>")
                }
                Utils.exec(pipeline, script: "echo Slurm job ID: ${slurmJobId}")
                // Record the live SLURM job so a dispatcher-pod death can be reconciled
                // off-pod (the in-pod cleanup can't reach the login node once the pod is
                // gone). Deregistered when cleanup actually runs.
                registerSlurmResource(stageName, [clusterName: partition.clusterName, jobUID: jobUID, slurmJobId: slurmJobId, usedSbatch: true])

                def scriptTrack = """#!/bin/bash
                    set -xEeuo pipefail
                    trap 'rc=\$?; echo "Error in file \${BASH_SOURCE[0]} on line \$LINENO: \$BASH_COMMAND (exit \$rc)"; exit \$rc' ERR

                    jobId=${slurmJobId}
                    tail -f ${slurmJobLogPath} &
                    tailPid=\$!

                    # Wait until Slurm job is done
                    while true; do
                        # Use --allocations to ensure we match the exact job ID and not job steps (like 123.batch, 123.0)
                        # Tolerate transient sacct failures (e.g. slurmdbd unreachable) so the tracker survives controller blips.
                        if ! STATUS=\$(sacct -j \$jobId --format=State -Pn --allocations 2>&1); then
                            echo "Warning: sacct failed for job \$jobId: \$STATUS"
                            sleep 60
                            continue
                        fi

                        if [[ -z \$STATUS || \$STATUS == "RUNNING" || \$STATUS == "PENDING" || \$STATUS == "CONFIGURING" ]]; then
                            echo "Slurm job \$jobId state: \${STATUS:-UNKNOWN}"
                            sleep 300
                        else
                            echo "Slurm job \$jobId finished with state: \$STATUS"
                            break
                        fi
                    done

                    # Stop and reap the log follower. It may have already exited
                    # when the remote log stream closes; that is not a test failure.
                    kill \$tailPid 2>/dev/null || true
                    wait \$tailPid 2>/dev/null || true

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
                    NODE_LIST=""
                    if [ -s "${jobWorkspace}/slurm_node_list.txt" ]; then
                        NODE_LIST=\$(awk 'NF { print; exit }' "${jobWorkspace}/slurm_node_list.txt" || true)
                    fi
                    if [ -z "\$NODE_LIST" ]; then
                        NODE_LIST=\$(sacct -j \$jobId --format=NodeList -Pn --allocations 2>/dev/null | head -1 || true)
                    fi
                    echo "Slurm job \$jobId nodelist: \${NODE_LIST:-UNKNOWN}"
                    printf '%s\n' "\$NODE_LIST" > "${jobWorkspace}/slurm_node_list.txt"

                    # Record the verdict and always exit 0: a re-run can't change a
                    # terminal state, so numRetries should only fire on transport loss.
                    printf '%s|%s\n' "\$STATUS" "\$EXIT_CODE" > "${jobWorkspace}/slurm_job_result.txt"
                    if [[ "\$STATUS" == "COMPLETED" && \$EXIT_CODE -eq 0 ]]; then
                        echo "Pytest succeed in Slurm job \$jobId"
                    else
                        echo "Pytest failed in Slurm job \$jobId"
                        echo "Full test output (logs not shown above) is uploaded after stage teardown to:"
                        echo "  https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/test-results/results-${stageName}${postTag}.tar.gz"
                    fi
                    echo "Status: \$STATUS | Exit_code \$EXIT_CODE"
                    exit 0
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

                // Monitor the job. The track script always exits 0 once it records a
                // verdict, so a re-run can't change a terminal state. A frontend lost
                // mid-monitor is recovered by the enclosing withSlurmFrontendFailover
                // -- it fails the closure over to another frontend and the submit guard
                // reuses the still-active job -- so the monitor needs no same-frontend
                // retries of its own.
                // Track the Slurm job alongside a controller-side watcher
                // that SSH-stats the remote results.xml and SCPs / uploads
                // a progress tar whenever the file's mtime advances. Both
                // run inside a single `sh` step (track in foreground, watcher
                // as a background subshell) so Blue Ocean renders the stage
                // as a single box instead of a nested parallel split.
                def pytestDoneFile = "${WORKSPACE}/.pytest-done-${stageName}"
                def progressTar = "results-${stageName}${postTag}-progress.tar.gz"
                def progressUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/test-results/${progressTar}"
                def remoteWorkspaceTrk = "/home/svc_tensorrt/bloom/scripts/${jobUID}"
                def trackCmd = Utils.sshUserCmd(remote, scriptTrackPathNode)
                // Stat on the compute node via srun --overlap to bypass the login
                // node's NFS attribute cache.  The login node's NFS client can cache
                // a stale directory view for up to acdirmax seconds after the compute
                // node creates results.xml; running stat directly on the node that
                // wrote the file avoids this cross-client visibility window entirely.
                // After the mtime check confirms a change, sshRefreshCacheCmd primes
                // the login node's NFS cache before the SCP step reads the file.
                // Some clusters' srun cli_filter plugin rejects any job step that
                // doesn't carry -A <account> (e.g. "You must specify an account"),
                // so this overlap step needs the same account the sbatch submission
                // used, pulled out of partition.additionalArgs.
                def slurmAccountMatch = (partition.additionalArgs =~ /(?:^|\s)-A\s+(\S+)/)
                def srunAccountArg = slurmAccountMatch.find() ? "-A ${slurmAccountMatch.group(1)} " : ""
                def sshStatCmd = Utils.sshUserCmd(remote, "\"srun --overlap ${srunAccountArg}--quiet --jobid='${slurmJobId}' --ntasks=1 stat -c %Y '${remoteWorkspaceTrk}/results.xml' || echo 0\"")
                def scpXmlCmd = scpFromRemoteCmd(remote, "${remoteWorkspaceTrk}/results*.xml", "${stageName}/")
                def scpUnfinishedCmd = scpFromRemoteCmd(remote, "${remoteWorkspaceTrk}/unfinished_test.txt", "${stageName}/")
                def sshRefreshCacheCmd = Utils.sshUserCmd(remote, "\"ls '${remoteWorkspaceTrk}/' > /dev/null 2>&1; ls -la '${remoteWorkspaceTrk}/results.xml' > /dev/null 2>&1 || true\"")
                def sshListPerfCmd = Utils.sshUserCmd(remote, "\"find '${remoteWorkspaceTrk}' -maxdepth 1 -type d \\( -name 'aggr*' -o -name 'disagg*' \\) -print 2>/dev/null || true\"")
                def scpPerfTemplate = scpFromRemoteCmd(remote, "PERF_FOLDER_PLACEHOLDER", "${stageName}/")
                sh "rm -f ${pytestDoneFile}"
                withCredentials([usernamePassword(
                        credentialsId: 'urm-artifactory-creds',
                        usernameVariable: 'ART_USER',
                        passwordVariable: 'ART_PASS')]) {
                    sh """#!/bin/bash
                        set +e
                        export STAGE_NAME='${stageName}'
                        export PROGRESS_TAR='${progressTar}'
                        export PROGRESS_URL='${progressUrl}'
                        export TIMEOUT_XML_SCRIPT='${llmSrcLocal}/jenkins/scripts/generate_timeout_xml.py'
                        export POST_TAG='${postTag}'
                        # ---- background watcher: SSH-stat remote XML, SCP, tar, upload ----
                        PROGRESS_DONE_FILE='${pytestDoneFile}' \\
                        PROGRESS_INTERVAL=${PROGRESS_UPLOAD_INTERVAL_SEC} \\
                        LABEL_PREFIX='sbatch checkpoint' \\
                        SLURM_SSH_STAT_CMD='${sshStatCmd}' \\
                        SLURM_SSH_REFRESH_CACHE_CMD='${sshRefreshCacheCmd}' \\
                        SLURM_SCP_XML_CMD='${scpXmlCmd}' \\
                        SLURM_SCP_UNFINISHED_CMD='${scpUnfinishedCmd}' \\
                        SLURM_SSH_LIST_PERF_CMD='${sshListPerfCmd}' \\
                        SLURM_SCP_PERF_TEMPLATE='${scpPerfTemplate}' \\
                        bash '${llmSrcLocal}/jenkins/scripts/progress_upload_watcher.sh' &
                        WATCHER_PID=\$!

                        # ---- foreground track: retry up to 3 times on failure ----
                        attempt=0
                        rc=1
                        while [ \$attempt -lt 3 ]; do
                            ${trackCmd}
                            rc=\$?
                            [ \$rc -eq 0 ] && break
                            attempt=\$((attempt+1))
                            [ \$attempt -lt 3 ] && echo "Track failed (rc=\$rc), retry \$attempt/3" && sleep 30
                        done

                        touch '${pytestDoneFile}'
                        wait \$WATCHER_PID 2>/dev/null || true

                        # ---- immediate final snapshot ----
                        mkdir -p '${WORKSPACE}/${stageName}'
                        ${sshRefreshCacheCmd}
                        for _attempt in 1 2 3; do
                            ${scpXmlCmd} && break
                            [ \$_attempt -lt 3 ] && echo "[PROGRESS-UPLOAD] ${stageName}: scp xml failed, retry \$_attempt/3" && sleep 10
                        done || true
                        _unfinished_ok=0
                        for _attempt in 1 2 3; do
                            ${scpUnfinishedCmd} && { _unfinished_ok=1; break; }
                            echo "[PROGRESS-UPLOAD] ${stageName}: scp unfinished failed (attempt \$_attempt/3)"
                            [ \$_attempt -lt 3 ] && sleep 10
                        done
                        [ "\$_unfinished_ok" -eq 0 ] && echo "[PROGRESS-UPLOAD] ${stageName}: scp unfinished not available, skipping"
                        SCP_PERF_TMPL='${scpPerfTemplate}'
                        while IFS= read -r perf_folder; do
                            [ -z "\$perf_folder" ] && continue
                            _perf_ok=0
                            for _attempt in 1 2 3; do
                                eval "\${SCP_PERF_TMPL//PERF_FOLDER_PLACEHOLDER/\$perf_folder}" && { _perf_ok=1; break; }
                                echo "[PROGRESS-UPLOAD] ${stageName}: scp perf \$perf_folder failed (attempt \$_attempt/3)"
                                [ \$_attempt -lt 3 ] && sleep 10
                            done
                            [ "\$_perf_ok" -eq 0 ] && echo "[PROGRESS-UPLOAD] ${stageName}: scp perf \$perf_folder failed after 3 attempts"
                        done < <(eval '${sshListPerfCmd}' 2>/dev/null || true)
                        if [ -f '${WORKSPACE}/${stageName}/results.xml' ]; then
                            LABEL='sbatch final snapshot' FINAL_SNAPSHOT=1 \\
                            bash '${llmSrcLocal}/jenkins/scripts/progress_upload_snapshot.sh' || true
                        fi

                        exit \$rc
                    """
                }

                // Verdict: "<STATE>|<EXIT_CODE>"; success is COMPLETED + exit 0.
                // STATE may contain spaces.
                def jobResult = readSlurmWorkspaceFile(pipeline, remote, "${jobWorkspace}/slurm_job_result.txt", stageName, 3)
                def resultFields = jobResult ? jobResult.tokenize('|') : []
                def jobState = resultFields ? resultFields[0]?.tokenize(' ')?.getAt(0) : null
                def jobExit = resultFields.size() > 1 ? resultFields[1] : null
                if (jobState != "COMPLETED" || jobExit != "0") {
                    // Verdict unreadable: fall back to an authoritative sacct query.
                    def slurmState = jobState ?: querySlurmJobState(pipeline, cluster, partition.clusterName, slurmJobId)
                    // ... and re-confirm success, so a transient read blip on a job
                    // that actually passed doesn't fail the stage.
                    if (jobState == null && slurmState == "COMPLETED") {
                        echo "[INFRA-RETRY] ${stageName}: verdict unreadable but sacct reports COMPLETED for ${slurmJobId}; treating as success."
                        return
                    }
                    // TIMEOUT is a walltime kill -- typed UserFailure so neither
                    // retry layer re-runs a job that would just time out again.
                    if (slurmState == "TIMEOUT") {
                        throw new UserFailure(
                            "SLURM job ${slurmJobId} for ${stageName} ended in state TIMEOUT " +
                            "(hit partition walltime ${partition?.time}min); not retrying.",
                            null)
                    }
                    // Verdict unreadable but the job is still alive: a transport blip
                    // dropped the monitor while the job kept running, so this is infra,
                    // not a test failure.
                    if (isNonTerminalSlurmState(slurmState)) {
                        throw new InfraFailure(
                            "SLURM job ${slurmJobId} for ${stageName} is still in non-terminal state ${slurmState}; " +
                            "the monitor lost contact while the job was alive (transient infra), so this is not a " +
                            "test failure.",
                            null, InfraFailure.TRANSIENT, InfraFailure.SLURM, "<typed:slurm-job-still-running>")
                    }
                    // A terminal FAILED state may be a node/device fault whose signature
                    // (CUDA/NVLink/ECC/driver) printed only into the SLURM job output log,
                    // never into this verdict. Scrape the log and, on a hit, surface the
                    // matched line into a fresh exception so the authoritative catalog
                    // (FailureClassifier.classify at the runLLMTestlistWithSbatch caller)
                    // can match it and steer the retry off the bad node. A miss falls
                    // through to the plain "Pytest failed" rethrow below.
                    if (slurmState == "FAILED") {
                        def deviceHit = scrapeSlurmLogForDeviceFault(pipeline, remote, slurmJobLogPath)
                        if (deviceHit) {
                            echo "[INFRA-RETRY] ${stageName}: device-fault signature in SLURM job ${slurmJobId} log; " +
                                 "surfacing to classifier: ${deviceHit}"
                            throw new Exception(
                                "Device/interconnect fault on SLURM node during job ${slurmJobId} for ${stageName}: ${deviceHit}")
                        }
                    }
                    echo "[INFRA-RETRY] ${stageName}: SLURM job ${slurmJobId} state=${slurmState ?: 'unknown'}, exit=${jobExit ?: 'unknown'}; deferring to classifier."
                    throw new Exception("Pytest failed in SLURM job ${slurmJobId} for ${stageName}")
                }
            }
            echo "Finished test stage execution."
            }  // end CloudManager.withSlurmFrontendFailover
        }  // end withCredentials
    } catch (InterruptedException e) {
        stageIsInterrupted = true
        throw e
    } catch (Exception e) {
        caughtStageError = e
        throw e
    } finally {
        // Resource cleanup must run even if metadata capture or result upload is interrupted.
        try {
            captureSlurmJobNodeList(pipeline, cluster, partition.clusterName, placementContext?.slurmJobId ?: null, placementContext, stageName, jobWorkspace)
            // Suppress this attempt's junit when a retry is still planned (a retryable
            // infra failure with budget), so a retried-and-passed stage doesn't leave
            // the build UNSTABLE from an intermediate attempt's results. A genuine test
            // failure classifies as UserFailure -> not suppressed -> reported.
            boolean suppressTestReporting = (caughtStageError != null && retryContext != null) &&
                retryContextAllowsRetry(null, retryContext, caughtStageError, false)
            uploadResults(pipeline, cluster, partition.clusterName, jobUID, stageName, postTag, suppressTestReporting)
            deleteProgressArtifact(stageName, postTag)
        } finally {
            stage("Clean Up Slurm Resource") {
                // Workaround to handle the interruption during clean up SLURM resources
                retry(3) {
                    try {
                        cleanUpSlurmResources(pipeline, cluster, partition.clusterName, jobUID)
                    } catch (Exception e) {
                        error "Error during clean up SLURM resources: ${e.getMessage()} and retrying."
                    }
                }
            }
            // Cleanup ran on the live pod; drop the registry entry so the off-pod
            // finalizer/sweep does not reconcile already-freed resources.
            deregisterSlurmResource(stageName)
        }
    }
}

// CBTS Layer 2.5: rename narrowed stages (reuse-safety) and resize their splits to k.
def cbtsResizeSplits(configs) {
    def cbts = testFilter[(CBTS_RESULT)]
    if (cbts == null || !cbts.cbts_test_db_artifact_path) {
        return configs
    }
    def kByStage = cbts.affected_stage_split_counts
    if (!kByStage) {
        return configs
    }
    def resized = [:]
    configs.each { key, values ->
        def k = kByStage[key]
        if (k == null) {
            resized[key] = values
            return
        }
        int kk = Math.max(1, k as int)
        if ((values[2] as int) > kk) {
            echo "CBTS [${cbts.scope}]: ${key} narrowed -> ${kk} shard(s); dropping group ${values[2]}"
            return
        }
        def v = values.collect()
        v[3] = kk
        resized[key + CBTS_STAGE_SUFFIX] = v
    }
    return resized
}

// CBTS Layer 2: replace the normal stage set with the selector's affected
// stages while retaining the baseline sanity and multi-GPU gates.
def filterCbtsStageJobs(parallelJobs, parallelJobsFiltered, multiGpuJobs, testFilter) {
    def cbts = testFilter[(CBTS_RESULT)]
    if (cbts == null) {
        return parallelJobsFiltered
    }

    // cbtsResizeSplits renames only narrowed stages (those in
    // affected_stage_split_counts) to `-cbts`; affected-but-not-narrowed
    // stages keep their original name, so match each per its actual key.
    def stageSuffix = cbts.cbts_test_db_artifact_path ? CBTS_STAGE_SUFFIX : ""
    def narrowed = (cbts.affected_stage_split_counts ?: [:]).keySet()
    def affectedSet = (cbts.affected_stages ?: []).collect {
        (stageSuffix && narrowed.contains(it)) ? (it + stageSuffix) : it
    } as Set
    def needsSanity = cbts.sanity_required
    def needsPerfSanity = cbts.perfsanity_required
    def filtered = parallelJobs.findAll { key, _ ->
        if (key.contains("-OnDemand-")) {
            return false
        }
        if (key =~ /Post-Merge/) return affectedSet.contains(key)
        return affectedSet.contains(key) ||
               (needsSanity && key =~ /PackageSanityCheck/) ||
               (needsPerfSanity && key =~ /PerfSanity/)
    }
    if (affectedSet.isEmpty()) {
        if (filtered.isEmpty()) {
            echo "CBTS [${cbts.scope}]: trigger-mode mismatch + nothing force-kept → no-op"
        } else {
            echo "CBTS [${cbts.scope}]: trigger-mode mismatch — running " +
                 "${filtered.size()} force-kept stage(s) only"
        }
    } else if (filtered) {
        echo "CBTS [${cbts.scope}]: limiting to ${filtered.size()} stages " +
             "(sanity_required=${needsSanity}, perfsanity_required=${needsPerfSanity})"
    } else {
        echo "CBTS [${cbts.scope}]: empty stage set after filtering"
    }

    // The coverage tier omits multi-GPU; re-add it under the baseline gate.
    if (cbts.enable_multi_gpu && testFilter[(MULTI_GPU_FILE_CHANGED)]) {
        filtered += multiGpuJobs
        echo "CBTS [${cbts.scope}]: multi-GPU file changed → running " +
             "${multiGpuJobs.size()} multi-GPU stage(s) at baseline"
    }
    return filtered
}

// True when an exception indicates the K8s dispatcher pod this SLURM stage runs
// inside died mid-run -- kubelet eviction, container termination, or the JNLP
// agent otherwise going offline. Retrying inside such a pod is futile (every
// step runs on the dead agent and fails immediately) and its in-pod cleanup can
// no longer reach the SLURM controller, so callers stop retrying in place and
// reconcile the orphaned SLURM job / Jenkins node off-pod. Delegates to the shared
// ContextDeath.isContextDeath, which matches the same pattern set the inlined
// version used (see the contract block above) across the flattened cause chain --
// so the signal is still recognized when wrapped by the cleanup's AbortException
// (e.g. "Error during clean up SLURM resources: ... marked offline: Pod failed
// (Reason: Evicted ...)") -- and additionally traverses suppressed causes.
boolean isDispatcherPodFailure(Throwable e) {
    return ContextDeath.isContextDeath(e)
}

def runLLMTestlistOnSlurm(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, gpuCount=1, nodeCount=1, runWithSbatch=false, skipInstallWheel=false, cpver="cp312", String outerAttemptTag="", boolean useClusterDurations=false, Integer infraRetryMax=null)
{
  echo "Run Slurm job with native sbatch: $runWithSbatch"

  // Per-stage override of the SLURM infra-retry budget (from opts.infraRetryMax,
  // threaded via the dispatcher's retryContext). Lets resource-scarce pools cap
  // or disable stage-level retries (0 = no retry). Null falls back to the global.
  int slurmInfraRetryMax = resolveInfraRetryMax(InfraFailure.SLURM, infraRetryMax)

  def attempt = 0
  // Avoided SLURM nodes keyed by cluster. The platform can resolve to a
  // different cluster on each attempt (auto: platforms pick one at random), and
  // a node name from one cluster is unknown to another's controller -- passing
  // it to --exclude makes sbatch hard-fail with "Invalid node name specified".
  // Keying by cluster lets each attempt exclude only nodes that belong to the
  // cluster it actually submits to.
  def avoidedSlurmNodeListsByCluster = [:]

  while (true) {
    attempt++
    Map attemptPlacementContext = [
      excludedSlurmNodeListsByCluster: avoidedSlurmNodeListsByCluster.collectEntries { c, ns -> [(c): ns.collect()] }
    ]
    try {
      if (attempt > 1) {
        echo "[INFRA-RETRY] ${stageName}: Starting attempt ${attempt} of ${slurmInfraRetryMax + 1}"
        if (!avoidedSlurmNodeListsByCluster.isEmpty()) {
          echo "[INFRA-RETRY] ${stageName}: avoiding prior SLURM node list(s): " +
               avoidedSlurmNodeListsByCluster.collect { c, ns -> "${c}: ${ns.join(' ')}" }.join('; ')
        }
      }

      // Each attempt uploads its own test-result artifact under a unique name so
      // the attempt-1 tar (already in Artifactory from its finally block) is not
      // clobbered and the ensureStageResultNotUploaded guard does not trip on
      // the retry. First attempt keeps the canonical unsuffixed name so existing
      // downstream consumers (dashboards, the JIRA bot, etc.) are unaffected.
      //
      // outerAttemptTag is the K8s outer dispatcher pod's tag ("" for outer
      // attempt 1, "-pod-${N}" for outer attempt N>=2). Prefixing the inner
      // postTag with it ensures the new dispatcher pod's inner attempt 1
      // upload doesn't collide with the dead pod's already-recorded upload
      // in GlobalState.uploadResultStageNames.
      def innerSuffix = (attempt == 1) ? "" : "-attempt-${attempt}"
      def postTag = "${outerAttemptTag}${innerSuffix}"

      // Describes this attempt so the stage body can suppress its junit when a
      // retryable infra failure means another attempt will follow (mirrors the K8s
      // path's retryContext). scope=SLURM so classification/budget match this loop.
      def slurmRetryContext = [
        scope: InfraFailure.SLURM,
        stageName: stageName,
        attempt: attempt,
        backoffMs: 60L * 1000L,
      ]

      if (nodeCount > 1 || runWithSbatch) {
        runLLMTestlistWithSbatch(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, gpuCount, nodeCount, skipInstallWheel, cpver, postTag, useClusterDurations, attemptPlacementContext, slurmRetryContext)
      } else {
        runLLMTestlistWithAgent(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, gpuCount, skipInstallWheel, cpver, postTag, useClusterDurations, attemptPlacementContext, slurmRetryContext)
      }

      // Job succeeded
      if (attempt > 1) {
        echo "[INFRA-RETRY] ${stageName}: Succeeded on attempt ${attempt}"
      }
      return

    } catch (InterruptedException e) {
      // User abort / pipeline timeout -- never retry
      throw e
    } catch (Exception e) {
      // If the K8s dispatcher pod this stage runs inside died mid-run, every
      // retry attempt would execute on that same dead pod and fail immediately,
      // and the in-pod cleanup can no longer reach the SLURM controller. Stop
      // retrying in place and propagate so the pod-level wrapper reconciles the
      // orphaned SLURM job / Jenkins node off-pod (fail closed).
      if (isDispatcherPodFailure(e)) {
        echo "[INFRA-RETRY] ${stageName}: dispatcher pod died mid-run; not retrying on the dead pod (${e.toString()}). Failing closed for off-pod reconciliation."
        throw e
      }
      // classify() handles FlowInterruptedException + exit-code-143 +
      // typed throws + cause-chain pattern matching, returning one of
      // PipelineInterruption / InfraFailure / UserFailure. Scope=SLURM
      // ensures we only match catalog rows tagged SLURM or BOTH.
      def c = FailureClassifier.classify(e, InfraFailure.SLURM)
      if (c instanceof PipelineInterruption) throw e
      if (!(c instanceof InfraFailure)) {
        // UserFailure -> don't retry, but leave a trace: a failure whose
        // exception matches no catalog pattern lands here and would
        // otherwise decline the retry with no log output at all.
        echo "[INFRA-RETRY] ${stageName}: SLURM attempt failed with no infra pattern matched (classified as user failure); not retrying. Exception: ${e.toString()}"
        throw e
      }

      rememberAvoidedSlurmNodeLists(avoidedSlurmNodeListsByCluster, attemptPlacementContext.lastSlurmClusterName, attemptPlacementContext.lastSlurmNodeList, stageName)

      def effectiveMax = (c.severity == InfraFailure.PERSISTENT) ? Math.min(1, slurmInfraRetryMax) : slurmInfraRetryMax

      if (attempt > effectiveMax) {
        echo "[INFRA-RETRY] ${stageName}: Infrastructure failure (${c.detectedPattern}) " +
             "but max retries (${effectiveMax}) exhausted after ${attempt} attempts. Failing."
        throw e
      }
      if (!hasBudgetForInfraRetry(pipeline, stageName, InfraFailure.SLURM, c, attempt, effectiveMax, 60L * 1000L, true)) {
        echo "[INFRA-RETRY] ${stageName}: Infrastructure failure (${c.detectedPattern}) is retryable, " +
             "but remaining CI timeout budget is too small for another SLURM attempt. Failing without retry."
        throw e
      }

      echo "[INFRA-RETRY] ${stageName}: Infrastructure failure detected on attempt ${attempt}: " +
           "${c.detectedPattern}"
      echo "[INFRA-RETRY] ${stageName}: Exception: ${e.toString()}"
      echo "[INFRA-RETRY] ${stageName}: Will retry (attempt ${attempt + 1} of ${effectiveMax + 1}) after 60s cooldown."

      sleep(60)
    }
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

// Check if a stage key matches a pattern.
// Supports exact match and wildcard '*' for glob-style matching.
// Uses Pattern.quote() to safely handle special characters in stage names.
// Examples: "A10-PyTorch-1" (exact), "*PerfSanity*" (contains), "A10-*" (prefix).
def stageMatchesPattern(String key, String pattern) {
    if (!pattern.contains('*')) {
        return key == pattern
    }
    def regex = '^' + pattern.split('\\*', -1).collect { java.util.regex.Pattern.quote(it) }.join('.*') + '$'
    return key ==~ regex
}

// Check if a stage key matches any pattern in the list.
def stageMatchesAnyPattern(String key, List patterns) {
    return patterns.any { pattern -> stageMatchesPattern(key, pattern) }
}

// Test filter flags
// Multi-GPU stages matching any entry here run inside the single-GPU job
// instead of waiting for the separate multi-GPU dispatch (which requires
// the 'ci: full pre-merge approved' label). Supports exact names and
// wildcard (*) patterns.
@Field
def MULTI_GPU_RUN_WITH_SINGLE = [
    // Add stage patterns here, e.g.:
    // "DGX_H100-2_GPUs-*",
]

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
def CBTS_RESULT = "cbts_result"
// Pipeline-level CBTS coverage eligibility, decided in L0_MergeRequest.groovy.
@Field
def CBTS_COVERAGE = "cbts_coverage"
@Field
def INFRA_DRY_RUN = "infra_dry_run"
// Suffix for CBTS-narrowed stages so their results aren't reused by non-CBTS runs.
// A suffix (not prefix) keeps the GPU type as the first '-' token for positional parsers.
@Field
def CBTS_STAGE_SUFFIX = "-cbts"
// Sentinel in the stage output dir; while it exists no process writes a .cbtscov file.
@Field
def CBTS_STOP_FILE_NAME = "cbts_stop"
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
    (CBTS_RESULT): null,
    (CBTS_COVERAGE): false,
    (INFRA_DRY_RUN): false,
]

@Field
def GITHUB_PR_API_URL = "github_pr_api_url"
@Field
def CACHED_CHANGED_FILE_LIST = "cached_changed_file_list"
@Field
def ACTION_INFO = "action_info"
@Field
def IMAGE_KEY_TO_TAG = "image_key_to_tag"
@Field
def TRTLLM_VERSION_OVERRIDE = "trtllm_version_override"
@Field
def RUN_MODE = "run_mode"
def globalVars = [
    (GITHUB_PR_API_URL): null,
    (CACHED_CHANGED_FILE_LIST): null,
    (ACTION_INFO): null,
    (IMAGE_KEY_TO_TAG): [:],
    (TRTLLM_VERSION_OVERRIDE): null,
    (RUN_MODE): null,
]

class GlobalState {
    static def uploadResultStageNames = []
    static def stageAttemptEstimateMs = [:]
    static def stageAttemptEstimateDetails = [:]

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

def recordRenderedStageAttemptEstimate(pipeline, String llmSrc, String testListPath, String stageName, def renderedTestCount, String clusterName=null)
{
    def estimate = trtllm_utils.estimateRenderedStageAttemptMillis(pipeline, llmSrc, testListPath, stageName, renderedTestCount, clusterName)
    if (estimate.error) {
        echo "[CI-BUDGET] ${stageName}: failed to read .test_durations; using count-based estimate. Error: ${estimate.error}"
    }

    GlobalState.stageAttemptEstimateMs[stageName] = estimate.estimatedMs
    GlobalState.stageAttemptEstimateDetails[stageName] = [
        renderedCount: estimate.renderedCount,
        knownCount: estimate.knownCount,
        unknownCount: estimate.unknownCount,
    ]
    echo "[CI-BUDGET] ${stageName}: recorded test runtime estimate " +
         "${trtllm_utils.formatCiBudgetMillis(estimate.estimatedMs)} " +
         "(rendered=${estimate.renderedCount}, known=${estimate.knownCount}, unknown=${estimate.unknownCount})"
}

long estimateStageRetryRuntimeMs(String stageName, String scope)
{
    Long testEstimateValue = trtllm_utils.parseCiBudgetLong(GlobalState.stageAttemptEstimateMs[stageName])
    long testEstimateMs = testEstimateValue != null ? testEstimateValue : 0L
    if (testEstimateMs > 0L) {
        long overheadMs = scope == InfraFailure.SLURM ? 45L * 60L * 1000L : 30L * 60L * 1000L
        return testEstimateMs + overheadMs
    }
    return scope == InfraFailure.SLURM ? 4L * 60L * 60L * 1000L : 2L * 60L * 60L * 1000L
}

long retrySafetyMarginMs(String scope)
{
    return scope == InfraFailure.SLURM ? 20L * 60L * 1000L : 15L * 60L * 1000L
}

// Resolve a per-stage infra-retry override (opts.infraRetryMax) against its
// scope's global budget. The override may only CAP or DISABLE retries, never
// exceed the global (which bounds CI time / worst-case attempts), so it is
// clamped to [0, scopeDefault]; e.g. with the default of 1, infraRetryMax=2
// still yields 1, and infraRetryMax=0 disables. Null falls back to the default.
int resolveInfraRetryMax(String scope, Integer override)
{
    int scopeDefault = (scope == InfraFailure.SLURM) ? SLURM_INFRA_RETRY_MAX : K8S_INFRA_RETRY_MAX
    if (override == null) {
        return scopeDefault
    }
    return Math.max(0, Math.min(override, scopeDefault))
}

int retryMaxForFailure(String scope, InfraFailure failure, Integer infraRetryMax=null)
{
    // Per-stage override caps the scope default (see resolveInfraRetryMax); a
    // PERSISTENT failure is still capped to at most one retry, so infraRetryMax=0
    // disables retries for every severity.
    int base = resolveInfraRetryMax(scope, infraRetryMax)
    return (failure.severity == InfraFailure.PERSISTENT) ? Math.min(1, base) : base
}

boolean hasBudgetForInfraRetry(def pipeline, String stageName, String scope, InfraFailure failure, int attempt, int effectiveMax, long backoffMs, boolean logDecision)
{
    if (attempt > effectiveMax) {
        return false
    }
    long estimateMs = estimateStageRetryRuntimeMs(stageName, scope)
    long safetyMs = retrySafetyMarginMs(scope)
    return trtllm_utils.canSpendCiBudget(pipeline, [
        globalVars: globalVars,
        label: "infra-retry:${scope}:${stageName}:attempt-${attempt + 1}",
        estimateMs: estimateMs,
        backoffMs: backoffMs,
        safetyMs: safetyMs,
        logDecision: logDecision,
    ])
}

boolean retryContextAllowsRetry(def pipeline, Map retryContext, Throwable error, boolean logDecision=false)
{
    if (retryContext == null || error == null) {
        return false
    }
    String scope = retryContext.scope ?: InfraFailure.K8S
    def classified = FailureClassifier.classify(error, scope)
    if (!(classified instanceof InfraFailure)) {
        return false
    }
    Long parsedAttempt = trtllm_utils.parseCiBudgetLong(retryContext.attempt)
    int attempt = parsedAttempt != null ? parsedAttempt as int : 1
    int effectiveMax = retryMaxForFailure(scope, classified, retryContext.infraRetryMax as Integer)
    Long parsedBackoffMs = trtllm_utils.parseCiBudgetLong(retryContext.backoffMs)
    long backoffMs = parsedBackoffMs != null ? parsedBackoffMs : 60L * 1000L
    return hasBudgetForInfraRetry(pipeline, retryContext.stageName ?: "Unknown", scope, classified, attempt, effectiveMax, backoffMs, logDecision)
}

def rememberAvoidedKubernetesHostNodes(List avoidedNodes, def nodes, String stageName)
{
    def hostNodes = trtllm_utils.normalizeKubernetesHostNames(nodes)
    hostNodes.each { nodeName ->
        if (!avoidedNodes.contains(nodeName)) {
            avoidedNodes << nodeName
        }
    }
    if (!hostNodes.isEmpty()) {
        echo "[INFRA-RETRY] ${stageName}: recorded Kubernetes host node(s) for retry avoidance: ${hostNodes.join(', ')}"
    }
}

def rememberAvoidedSlurmNodeLists(Map avoidedNodeListsByCluster, String clusterName, def nodes, String stageName)
{
    // Without the cluster the node ran on we can't safely exclude it (a node name
    // is only valid on its own cluster's controller), so skip -- degrading to no
    // avoidance rather than risking a cross-cluster "Invalid node name" submit.
    if (!clusterName) {
        return
    }
    def nodeLists = trtllm_utils.normalizeSlurmNodeLists(nodes)
    if (nodeLists.isEmpty()) {
        return
    }
    def clusterNodeLists = avoidedNodeListsByCluster.get(clusterName)
    if (clusterNodeLists == null) {
        clusterNodeLists = []
        avoidedNodeListsByCluster.put(clusterName, clusterNodeLists)
    }
    nodeLists.each { nodeList ->
        if (!clusterNodeLists.contains(nodeList)) {
            clusterNodeLists << nodeList
        }
    }
    echo "[INFRA-RETRY] ${stageName}: recorded SLURM node list(s) for retry avoidance on ${clusterName}: ${nodeLists.join(', ')}"
}

def readSlurmWorkspaceFile(def pipeline, Map remote, String path, String stageName, int numRetries=1)
{
    try {
        def value = Utils.exec(
            pipeline,
            script: Utils.sshUserCmd(
                remote,
                Utils.bashWrappedRemoteCmd("cat ${path} 2>/dev/null || true")
            ),
            returnStdout: true,
            numRetries: numRetries
        ).trim()
        return value.readLines().collect { it.trim() }.find { it } ?: ""
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        // A dead frontend must propagate so the enclosing withSlurmFrontendFailover
        // fails over to another remote; swallowing it as "" would strand the stage on
        // the unreachable frontend. Any other read failure (missing file, transient)
        // is non-fatal -- the metadata is best-effort, so return "" and carry on.
        if (CloudManager.isSlurmFrontendConnectionFailure(e)) {
            throw e
        }
        echo "[INFRA-RETRY] ${stageName}: unable to read SLURM metadata file ${path}: ${e.toString()}"
        return ""
    }
}

// State flows through the mutated placementContext map (read by the retry
// loop); this records into it and returns nothing.
def recordSlurmPlacementContext(Map placementContext, String slurmJobID, def nodeList, String stageName)
{
    if (placementContext == null) {
        return
    }

    if (isValidSlurmJobId(slurmJobID)) {
        placementContext.slurmJobId = slurmJobID
    }

    def normalized = trtllm_utils.normalizeSlurmNodeLists(nodeList)
    if (!normalized.isEmpty()) {
        placementContext.lastSlurmNodeList = normalized.join(' ')
        def jobLabel = slurmJobID ? "job ${slurmJobID}" : "job"
        echo "[INFRA-RETRY] ${stageName}: SLURM ${jobLabel} ran on node list(s): ${placementContext.lastSlurmNodeList}"
    }
}

def captureSlurmWorkspaceMetadata(def pipeline, Map remote, String jobWorkspace, Map placementContext, String stageName)
{
    def metadata = [slurmJobId: null, nodeList: null]
    if (!jobWorkspace) {
        return metadata
    }

    metadata.slurmJobId = readSlurmWorkspaceFile(pipeline, remote, "${jobWorkspace}/slurm_job_id.txt", stageName)
    if (metadata.slurmJobId && !isValidSlurmJobId(metadata.slurmJobId)) {
        // A garbage value (e.g. an error message that landed in the file) must
        // not be recorded or later fed into sacct/scontrol diagnostics.
        echo "[INFRA-RETRY] ${stageName}: ignoring non-numeric SLURM job ID '${metadata.slurmJobId}' from ${jobWorkspace}/slurm_job_id.txt"
        metadata.slurmJobId = null
    }
    metadata.nodeList = readSlurmWorkspaceFile(pipeline, remote, "${jobWorkspace}/slurm_node_list.txt", stageName)
    if (placementContext != null && metadata.slurmJobId) {
        placementContext.slurmJobId = metadata.slurmJobId
    }
    def normalized = trtllm_utils.normalizeSlurmNodeLists(metadata.nodeList)
    if (placementContext != null && !normalized.isEmpty()) {
        placementContext.lastSlurmNodeList = normalized.join(' ')
        metadata.nodeList = placementContext.lastSlurmNodeList
    }
    return metadata
}

def captureSlurmJobNodeList(def pipeline, SlurmCluster cluster, String clusterName, String slurmJobID, Map placementContext, String stageName, String jobWorkspace=null)
{
    if (placementContext == null) {
        return
    }

    def capturedJobID = slurmJobID
    def nodeList = null
    try {
        CloudManager.withSlurmFrontendFailover(pipeline, clusterName, cluster) { remote ->
            def metadata = captureSlurmWorkspaceMetadata(pipeline, remote, jobWorkspace, placementContext, stageName)
            capturedJobID = capturedJobID ?: metadata.slurmJobId
            nodeList = metadata.nodeList
            if (!capturedJobID && jobWorkspace) {
                capturedJobID = readSlurmWorkspaceFile(pipeline, remote, "${jobWorkspace}/slurm_job_id.txt", stageName, 3)
                recordSlurmPlacementContext(placementContext, capturedJobID, null, stageName)
            }
            if (!nodeList && jobWorkspace) {
                nodeList = readSlurmWorkspaceFile(pipeline, remote, "${jobWorkspace}/slurm_node_list.txt", stageName, 3)
            }
            // Digit-only revalidation: capturedJobID may come from the caller or
            // a raw workspace-file read, and an invalid value here would produce
            // the same "Invalid job id specified" noise in sacct/scontrol.
            if (!isValidSlurmJobId(capturedJobID)) {
                return
            }

            if (nodeList) {
                return
            }

            nodeList = Utils.exec(
                pipeline,
                script: Utils.sshUserCmd(
                    remote,
                    "\"bash -c 'sacct -j ${capturedJobID} --format=NodeList -Pn --allocations 2>/dev/null | head -1 || true'\""
                ),
                returnStdout: true,
                numRetries: 1
            ).trim()
            if (!nodeList) {
                nodeList = Utils.exec(
                    pipeline,
                    script: Utils.sshUserCmd(
                        remote,
                        "\"bash -c 'scontrol show job ${capturedJobID} 2>/dev/null | tr \" \" \"\\n\" | sed -n \"s/^NodeList=//p\" | head -1 || true'\""
                    ),
                    returnStdout: true,
                    numRetries: 1
                ).trim()
            }
        }
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        echo "[INFRA-RETRY] ${stageName}: unable to capture SLURM node list for job ${capturedJobID}: ${e.toString()}"
    }

    recordSlurmPlacementContext(placementContext, capturedJobID, nodeList, stageName)
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

def cacheErrorAndUploadResult(stageName, taskRunner, finallyRunner, noResultIfSuccess=false, postTag="", boolean isFinalAttempt=true, Map retryContext=null)
{
    checkStageName([stageName])
    def Boolean stageIsInterrupted = false
    def Boolean stageIsFailed = true
    Throwable caughtError = null
    try {
        taskRunner()
        stageIsFailed = false
    } catch (InterruptedException e) {
        stageIsInterrupted = true
        throw e
    } catch (Exception e) {
        caughtError = e
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

            // Suppress synthetic stage-fail XML and junit() when this attempt is
            // an intermediate retry that classified as a retryable infra failure.
            // Without this, every transient pod-level failure poisons the per-build
            // junit report with a permanent "Stage Failed" entry that the retry
            // (on a fresh pod) cannot remove. The tar artifact is still uploaded
            // for forensics; only the in-build test reporting is gated.
            boolean suppressTestReporting = false
            if (stageIsFailed && caughtError != null) {
                if (retryContext != null) {
                    suppressTestReporting = retryContextAllowsRetry(null, retryContext, caughtError, false)
                } else if (!isFinalAttempt) {
                    def c = FailureClassifier.classify(caughtError, InfraFailure.K8S)
                    suppressTestReporting = c instanceof InfraFailure
                }
                if (suppressTestReporting) {
                    suppressTestReporting = true
                    echo "[INFRA-RETRY] ${stageName}${postTag}: suppressing synthetic stage-fail XML and junit() because a retry is still planned"
                }
            }

            echo "noResultIfSuccess: ${noResultIfSuccess}, stageIsFailed: ${stageIsFailed}"
            sh "mkdir -p ${stageName}"
            finallyRunner()
            if (stageIsFailed && !suppressTestReporting) {
                if (stageIsInterrupted) {
                    echo "Stage is interrupted, skip to generate terminated unexpectedly test result."
                } else if (!fileExists("${stageName}/results-timeout.xml")) {
                    // Generate timeout test result xml if there are terminated unexpectedly tests
                    generateTimeoutTestResultXml(pipeline, stageName)
                }
                // Generate stage fail test result xml if the stage failed and there is no result*.xml
                def stageXml = generateStageFailTestResultXml(stageName, "Stage Failed", "Stage run failed without result", "results*.xml")
                if (stageXml != null) {
                    sh "echo '${stageXml}' > ${stageName}/results-stage.xml"
                }
            }
            sh "STAGE_NAME=${stageName} && env | sort > ${stageName}/debug_env.txt"
            if (isCbtsStage(stageName)) {
                freezeCbtsCoverage(stageName)
            }
            echo "Upload test results."
            // promoteProgressTar is a server-side move of the already-uploaded
            // progress snapshot. It is only valid when on-disk results*.xml are
            // unchanged from that snapshot. After a rename (or any other local
            // XML mutation) the snapshot is stale and must be re-tarred.
            boolean xmlsMutated = false
            if (suppressTestReporting) {
                // This attempt is superseded by a planned retry. Keep the tar for
                // forensics, but move its result XMLs aside so the top-level Collect
                // Test Result stage's junit('**/results*.xml') does not re-ingest a
                // superseded attempt's results (e.g. a results-timeout.xml left by a
                // monitor-cut still-running job) and flip the build UNSTABLE even
                // though the stage passed on retry. junit() here was already gated.
                sh """
                    cd ${stageName} && for f in results*.xml; do
                        [ -e "\$f" ] && mv "\$f" "superseded-\$f"
                    done || true
                """
                xmlsMutated = true
            }

            if (xmlsMutated || !promoteProgressTar(stageName, postTag)) {
                if (xmlsMutated) {
                    echo "[PROGRESS-UPLOAD] ${stageName}: results*.xml changed on disk, re-uploading instead of promoting progress tar"
                } else {
                    echo "[PROGRESS-UPLOAD] ${stageName}: no successful progress upload recorded, falling back to direct upload"
                }
                def transformOpt = postTag ? "--transform 's|^\\(${stageName}/results[^/]*\\)\\.xml\$|\\1${postTag}.xml|'" : ""
                sh "tar -czvf results-${stageName}${postTag}.tar.gz ${transformOpt} ${stageName}/"
                trtllm_utils.uploadArtifacts(
                    "results-${stageName}${postTag}.tar.gz",
                    "${UPLOAD_PATH}/test-results/"
                )
            }
            deleteProgressArtifact(stageName, postTag)
            if (!suppressTestReporting) {
                junit(testResults: "${stageName}/results*.xml")
            }
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

def createKubernetesPodConfig(image, type, arch = "amd64", gpuCount = 1, perfMode = false, modelExpress = false)
{
    def targetCloud = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/arch: ${arch}
                  kubernetes.io/os: linux"""
    def containerConfig = ""
    def nodeLabelPrefix = ""
    def tolerations = ""
    def extraDeviceEnv = ""
    def serviceContainerConfig = ""

    def archSuffix = arch == "arm64" ? "arm" : "amd"
    def jnlpImage = "artifactory.pdx.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_${archSuffix}_linux:jdk17"

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
    case "cpu":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS_TEST}]
                    tty: true
                    resources:
                      requests:
                        cpu: ${TESTER_CPU_ONLY_CORES}
                        memory: ${TESTER_CPU_ONLY_MEMORY}
                        ephemeral-storage: 300Gi
                      limits:
                        cpu: ${TESTER_CPU_ONLY_CORES}
                        memory: ${TESTER_CPU_ONLY_MEMORY}
                        ephemeral-storage: 300Gi
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
            // Keep ModelExpress services inside the existing pod resource envelope;
            // otherwise their requests can make an otherwise valid GPU pod unschedulable.
            def serviceCpuReserve = modelExpress ? 6 : 0
            def serviceMemoryReserveGi = modelExpress ? 16 : 0
            def serviceStorageReserveGi = modelExpress ? 8 : 0
            memorySize = "${gpuCount * 150 - serviceMemoryReserveGi}" + "Gi"
            storageSize = "${gpuCount * 150 - serviceStorageReserveGi}" + "Gi"
            cpuCount = "${gpuCount * 12 - serviceCpuReserve}"
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
    if (modelExpress) {
        if (arch != "amd64") {
            throw new Exception("ModelExpress CI sidecars currently support amd64 test pods only.")
        }
        extraDeviceEnv += """
                    - name: MODEL_EXPRESS_URL
                      value: "http://127.0.0.1:8001"
                    - name: TRTLLM_MX_E2E_REQUIRED
                      value: "1"
        """
        // Mirrors the ModelExpress v0.4.1 Redis deployment and image contract.
        // The image exposes /app/modelexpress-server and accepts the port/backend settings below.
        // Use regular containers because the Jenkins Kubernetes launcher does not
        // reliably attach to pods containing restartable init-container sidecars.
        // The server waits for Redis, and the E2E preflight waits for port 8001.
        serviceContainerConfig = """
                  - name: redis
                    image: ${MODEL_EXPRESS_REDIS_IMAGE}
                    args: ["--save", "", "--appendonly", "no"]
                    ports:
                    - containerPort: 6379
                    resources:
                      requests:
                        cpu: '1'
                        memory: 4Gi
                        ephemeral-storage: 2Gi
                      limits:
                        cpu: '1'
                        memory: 4Gi
                        ephemeral-storage: 2Gi
                    imagePullPolicy: Always
                  - name: model-express-server
                    image: ${MODEL_EXPRESS_SERVER_IMAGE}
                    command: ["/bin/bash", "-c"]
                    args:
                    - |
                      until (echo > /dev/tcp/127.0.0.1/6379) >/dev/null 2>&1; do
                        sleep 1
                      done
                      exec /app/modelexpress-server --port 8001
                    env:
                    - name: MX_METADATA_BACKEND
                      value: "redis"
                    - name: REDIS_URL
                      value: "redis://127.0.0.1:6379"
                    ports:
                    - containerPort: 8001
                    resources:
                      requests:
                        cpu: '4'
                        memory: 8Gi
                        ephemeral-storage: 4Gi
                      limits:
                        cpu: '4'
                        memory: 8Gi
                        ephemeral-storage: 4Gi
                    imagePullPolicy: Always
        """
    }
    // Temporarily avoid an arm64 CPU builder with repeated pod DNS/JNLP failures seen in Build-SBSA #5564.
    def blockedNodeAffinity = targetCloud == "kubernetes-cpu" && arch == "arm64" ? '''
                              - key: "kubernetes.io/hostname"
                                operator: NotIn
                                values:
                                - "rl300-0021.ipp2a1.colossus"''' : ""
    def nodeLabel = trtllm_utils.generateNodeLabel(nodeLabelPrefix)
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
    if (type.contains("6000d") || type.contains("rtx-5080")) {
        // rtx-pro-6000d, gh200 and rtx-5080 nodes are located in Austin DC, we use the FlexCache to speed up the data access.
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
                                - "qa_only"
${blockedNodeAffinity}
                nodeSelector: ${selectors}
                imagePullSecrets:
                  - name: ${ARTIFACTORY_IMAGE_PULL_SECRET}
                containers:
                  ${containerConfig}
                    env:
                    - name: HOST_NODE_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: spec.nodeName
                    ${extraDeviceEnv}
                  ${serviceContainerConfig}
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
                export TRTLLM_DOCS_REQUIRE_IMPORT=1 && \
                export LD_LIBRARY_PATH=\$(python3 ../scripts/cuda_driver_stub.py) && \
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

def runLLMAgentFlowTest(pipeline, stageName)
{
    // agent-flow is a self-contained, pure-CPU sub-project: it needs neither a
    // GPU nor the TRT-LLM wheel, only its own dependencies (pure-Python wheels
    // declared in agent-flow/pyproject.toml).
    sh "pwd && ls -alh"
    trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, false, true)
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "git config --global --add safe.directory \"*\"")

    def agentFlowRoot = "${LLM_ROOT}/agent-flow"

    // Install agent-flow with its test extras (pytest, pytest-asyncio) and the
    // runtime deps from pyproject.toml (claude-agent-sdk, openai-codex, ...).
    // These resolve from the container's default PyPI mirror.
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${agentFlowRoot} && pip3 install -e \".[test]\"")

    if (isInfraDryRun()) {
        // Keep the normal environment and reporting path, but replace product tests.
        sh """
            rm -rf "${agentFlowRoot}/tests" && \
            mkdir -p "${agentFlowRoot}/tests" && \
            printf '%s\\n' \
                'def test_infra_dry_run_placeholder():' \
                '    pass' \
                > "${agentFlowRoot}/tests/test_infra_dry_run_placeholder.py"
        """
    }

    sh "mkdir -p ${WORKSPACE}/${stageName}"

    // test_workflow_entrypoint_modules_run_without_import_warnings is deselected
    // because importing agent_flow.workflows.modeling_bringup runs a live skill
    // probe (a real Claude/Codex session) at module-import time, which stalls in
    // headless CI. This is a known agent-flow bug slated for an upstream fix;
    // remove the --deselect once that lands. The outer `timeout` guards against
    // the same probe stalling the whole stage in-process.
    def deselect = "--deselect tests/test_examples.py::test_workflow_entrypoint_modules_run_without_import_warnings"
    sh """
        cd ${agentFlowRoot} && \
        timeout 1200 python3 -m pytest tests -v ${deselect} \
            --junitxml=${WORKSPACE}/${stageName}/results.xml
    """

    // Rename the JUnit testsuite from pytest's default "pytest" to the stage
    // name so CI reporting attributes these cases to this stage (mirrors the
    // unittest path in runLLMTestlistOnPlatformImpl).
    sh "cd ${WORKSPACE}/${stageName} && sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' results.xml || true"
}

def launchTestListCheck(pipeline)
{
    stageName = "Test List Check"
    trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(LLM_DOCKER_IMAGE, "a10"), "trt-llm", {
        try {
            echoNodeAndGpuInfo(pipeline, stageName)
            sh "nvidia-smi && nvidia-smi -q && nvidia-smi topo -m"
            // download TRT-LLM tarfile
            def tarName = BUILD_CONFIGS[VANILLA_CONFIG][TARNAME]
            def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pwd && wget -nv ${llmTarfile} && ls -alh")
            sh "tar -zxf ${tarName}"
            def llmPath = sh (script: "realpath .", returnStdout: true).trim()
            def llmSrc = "${llmPath}/TensorRT-LLM/src"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install -r ${llmSrc}/requirements-dev.txt")
            // --validate --parity: after --l0/--qa generate the collectable lists, assert every
            // statically-verified parametrize ID is actually collectable (validate<->collection parity).
            sh "NVIDIA_TRITON_SERVER_VERSION=26.05 LLM_ROOT=${llmSrc} LLM_BACKEND_ROOT=${llmSrc}/triton_backend python3 ${llmSrc}/scripts/check_test_list.py --l0 --qa --waive --validate --parity"
        } catch (InterruptedException e) {
            throw e
        } catch (Exception e) {
            throw e
        }
    })
}

def generateTimeoutTestResultXml(pipeline, stageName) {
    def scriptPath = sh(
        script: "find . -name generate_timeout_xml.py | head -n 1 | xargs realpath",
        returnStdout: true
    ).trim()
    def curPath = sh(script: "realpath .", returnStdout: true).trim()
    def outputFilePath = "${curPath}/${stageName}/results-timeout.xml"
    sh """python3 ${scriptPath} --stage-name '${stageName}' --test-file-path 'unfinished_test.txt' --output-file '${outputFilePath}'"""
    if (fileExists(outputFilePath)) {
        return true
    }
    return false
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

def parseTaskConfigFromStageName(String stageName) {
    def taskConfig = null
    def matcher = (stageName =~ /([^-]+)(?:-(\d+)_GPUs)?(?:-(\d+)_Nodes)?/)
    if (matcher.find()) {
        taskConfig = [
            gpu: "${matcher.group(1)}",
            system_gpu_count: matcher.group(2) ?: "1", // Default to 1 if _GPUs not present
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
    } else if (stageName.contains("-Generic-")) {
        // Generic stages select tests by marker expression rather than backend ownership.
        makoArgs += ["backend=generic"]
    } else if (stageName.contains("-Verl-")) {
        // If stageName contains "-Verl-", add "backend=verl" to makoArgs
        // At this point, only tests with backend=verl or unspecified backend will be run
        makoArgs += ["backend=verl"]
    } else {
        // If stageName does not contain "-PyTorch-", "-CPP-", "-Triton-", "-FMHA-", or "-Verl-", do not add any backend
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
        def taskConfig = parseTaskConfigFromStageName(stageName)
        if (taskConfig) {
            makoArgs += [
                "gpu=${taskConfig.gpu}",
                "system_gpu_count=${taskConfig.system_gpu_count}"
            ]
        }
    }

    return makoArgs
}

def renderTestDB(pipeline, testContext, llmSrc, stageName, preDefinedMakoOpts=null, String clusterName=null) {
    def makoOpts = preDefinedMakoOpts

    if (!makoOpts) {
        def makoArgs = getMakoArgsFromStageName(stageName)
        if (stageName.startsWith("CPU-")) {
            def cpuName = env.targetArch == AARCH64_TRIPLE ? "aarch64" : "x86_64"
            makoOpts = transformMakoArgsToJson(
                ["Mako options:"] + makoArgs + [
                    "system_gpu_count=0",
                    "cpu=${cpuName}",
                    "linux_distribution_name=ubuntu"
                ])
        } else {
            def scriptPath = "${llmSrc}/tests/integration/defs/sysinfo/get_sysinfo.py"
            makoOpts = getMakoOpts(scriptPath, makoArgs)
        }
    }

    // Log the resolved mako match on every render, tagged with stage+context.
    // transformMakoArgsToJson already echoes the bare JSON ("Test DB Mako opts:"),
    // but unlabeled and far upstream of the render; the preDefinedMakoOpts path
    // skips it entirely. This co-locates the match with the stage/context and the
    // "-> N tests" summary below under one greppable renderTestDB: prefix, so a
    // wrong-but-non-empty render (a stale/unexpected sysinfo value selecting the
    // wrong block) is diagnosable per stage, not just the "na"/empty cases.
    echo "renderTestDB: stage=${stageName} context=${testContext} mako match: ${makoOpts}"

    if (makoOpts.contains('"na"')) {
        // "na" is a sysinfo probe failure sentinel (see get_sysinfo.py). Blocks
        // conditioned on the failed property silently drop out of the render, so
        // even a non-empty list may be missing tests. Warn here, where every
        // sysinfo-based stage passes through, not just when the list ends up empty.
        echo "WARNING: renderTestDB: some sysinfo probes returned \"na\": ${makoOpts}. " +
             "Test-db blocks conditioned on those properties (e.g. linux_distribution_name: ubuntu*) " +
             "will NOT be selected."
    }
    sh "pip3 install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-tensorrt-pypi/simple --ignore-installed trt-test-db==1.8.5+bc6df7"
    // CBTS Layer 3: download the pre-built cbts_test_db/ tarball that the
    // orchestrator uploaded to Artifactory (see getCbtsResult in
    // L0_MergeRequest.groovy). This avoids re-running main.py locally and
    // avoids passing large PR-diff payloads as Jenkins parameters (env vars).
    // If the download or extraction fails we swallow the error: the override
    // directory will be absent below, the overrideYaml check will fail, and
    // renderTestDB falls back to the source test-db.
    def cbts = testFilter[(CBTS_RESULT)]
    if (cbts != null && cbts.test_db_dir_override && cbts.cbts_test_db_artifact_path) {
        try {
            // Always re-fetch: a reused workspace may hold a stale cbts_test_db/ shadowing this build's YAMLs.
            def artifactUrl = "${URM_ARTIFACTORY_BASE}/${cbts.cbts_test_db_artifact_path}"
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv '${artifactUrl}' -O /tmp/cbts_test_db.tar.gz && tar xzf /tmp/cbts_test_db.tar.gz -C ${llmSrc}")
            echo "CBTS Layer 3: extracted cbts_test_db from artifact"
        } catch (Exception e) {
            echo "CBTS Layer 3: artifact download failed " +
                 "(${e.class.simpleName}: ${e.message}); falling back to source test-db"
        }
    }
    def testDBPath = "${llmSrc}/tests/integration/test_lists/test-db"
    if (cbts != null && cbts.test_db_dir_override) {
        def overrideYaml = "${llmSrc}/${cbts.test_db_dir_override}/${testContext}.yml"
        def overrideOk = sh(returnStdout: true, script: "test -s ${overrideYaml} && echo yes || echo no").trim()
        if (overrideOk == "yes") {
            testDBPath = "${llmSrc}/${cbts.test_db_dir_override}"
            echo "CBTS [${cbts.scope}]: rendering test list from filtered test-db at ${testDBPath}"
        } else {
            echo "CBTS [${cbts.scope}]: ${overrideYaml} missing/empty -- falling back to source test-db"
        }
    }
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
    // Count non-empty lines, not newlines: trt-test-db writes the test names
    // with no trailing newline, so `wc -l` undercounts by one -- it reports 0
    // for a single-test render, which would trip the empty-list guard below
    // (and mis-report every stage's count by one). `grep -c .` is agnostic to
    // the missing terminator. It exits 1 for no matches (a legitimately empty
    // render -> count "0"); accept only that, so a read failure (exit 2:
    // missing file, unreadable, etc.) still aborts the step instead of being
    // silently masked.
    def testCount = sh(returnStdout: true, script: "grep -c . -- ${testList} || test \$? -eq 1").trim()
    def testDBLabel = (cbts != null && cbts.test_db_dir_override) ? "CBTS-narrowed [${cbts.scope}]" : "source"
    echo "renderTestDB: stage=${stageName} context=${testContext} test-db=${testDBLabel} dir=${testDBPath} -> ${testCount} tests"
    sh(script: "cat ${testList}")
    if (testCount == "0") {
        // An empty render is never legitimate here: every launched stage must
        // have tests (CBTS drops stages with an empty selection before launch).
        // Fail now with the match query rather than letting pytest --collect-only
        // exit 5 later with an unattributable "Test collection failed" message.
        def hint = makoOpts.contains('"na"') ?
            " Some sysinfo probes returned \"na\" (see the match JSON above); a broken probe" +
            " (e.g. the python 'distro' module missing) makes conditions like" +
            " linux_distribution_name: ubuntu* match nothing." : ""
        error("renderTestDB: rendered EMPTY test list for stage=${stageName} " +
              "context=${testContext} test-db=${testDBLabel}. Match query: ${makoOpts}.${hint}")
    }
    recordRenderedStageAttemptEstimate(pipeline, llmSrc, testList, stageName, testCount, clusterName)

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
def rerunFailedTests(stageName, llmSrc, testCmdLine, resultFileName="results.xml", testType="regular", postTag="") {
    if (!fileExists("${WORKSPACE}/${stageName}/${resultFileName}")) {
        echo "There is no ${resultFileName} file, skip the rerun step"
        return true
    }

    // Create rerun directory structure to avoid conflicts
    def rerunDir = "${WORKSPACE}/${stageName}/rerun/${testType}"
    sh "mkdir -p ${rerunDir}"

    // Generate rerun test lists
    def failSignaturesList = trtllm_utils.getFailSignaturesList().join(",")
    def testListCmd = testCmdLine.find { it.contains("--test-list=") }
    def testListArg = testListCmd ? "--test-list=${testListCmd.split('=', 2)[1]}" : ""
    def unfinishedTestFile = "${WORKSPACE}/${stageName}/unfinished_test.txt"
    def unfinishedTestArg = fileExists(unfinishedTestFile) ? "--unfinished-test-file=${unfinishedTestFile}" : ""
    sh """
        python3 ${llmSrc}/jenkins/scripts/test_rerun.py \
        generate_rerun_tests_list \
        --output-dir=${rerunDir}/ \
        --input-file=${WORKSPACE}/${stageName}/${resultFileName} \
        --fail-signatures='${failSignaturesList}' \
        ${testListArg} \
        ${unfinishedTestArg}
    """

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
        def rerunProgressTar = "results-${stageName}${postTag}-progress.tar.gz"
        def rerunProgressUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/test-results/${rerunProgressTar}"
        def rerunDoneFile = "${WORKSPACE}/.rerun${times}-done-${stageName}"
        sh "rm -f ${rerunDoneFile}"
        try {
            withCredentials([usernamePassword(
                    credentialsId: 'urm-artifactory-creds',
                    usernameVariable: 'ART_USER',
                    passwordVariable: 'ART_PASS')]) {
                sh """
                    set +e
                    export STAGE_NAME='${stageName}'
                    export PROGRESS_TAR='${rerunProgressTar}'
                    export PROGRESS_URL='${rerunProgressUrl}'
                    export POST_TAG='${postTag}'
                    # ---- background watcher for rerun${times} ----
                    PROGRESS_DONE_FILE='${rerunDoneFile}' \\
                    PROGRESS_INTERVAL=${PROGRESS_UPLOAD_INTERVAL_SEC} \\
                    LABEL_PREFIX='rerun${times} checkpoint' \\
                    XML_PATH='${xmlFile}' \\
                    bash '${llmSrc}/jenkins/scripts/progress_upload_watcher.sh' &
                    WATCHER_PID=\$!

                    # ---- foreground rerun ----
                    cd ${llmSrc}/tests/integration/defs && \\
                    ${newTestCmdLine.join(" ")}
                    rc=\$?

                    touch '${rerunDoneFile}'
                    wait \$WATCHER_PID 2>/dev/null || true

                    # ---- immediate final snapshot of rerun${times} ----
                    LABEL='rerun${times} final snapshot' FINAL_SNAPSHOT=1 \\
                    bash '${llmSrc}/jenkins/scripts/progress_upload_snapshot.sh' || true

                    exit \$rc
                """
            }
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

/**
 * Append passes that previously succeeded for this commit + stage to the
 * stage's waives.txt as SKIPs, so the upcoming pytest run skips them.
 *
 * Two sources are merged:
 *
 *  1. OpenSearch records of prior pipeline runs for the same commit and
 *     stage (the historical mechanism). Populated only after a pipeline
 *     completes -- has nothing to say about the current run.
 *
 *  2. Tarballs uploaded by earlier attempts of the *current* build, when
 *     the infra-failure retry loop has fired. The current run's attempt 1
 *     uploads its (partial) results before the retry kicks off; this
 *     function downloads them and extracts passes via test_rerun.py's
 *     extract_passed_tests mode. Closes the gap where a retry would
 *     otherwise re-run every test that already passed pre-failure.
 *
 * @param llmSrc      Local TRT-LLM source root.
 * @param stageName   Stage name; both the OpenSearch query key and the
 *                    Artifactory artifact prefix (`results-${stageName}*`).
 * @param waivesTxt   Path to the stage's waives.txt; passes are appended
 *                    here with reason "SKIP (Reused from previous pipeline)".
 * @param postTag     This attempt's full tar suffix (e.g. ""
 *                    on attempt 1, "-attempt-2" on the first retry,
 *                    "-SubJob-RunTest-attempt-2" for a retried sub-job).
 *                    Used by priorAttemptTags() to enumerate earlier
 *                    attempts in this build.
 */
def reusePassedTestResults(llmSrc, stageName, waivesTxt, String postTag = "") {
    try {
        def reusedTests = []
        def workDir = "${WORKSPACE}/${stageName}"
        sh "mkdir -p ${workDir}"

        // 1. OpenSearch lookup -- tests that PASSED in a previous pipeline run
        //    for this commit + stage.
        def passedTestListFile = "${workDir}/passed_test_list.txt"
        sh """
            python3 ${llmSrc}/jenkins/scripts/open_search_query.py \
            --commit-id ${env.gitlabCommit} \
            --stage-name ${stageName} \
            --output-file ${passedTestListFile}
        """
        if (fileExists(passedTestListFile)) {
            reusedTests += readFile(file: passedTestListFile).readLines().collect { it.trim() }.findAll { it }
        }

        // 2. Prior-attempt recovery -- tests that PASSED in an earlier attempt
        //    of THIS pipeline run before infra retry fired. Only runs if postTag
        //    decodes as a retry attempt (matches "...-attempt-N").
        def priorTags = priorAttemptTags(postTag)
        if (!priorTags.isEmpty()) {
            def priorXmls = []
            priorTags.each { priorTag ->
                def tarName = "results-${stageName}${priorTag}.tar.gz"
                def tarUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/test-results/${tarName}"
                def priorDir = "${workDir}/prior${priorTag.replace('-', '_')}"
                sh "mkdir -p ${priorDir}"
                // Probe with HEAD so we can distinguish "this prior attempt never
                // uploaded a tarball" (HTTP 404, expected when an attempt died
                // before its finally block ran) from real errors (auth, 5xx,
                // network). Only 404 is benign; anything else fails the build
                // so silent skips don't mask a configuration regression.
                def httpStatus = sh(returnStdout: true,
                                    script: "curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 10 --max-time 30 '${tarUrl}'").trim()
                if (httpStatus == '404') {
                    echo "Prior attempt artifact ${tarName} not present (HTTP 404); skipping"
                    return
                }
                if (httpStatus != '200') {
                    error "Probing prior attempt artifact ${tarName} returned HTTP ${httpStatus} (expected 200 or 404)"
                }
                sh "cd ${priorDir} && wget -nv -nc '${tarUrl}'"
                sh "cd ${priorDir} && tar -xzf ${tarName}"
                // results.xml may live at ${stageName}/results.xml inside the
                // tar, or at the tar's root depending on how it was packaged.
                // Scan both. Also match superseded-results*.xml: a suppressed
                // intermediate attempt renames its result XMLs with that prefix
                // (so the build-level junit does not re-ingest the superseded
                // attempt), but its PASSED tests are still valid to reuse here --
                // extract_passed_tests only pulls the passing subset.
                def xmlFiles = sh(returnStdout: true,
                                  script: "find ${priorDir} -maxdepth 4 \\( -name 'results*.xml' -o -name 'superseded-results*.xml' \\) 2>/dev/null | tr '\\n' ',' | sed 's/,\$//'").trim()
                if (xmlFiles) {
                    priorXmls += xmlFiles.split(',') as List
                }
            }
            if (!priorXmls.isEmpty()) {
                def priorPassedFile = "${workDir}/prior_attempt_passed.txt"
                sh """
                    python3 ${llmSrc}/jenkins/scripts/test_rerun.py \
                    extract_passed_tests \
                    --output-file ${priorPassedFile} \
                    --input-files ${priorXmls.join(',')}
                """
                if (fileExists(priorPassedFile)) {
                    def priorPasses = readFile(file: priorPassedFile).readLines().collect { it.trim() }.findAll { it }
                    if (!priorPasses.isEmpty()) {
                        echo "Reusing ${priorPasses.size()} passed test(s) from prior attempt(s): ${priorTags}"
                    }
                    reusedTests += priorPasses
                }
            }
        }

        // 3. Dedupe and append everything to waives.txt as SKIPs.
        reusedTests = reusedTests.unique()
        if (reusedTests.size() > 0) {
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

// Promotes the progress tar to the final results path via an Artifactory
// server-side move (no data re-transfer). Returns true when a progress
// snapshot existed and was promoted; false when no snapshot was uploaded or
// the server-side move failed.
// Virtual repo sw-tensorrt-generic does not support move; rewrite to the
// backing local repo as we do for DELETE in deleteProgressArtifact().
def promoteProgressTar(stageName, postTag="") {
    def progressOkFile = "${WORKSPACE}/results-${stageName}${postTag}-progress.tar.gz.upload_ok"
    if (!fileExists(progressOkFile)) {
        return false
    }
    def localUploadPath = UPLOAD_PATH.replaceFirst(/^sw-tensorrt-generic\//, 'sw-tensorrt-generic-local/')
    def srcArtPath = "${localUploadPath}/test-results/results-${stageName}${postTag}-progress.tar.gz"
    def dstArtPath = "${localUploadPath}/test-results/results-${stageName}${postTag}.tar.gz"
    def rc
    withCredentials([usernamePassword(
            credentialsId: 'urm-artifactory-creds',
            usernameVariable: 'ART_USER',
            passwordVariable: 'ART_PASS')]) {
        rc = sh(
            script: """curl -fsSL --retry 2 -u "\$ART_USER:\$ART_PASS" -X POST \
                'https://urm.nvidia.com/artifactory/api/move/${srcArtPath}?to=/${dstArtPath}'""",
            returnStatus: true
        )
        if (rc == 0) {
            echo "[PROGRESS-UPLOAD] ${stageName}: progress tar moved to test-results/ as results-${stageName}${postTag}.tar.gz"
        } else {
            echo "[PROGRESS-UPLOAD] ${stageName}: move failed (rc=${rc}); results may already be at destination or progress tar was deleted"
        }
    }
    return rc == 0
}

// Removes the in-progress checkpoint tarball uploaded by the inline
// shell watcher in runLLMTestlistOnPlatformImpl / runLLMTestlistWithSbatch.
// Called after final-result handling in both execution paths, including when
// test execution failed. Progress checkpoints are not intentionally retained
// based on the test outcome; interruption or upload failure may skip cleanup.
//
// Build-scoped: ${UPLOAD_PATH} is per-build, so unswept progress tars are
// garbage-collected with the rest of the build's artifacts anyway.
def deleteProgressArtifact(stageName, postTag="") {
    // UPLOAD_PATH points to the virtual repo `sw-tensorrt-generic`, which
    // accepts GET/PUT but returns 404 on DELETE. Rewrite to the backing
    // local repo so the delete actually lands.
    def deletePath = UPLOAD_PATH.replaceFirst(/^sw-tensorrt-generic\//, 'sw-tensorrt-generic-local/')
    def targetUrl = "https://urm.nvidia.com/artifactory/${deletePath}/test-results/results-${stageName}${postTag}-progress.tar.gz"
    try {
        withCredentials([usernamePassword(
                credentialsId: 'urm-artifactory-creds',
                usernameVariable: 'ART_USER',
                passwordVariable: 'ART_PASS')]) {
            def httpStatus = sh(
                script: "curl -sSo /dev/null -w '%{http_code}' --retry 2 -X DELETE -u \"\$ART_USER:\$ART_PASS\" '${targetUrl}'",
                returnStdout: true
            ).trim()
            if (httpStatus == '204' || httpStatus == '200') {
                echo "[PROGRESS-UPLOAD] ${stageName}: deleted progress tar ${targetUrl}"
            } else if (httpStatus == '404') {
                echo "[PROGRESS-UPLOAD] ${stageName}: progress tar not found, skipping delete"
            } else {
                echo "[PROGRESS-UPLOAD] ${stageName}: delete returned HTTP ${httpStatus} (non-fatal)"
            }
        }
    } catch (InterruptedException e) {
        throw e
    } catch (Exception e) {
        echo "[PROGRESS-UPLOAD] ${stageName}: progress tar delete failed (non-fatal): ${e.message}"
    }
}

/**
 * Decode a postTag into the postTags used by prior attempts of the same
 * stage in this build.
 *
 * The retry runners compose postTag from two retry layers:
 *   - runKubernetesPodWithInfraRetry (outer K8s pod retry) contributes
 *     ``"-pod-${P}"`` for outer attempt P>=2 (and ``""`` for P=1).
 *   - runLLMTestlistOnSlurm (inner SLURM retry) appends ``"-attempt-${I}"``
 *     for inner attempt I>=2 (and ``""`` for I=1).
 * The two suffixes nest: full postTag = ``${base}${outerTag}${innerSuffix}``
 * where base is the caller-supplied prefix (e.g. ``""`` or
 * ``"-SubJob-RunTest"``). The ``-pod-`` and ``-attempt-`` separators are
 * distinct so the nested form is unambiguous.
 *
 * This function inverts that composition to enumerate the postTags of
 * earlier attempts so reusePassedTestResults() can locate their uploaded
 * tarballs.
 *
 * Returns an empty list when ``postTag`` does not encode a retry — either
 * attempt 1 of attempt 1 (``postTag == ""``), or a caller-supplied tag that
 * never went through the retry loop (e.g. ``"-SubJob-RunTest"``).
 *
 * For nested cases we don't know how many inner attempts each prior outer
 * pod attempt completed, so we over-enumerate up to SLURM_INFRA_RETRY_MAX+1;
 * the HTTP probe in reusePassedTestResults handles 404 for absent tarballs.
 *
 * Examples (with default SLURM_INFRA_RETRY_MAX=2):
 *
 *   ""                            -> []
 *   "-attempt-2"                  -> [""]
 *   "-attempt-3"                  -> ["", "-attempt-2"]
 *   "-pod-2"                      -> ["", "-attempt-2", "-attempt-3"]
 *   "-pod-2-attempt-2"            -> ["", "-attempt-2", "-attempt-3", "-pod-2"]
 *   "-SubJob-RunTest"             -> []
 *   "-SubJob-RunTest-pod-2"       -> ["-SubJob-RunTest", "-SubJob-RunTest-attempt-2", "-SubJob-RunTest-attempt-3"]
 *
 * @param postTag the current attempt's full tar suffix.
 * @return List of postTag strings, ordered oldest-attempt-first.
 */
@NonCPS
def priorAttemptTags(String postTag) {
    if (!postTag) return []
    // Peel "-attempt-N" suffix (inner SLURM retry counter) if present.
    String remaining = postTag
    Integer innerAttempt = null
    def mInner = remaining =~ /^(.*)-attempt-(\d+)$/
    if (mInner.matches()) {
        innerAttempt = (mInner[0][2] as Integer)
        remaining = mInner[0][1]
    }
    // Peel "-pod-P" suffix (outer K8s retry counter) if present.
    Integer podAttempt = null
    def mPod = remaining =~ /^(.*)-pod-(\d+)$/
    if (mPod.matches()) {
        podAttempt = (mPod[0][2] as Integer)
        remaining = mPod[0][1]
    }
    String base = remaining
    if (innerAttempt == null && podAttempt == null) return []   // no retry encoded
    int outerN = podAttempt ?: 1
    int innerN = innerAttempt ?: 1
    int maxInner = SLURM_INFRA_RETRY_MAX + 1
    def priors = []
    // Earlier outer attempts (1..outerN-1) with all possible inner attempts.
    for (int p = 1; p < outerN; p++) {
        String podSuffix = (p == 1) ? "" : "-pod-${p}"
        priors << "${base}${podSuffix}".toString()                          // inner attempt 1
        for (int i = 2; i <= maxInner; i++) {
            priors << "${base}${podSuffix}-attempt-${i}".toString()
        }
    }
    // Current outer attempt's earlier inner attempts (1..innerN-1).
    String thisPodSuffix = (outerN == 1) ? "" : "-pod-${outerN}"
    if (innerN > 1) {
        priors << "${base}${thisPodSuffix}".toString()                      // inner attempt 1
        for (int i = 2; i < innerN; i++) {
            priors << "${base}${thisPodSuffix}-attempt-${i}".toString()
        }
    }
    return priors
}

def runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312", String postTag="", boolean useClusterDurations=false)
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
        if (stageName.startsWith("CPU-")) {
            sh "ln -s /usr/local/cuda/compat/lib.real /usr/local/cuda/compat/lib"
        } else {
            sh "nvidia-smi && nvidia-smi -q && nvidia-smi topo -m"
        }
        sh "df -h"

        // setup HF_HOME to cache model and datasets
        // init the huggingface cache from nfs, since the nfs is read-only, and HF_HOME needs to be writable, otherwise it will fail at creating file lock
        sh "mkdir -p ${HF_HOME} && ls -alh ${HF_HOME}"
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "rsync -r ${MODEL_CACHE_DIR}/hugging-face-cache/ ${HF_HOME}/ && ls -lh ${HF_HOME}")
        sh "df -h"

        // install package
        sh "env | sort"
        sh "which python3"
        sh "python3 --version"

        sh "rm -rf results-${stageName}.tar.gz ${stageName}/*"
        // download TRT-LLM tarfile
        def tarName = BUILD_CONFIGS[config][TARNAME]
        def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarName}"
        timeout(time: 30, unit: 'MINUTES') {
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
        }
        sh "cd ${llmPath} && tar -zxf ${tarName}"

        // install python package
        timeout(time: 45, unit: 'MINUTES') {
            if (env.alternativeTRT) {
                sh "cd ${llmSrc} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
            }
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmSrc} && pip3 install -r requirements-dev.txt")
            // Gateway adapters are opt-in extras excluded from requirements.txt;
            // each gateway declares its pins in a dedicated
            // requirements-<gateway>.txt, and a test stage installs exactly
            // zero or one gateway file so every adapter is tested under the
            // dependency set its real opt-in users receive. A gateway whose
            // pins co-resolve with the default environment (SMG today) is
            // installed in the shared stages so its unit tests run from the
            // regular shard pool instead of being skipped at collection; a
            // gateway whose pins conflict with the default environment (for
            // example a protobuf major-version floor or a custom package
            // index) must instead install its file behind a dedicated stage
            // guard and skip this one (see the Ray install below for the
            // stage-scoped pattern).
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmSrc} && pip3 install -r requirements-grpc-smg.txt")
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install opencv-python-headless")
            if (stageName.contains("-Ray-")) {
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install ray[default]==2.55.1")
                trtllm_utils.llmExecStepWithRetry(pipeline, script: """
                    mambaArch=\$(uname -m)
                    pip3 install --no-deps \
                        "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2/causal_conv1d-1.6.1%2Bcu13torch26.04cxx11abiTRUE-cp312-cp312-linux_\${mambaArch}.whl" \
                        "https://github.com/state-spaces/mamba/releases/download/v2.3.0/mamba_ssm-2.3.0%2Bcu13torch26.01cxx11abiTRUE-cp312-cp312-linux_\${mambaArch}.whl"
                """)
            }
            if (!skipInstallWheel) {
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && pip3 install --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl")
            }
            if (stageName.contains("-ModelExpress-")) {
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install modelexpress==${MODEL_EXPRESS_VERSION}")
                // ModelExpress 0.4.1 imports nixl._api, while requirements-dev.txt
                // installs only the nixl-cu13 backend. Install the matching
                // namespace shim without pulling the unused CUDA 12 backend.
                trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install --no-deps nixl==${MODEL_EXPRESS_NIXL_VERSION}")
            }
        }

        trtllm_utils.llmExecStepWithRetry(pipeline, script: "git config --global --add safe.directory \"*\"")
    }

    if (testFilter[(DEBUG_MODE)]) {
        stage("Interactive Debug Session")
        {
            testFilter[(DEBUG_MODE)] = false

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
        def hasUnrerunFailure = false
        def infraDryRun = isInfraDryRun()
        if (infraDryRun) {
            testList = INFRA_DRY_RUN_TEST_CONTEXT
            splitId = 1
            splits = 1
            perfMode = false
        }

        // When useClusterDurations is set, use a per-cluster durations file keyed on
        // partition.clusterName (e.g. "oci-hsg", "dlcluster").  This lets each cluster
        // build its own timing baseline so sharding is not skewed by timings collected
        // on different hardware.  Falls back to the shared .test_durations when unset.
        def clusterDurationsArgs = []
        def clusterDurationsPath = ""
        String clusterNameForDurations = null
        if (useClusterDurations) {
            def partition = SlurmConfig.resolvePlatform(platform)
            def clusterKey = partition.clusterName.replaceAll('[^a-zA-Z0-9]', '_')
            clusterNameForDurations = clusterKey
            clusterDurationsPath = "${llmSrc}/tests/integration/defs/.test_durations_${clusterKey}"
            clusterDurationsArgs = ["--durations-path ${clusterDurationsPath}"]
        }

        def testDBList = renderTestDB(pipeline, testList, llmSrc, stageName, null, clusterNameForDurations)
        def waivesFilePath = infraDryRun
            ? "${llmSrc}/infra_dry_run_waives.txt"
            : "${llmSrc}/tests/integration/test_lists/waives.txt"

        if (infraDryRun) {
            sh ": > ${waivesFilePath}"
        } else {
            // Download and Merge waives.txt
            mergeWaivesTxt(pipeline, llmSrc, stageName)

            // Add passed test list from previous pipeline run to the waives.txt
            if (testFilter[(REUSE_TEST)] != false) {
                reusePassedTestResults(llmSrc, stageName, waivesFilePath, postTag)
            }
        }

        // Process shard test list and create separate files for regular and isolate tests
        def preprocessedLists = processShardTestList(llmSrc, testDBList, splitId, splits, perfMode, clusterDurationsPath)

        // Test Coverage
        def TRTLLM_WHL_PATH = sh(returnStdout: true, script: "pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2").replaceAll("\\s","")
        sh "echo ${TRTLLM_WHL_PATH}"
        def coverageConfigFile = "${llmSrc}/${stageName}/.coveragerc"
        sh "mkdir -p ${llmSrc}/${stageName} && touch ${coverageConfigFile}"
        // CBTS stages render coveragerc.template; all other stages leave the rcfile empty (no coverage). Keep in sync with the SLURM branch.
        if (isCbtsStage(stageName)) {
            // K8s runner knows TRTLLM_WHL_PATH here, so all placeholders are substituted at controller time (no worker-side sed).
            sh """
                # A sentinel left in a reused workspace would suppress this stage's writes.
                mkdir -p ${WORKSPACE}/${stageName} && rm -f ${WORKSPACE}/${stageName}/${CBTS_STOP_FILE_NAME}
                cp ${llmSrc}/jenkins/scripts/cbts/coverage_utils/coveragerc.template ${coverageConfigFile}
                sed -i \\
                    -e 's|@TRTLLM_WHEEL_PATH@|${TRTLLM_WHL_PATH}|g' \\
                    -e 's|@JOB_WORKSPACE@|${WORKSPACE}/${stageName}|g' \\
                    -e 's|@STAGE_NAME@|${stageName}|g' \\
                    ${coverageConfigFile}
                cat ${coverageConfigFile}
            """
        }
        echoNodeAndGpuInfo(pipeline, stageName)

        // Allocate a unique port section for this container to avoid port conflicts
        def hostNodeName = getHostNodeName()
        def containerPortStart = getStartingPortForHost(hostNodeName, stageName)
        def containerPortNum = GlobalState.PORT_SECTION_SIZE
        def uploadPath = UPLOAD_PATH.replaceFirst("sw-tensorrt-generic/llm-artifacts/LLM/", "")

        // Some clusters do not allow dmesg -C so we add || true
        // Temporarily disable to reduce the log size
        // sh 'if [ "$(id -u)" -eq 0 ]; then dmesg -C || true; fi'
        def extraArgs = [*clusterDurationsArgs]
        if (ENABLE_UPLOAD_TEST_RESULTS && !testFilter[(DETAILED_LOG)]) {
            extraArgs += [
                "--capture=fd",
                "--s3-upload-path=${uploadPath}/${stageName}",
                "--s3-upload-mode=deferred",
            ]
        }
        def pytestCommand = getPytestBaseCommandLine(
            llmSrc,
            stageName,
            waivesFilePath,
            perfMode,
            "${WORKSPACE}/${stageName}",
            coverageConfigFile,
            "",  // pytestUtil
            extraArgs,  // extraArgs
            containerPortStart,
            containerPortNum
        )

        // Only add --test-list if there are regular tests to run
        if (preprocessedLists.regularCount > 0) {
            pytestCommand += ["--test-list=${preprocessedLists.regular}"]
            pytestCommand += getInfraDryRunPytestTargets(preprocessedLists.regular)
        }

        def containerPIP_LLM_LIB_PATH = sh(script: "pip3 show tensorrt_llm | grep \"Location\" | awk -F\":\" '{ gsub(/ /, \"\", \$2); print \$2\"/tensorrt_llm/libs\"}'", returnStdout: true).replaceAll("\\s","")
        def containerLD_LIBRARY_PATH = sh(script: "echo \${LD_LIBRARY_PATH}", returnStdout: true).replaceAll("\\s","")
        if (!containerLD_LIBRARY_PATH.contains("${containerPIP_LLM_LIB_PATH}:")) {
            echo "Prepend ${containerPIP_LLM_LIB_PATH} into \${LD_LIBRARY_PATH}"
            containerLD_LIBRARY_PATH = "${containerPIP_LLM_LIB_PATH}:${containerLD_LIBRARY_PATH}"
        }
        containerLD_LIBRARY_PATH = containerLD_LIBRARY_PATH.replaceAll(':+$', '')
        def testEnvironment = ["LD_LIBRARY_PATH=${containerLD_LIBRARY_PATH}"]
        if (infraDryRun) {
            testEnvironment += ["stageName=${stageName}"]
        }
        withEnv(testEnvironment) {
            withCredentials([
                string(credentialsId: 'TRTLLM_HF_TOKEN', variable: 'HF_TOKEN'),
                string(credentialsId: 'svc_tensorrt-swift-stack-key', variable: 'S3_SECRET_KEY'),
                string(credentialsId: 'llm_evaltool_repo_url', variable: 'EVALTOOL_REPO_URL')
            ]) {
                sh "env | sort"
                try {
                    // Sentinel that the watcher polls to know pytest has exited
                    // (success or failure). Lives outside ${stageName}/ so the
                    // `rm -rf ${stageName}/` at pytest startup doesn't wipe it.
                    def pytestDoneFile = "${WORKSPACE}/.pytest-done-${stageName}"
                    def progressTar = "results-${stageName}${postTag}-progress.tar.gz"
                    def progressUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/test-results/${progressTar}"
                    sh "rm -f ${pytestDoneFile}"
                    try {
                        if (preprocessedLists.regularCount > 0) {
                            // Run pytest in the foreground while a shell subshell
                            // watcher periodically tars + uploads results.xml. The
                            // watcher is a background subshell of the same `sh`
                            // step (not a Groovy parallel branch) so Blue Ocean
                            // renders the stage as a single box rather than a
                            // nested parallel split.
                            withCredentials([usernamePassword(
                                    credentialsId: 'urm-artifactory-creds',
                                    usernameVariable: 'ART_USER',
                                    passwordVariable: 'ART_PASS')]) {
                                sh """
                                    set +e
                                    export STAGE_NAME='${stageName}'
                                    export PROGRESS_TAR='${progressTar}'
                                    export PROGRESS_URL='${progressUrl}'
                                    export TIMEOUT_XML_SCRIPT='${llmSrc}/jenkins/scripts/generate_timeout_xml.py'
                                    export POST_TAG='${postTag}'
                                    # ---- background watcher ----
                                    PROGRESS_DONE_FILE='${pytestDoneFile}' \\
                                    PROGRESS_INTERVAL=${PROGRESS_UPLOAD_INTERVAL_SEC} \\
                                    LABEL_PREFIX='checkpoint' \\
                                    XML_PATH='${WORKSPACE}/${stageName}/results.xml' \\
                                    bash '${llmSrc}/jenkins/scripts/progress_upload_watcher.sh' &
                                    WATCHER_PID=\$!

                                    # ---- foreground pytest ----
                                    rm -rf '${stageName}/'
                                    cd '${llmSrc}/tests/integration/defs'
                                    ${pytestCommand.join(" ")}
                                    rc=\$?

                                    touch '${pytestDoneFile}'
                                    wait \$WATCHER_PID 2>/dev/null || true

                                    # ---- immediate final snapshot of run 1 ----
                                    if [ -f '${WORKSPACE}/${stageName}/results.xml' ]; then
                                        LABEL='run1 final snapshot' FINAL_SNAPSHOT=1 \\
                                        bash '${llmSrc}/jenkins/scripts/progress_upload_snapshot.sh' || true
                                    fi

                                    exit \$rc
                                """
                            }
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
                            sh "touch ${pytestDoneFile}"
                        }
                    } catch (InterruptedException e) {
                        throw e
                    } catch (Exception e) {
                        def isRerunFailed = rerunFailedTests(
                            stageName, llmSrc, pytestCommand, "results.xml", "regular", postTag)
                        if (isRerunFailed) {
                            catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                error "Regular tests failed after rerun attempt"
                            }
                            rerunFailed = true
                        } else if (generateTimeoutTestResultXml(pipeline, stageName)) {
                            // Rerun passed but the first run had a timeout: mark this
                            // stage FAILURE so "[${stageName}] Run Pytest" turns red,
                            // not just the enclosing parent stage.
                            catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                error "Some tests terminated unexpectedly, please check the test report."
                            }
                        } else if (fileExists("${WORKSPACE}/${stageName}/rerun/regular/rerun_0.txt")) {
                            // Failures that finished (not a timeout) but were never
                            // rerun because duration > 10 min and no known failure
                            // signature matched: results.xml still carries their
                            // original <failure>, but neither branch above fires for
                            // them, so without this the stage silently reports green.
                            catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
                                error "Some tests failed and were not eligible for rerun (duration > 10 min, no matching failure signature), please check the test report."
                            }
                            hasUnrerunFailure = true
                        }
                    }

                    // Run the isolated tests if exists
                    if (preprocessedLists.isolateCount > 0) {
                        stage ("[${stageName}] Run Pytest (Isolated)") {
                            echo "There are ${preprocessedLists.isolateCount} isolated tests to run"
                            def isolatedResult = runIsolatedTests(
                                pipeline, preprocessedLists, pytestCommand, llmSrc, stageName, postTag)
                            rerunFailed = isolatedResult.rerunFailed || rerunFailed
                            hasUnrerunFailure = isolatedResult.hasUnrerunFailure || hasUnrerunFailure
                        }
                    } else {
                        echo "No isolated tests to run for stage ${stageName}"
                        noIsolateTests = true
                    }

                    if (noRegularTests && noIsolateTests) {
                        error "No tests were executed for stage ${stageName}, please check the test list and test-db rendering result."
                    }
                } finally {
                    if (ENABLE_UPLOAD_TEST_RESULTS && !testFilter[(DETAILED_LOG)]) {
                        sh """
                            python3 ${llmSrc}/tests/test_common/s3_output.py \
                                --drain-spool "${WORKSPACE}/${stageName}" || true
                        """
                    }
                }
            }

            // CBTS coverage liveness signal: log this stage's touch counts (no artifacts); never fails the stage (|| true).
            if (isCbtsStage(stageName)) {
                sh """
                    cd ${WORKSPACE}/${stageName} && \
                    python3 ${llmSrc}/jenkins/scripts/cbts/coverage_utils/pystart_report.py \
                        --glob '.cbtscov.${stageName}*' || true
                """
            }
        }

        // Generate comprehensive rerun report if any reruns occurred
        stage ("Generate Report") {
            timeout(time: 15, unit: 'MINUTES'){
                generateRerunReport(stageName, llmSrc)
            }
        }

        if (rerunFailed) {
            error "Some tests still failed after rerun attempts, please check the test report."
        }

        if (fileExists("${stageName}/results-timeout.xml") || generateTimeoutTestResultXml(pipeline, stageName)) {
            error "Some tests terminated unexpectedly, please check the test report."
        }

        if (hasUnrerunFailure) {
            error "Some tests failed and were not eligible for rerun (duration > 10 min, no matching failure signature), please check the test report."
        }

        if (perfMode) {
            // Only PyTorch perf stages remain; the TensorRT perf baseline was removed.
            basePerfFilename = "base_perf_pytorch.csv"
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
                if (fileExists("${stageName}/perf_script_test_results.csv")) {
                    sh """
                        python3 ${llmSrc}/tests/integration/defs/perf/create_perf_comparison_report.py \
                        --output_path ${stageName}/report.pdf \
                        --files ${stageName}/perf_script_test_results.csv \
                        ${basePerfPath}
                    """
                } else {
                    echo "No perf script test results to create report"
                }
            }
        }
    }

}


// Single-attempt test path. The infra-failure retry loop now lives one layer up
// in `runKubernetesPodWithInfraRetry` so that retries get a fresh K8s pod
// (recovering from ImagePullBackOff, pod eviction, OOMKilled, JNLP disconnect,
// etc.). Callers that want pod-level retry pass through `postTag` (already
// composed with an attempt tag by the helper) and `isFinalAttempt` (so this
// function's `cacheErrorAndUploadResult` can suppress synthetic stage-fail XML
// and junit() for intermediate retryable failures).
def runLLMTestlistOnPlatform(pipeline, platform, testList, config=VANILLA_CONFIG, perfMode=false, stageName="Undefined", splitId=1, splits=1, skipInstallWheel=false, cpver="cp312", postTag="", boolean isFinalAttempt=true, Map retryContext=null, boolean useClusterDurations=false)
{
    cacheErrorAndUploadResult(stageName, {
        runLLMTestlistOnPlatformImpl(pipeline, platform, testList, config, perfMode, stageName, splitId, splits, skipInstallWheel, cpver, postTag, useClusterDurations)
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
            ls -al ${stageName}/
            if ! grep -q '<testcase' ${stageName}/results.xml; then
                rm ${stageName}/results.xml || true
            fi
        """
        def llmPath = sh (script: "realpath .", returnStdout: true).trim()
        def llmSrc = "${llmPath}/${LLM_ROOT}${config}/TensorRT-LLM/src"
        // CPP tests will generate test result in ${llmSrc}/cpp/build_backup/, move these files to job result folder
        sh "ls -al ${llmSrc}/cpp/build_backup/ || true"
        sh "ls -al ${llmSrc}/cpp/build/ || true"
        // Sed for CPP test result
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/\" classname=\"/\" classname=\"${stageName}./g' *.xml || true"
        sh "cd ${llmSrc}/cpp/build_backup/ && sed -i 's/testsuite name=\"[^\"]*\"/testsuite name=\"${stageName}\"/g' *.xml || true"
        // Sed for Pytest result
        sh "cd ${stageName} && sed -i 's/testsuite name=\"pytest\"/testsuite name=\"${stageName}\"/g' *.xml || true"
        // Copy CPP test result
        sh "cp ${llmSrc}/cpp/build_backup/*.xml ${stageName} || true"
        sh "ls -al ${stageName}/"
    }, false, postTag, isFinalAttempt, retryContext)
}


def checkPipInstall(pipeline, wheel_path, version_override)
{
    def wheelArtifactLinks = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/${wheel_path}"
    def versionLocal = version_override?.contains("+") ?
        version_override.substring(version_override.indexOf("+") + 1) : ""
    withEnv(["TRTLLM_VERSION_LOCAL=${versionLocal}"]) {
        trtllm_utils.llmExecStepWithRetry(pipeline, script: """
            cd ${LLM_ROOT}/tests/unittest && \
            python3 test_pip_install.py --wheel_path ${wheelArtifactLinks} --version_local "\${TRTLLM_VERSION_LOCAL}"
            """)
    }
}


def pythonVersionFromCpver(cpver)
{
    if (cpver == "cp310") {
        return "3.10"
    }
    if (cpver == "cp312") {
        return "3.12"
    }
    error "Unsupported Python ABI for Kitmaker dry run: ${cpver}"
}


def runKitmakerWheelDryRun(pipeline, wheel_path, python_bin, publish_to)
{
    def wheelUrl = "https://urm.nvidia.com/artifactory/${UPLOAD_PATH}/${wheel_path}"
    def releaseScriptsDir = "release-scripts"

    echo "Running Kitmaker wheel dry run for ${wheelUrl} with publish target ${publish_to}"
    def kitmakerDryRunMetadata = null
    stage("Kitmaker Publish Dry Run") {
        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
            sh "rm -rf ${releaseScriptsDir}"
            trtllm_utils.checkoutSource(RELEASE_SCRIPT_REPO, RELEASE_SCRIPT_COMMIT, releaseScriptsDir, false, true)
            trtllm_utils.llmExecStepWithRetry(
                pipeline,
                script: "${python_bin} -m pip install -r ${releaseScriptsDir}/requirements.txt")
            withCredentials([string(credentialsId: KITMAKER_CREDENTIALS_ID, variable: 'KITMAKER_API_TOKEN')]) {
                retry(3) {
                    echo "Publishing Kitmaker wheel dry run for ${wheelUrl}"
                    def resultData = sh(script: """${python_bin} ${releaseScriptsDir}/kitmaker_wheel.py publish \
                    --pic-email ${KITMAKER_DRY_RUN_PIC_EMAIL} \
                    --publish-to ${publish_to} \
                    --size large \
                    --wheel-urls ${wheelUrl} \
                    --no-upload
                """, returnStdout: true).trim()
                    echo "${resultData}"
                    def resultJson = readJSON text: resultData
                    kitmakerDryRunMetadata = [
                        releaseUuid: resultJson["release_uuid"],
                        releaseScriptsDir: releaseScriptsDir,
                        pythonBin: python_bin,
                    ]
                }
            }
        }
    }
    return kitmakerDryRunMetadata
}


def isKitmakerWheelDryRunEnabled()
{
    return RELEASE_SCRIPT_REPO && RELEASE_SCRIPT_COMMIT
}


def checkKitmakerWheelDryRun(pipeline, kitmakerDryRunMetadata)
{
    if (!kitmakerDryRunMetadata?.releaseUuid) {
        echo "Skipping Kitmaker wheel dry run check because publish did not return a release UUID"
        return
    }
    stage("Kitmaker Check Dry Run") {
        catchError(buildResult: 'SUCCESS', stageResult: 'FAILURE') {
            retry(3) {
                echo "Checking Kitmaker wheel dry run ${kitmakerDryRunMetadata.releaseUuid}"
                withCredentials([string(credentialsId: KITMAKER_CREDENTIALS_ID, variable: 'KITMAKER_API_TOKEN')]) {
                    sh """${kitmakerDryRunMetadata.pythonBin} ${kitmakerDryRunMetadata.releaseScriptsDir}/kitmaker_wheel.py check \
                        ${kitmakerDryRunMetadata.releaseUuid} \
                        --wait \
                        --ignore-missing-logs-error
                    """
                }
            }
        }
    }
}


def runLLMBuild(
    pipeline,
    cpu_arch,
    reinstall_dependencies=false,
    wheel_path="",
    version_override="",
    cpver="cp312",
    plat_name="")
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
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && pip3 install -r requirements-grpc-smg.txt")
    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
    }
    buildArgs = "--clean --nixl_root /opt/nvidia/nvda_nixl"
    if (cpu_arch == AARCH64_TRIPLE) {
        buildArgs += " -a '90-real;100-real;103-real;120-real'"
    }
    def platNameArg = plat_name ? " --plat-name ${plat_name}" : ""

    withEnv([
        "TRTLLM_BUILD_SOURCE_COMMIT=${env.gitlabCommit}",
        "TRTLLM_VERSION_OVERRIDE=${version_override}",
    ]) {
        withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'CONAN_LOGIN_USERNAME', passwordVariable: 'CONAN_PASSWORD')]) {
            trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && python3 scripts/build_wheel.py --version-override \"\${TRTLLM_VERSION_OVERRIDE}\" --use_ccache -G Ninja -j ${BUILD_JOBS} -D 'WARNING_IS_ERROR=ON' --extra-cmake-vars ENABLE_BOLT_COMPATIBLE=ON ${buildArgs}${platNameArg}")
        }
    }
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    def wheelName = sh(returnStdout: true, script: 'cd tensorrt_llm/build && ls -1 *.whl').trim()
    echo "uploading ${wheelName} to ${cpu_arch}/${wheel_path}"
    trtllm_utils.uploadArtifacts("tensorrt_llm/build/${wheelName}",  "${UPLOAD_PATH}/${cpu_arch}/${wheel_path}")
    def uploadedWheelPath = "${cpu_arch}/${wheel_path}${wheelName}"
    def kitmakerDryRunMetadata = null
    if (version_override?.contains("+")) {
        echo "Skipping Kitmaker wheel dry run for local version '${version_override}'"
    } else if (!isKitmakerWheelDryRunEnabled()) {
        echo "Skipping Kitmaker wheel dry run because releaseScriptRepo or releaseScriptCommit is not set"
    } else {
        def kitmakerPython = "tensorrt_llm/.venv-${pythonVersionFromCpver(cpver)}/bin/python3"
        kitmakerDryRunMetadata = runKitmakerWheelDryRun(pipeline, uploadedWheelPath, kitmakerPython, KITMAKER_PUBLISH_TO)
    }

    if (reinstall_dependencies) {
        // Test installation in the new environment
        // Reserve CUDA 13.0 torch and torchvision packages
        def pip_keep = "^pip==|^torch==|^torchvision=="
        def remove_trt = "rm -rf /usr/local/tensorrt"
        if (env.alternativeTRT) {
            pip_keep += "|^tensorrt=="
            remove_trt = "echo keep /usr/local/tensorrt"
        }
        sh "bash -c 'pip3 list --format=freeze | grep -Ev \"${pip_keep}\" | xargs -r pip3 uninstall -y'"
        sh "bash -c 'yum remove -y libcudnn* libnccl* libcublas* && ${remove_trt}'"
    }

    // Test preview installation
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "#!/bin/bash \n" + "cd tensorrt_llm/ && pip3 install pytest build/tensorrt_llm-*.whl")
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    // Upload attribution files when they exist
    def attrDir = "tensorrt_llm/cpp/build/attribution"
    def wheelBase = wheelName.replace('.whl', '')
    for (f in ["missing_files.json", "import_payload.json", "file_mappings.json"]) {
        if (fileExists("${attrDir}/${f}")) {
            trtllm_utils.uploadArtifacts("${attrDir}/${f}", "${UPLOAD_PATH}/${cpu_arch}/attribution/${wheel_path}${wheelBase}/")
        }
    }
    checkKitmakerWheelDryRun(pipeline, kitmakerDryRunMetadata)

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
        // Reserve CUDA 13.0 torch and torchvision packages
        def pip_keep = "^pip==|^torch==|^torchvision=="
        def remove_trt = "rm -rf /usr/local/tensorrt"
        if (env.alternativeTRT) {
            pip_keep += "|^tensorrt=="
            remove_trt = "echo keep /usr/local/tensorrt"
        }
        sh "bash -c 'pip3 list --format=freeze | grep -Ev \"${pip_keep}\" | xargs -r pip3 uninstall -y'"
        sh "bash -c 'yum remove -y libcudnn* libnccl* libcublas* && ${remove_trt}'"
    }
    //WAR: remove python3-pygments first since it is installed in NGC PyTorch image
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get remove -y python3-pygments")

    // Test preview installation
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'pip3 install pytest tensorrt_llm-*.whl'")
    if (env.alternativeTRT) {
        sh "bash -c 'pip3 show tensorrt || true'"
    }

    def pkgUrl = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${linuxPkgName}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget -nv ${pkgUrl}")
    sh "tar -zvxf ${linuxPkgName}"

    // TODO: The steps below drove the removed TensorRT engine flow (trtllm-build / examples/run.py).
    // When re-enabling this sanity check, use PyTorch backend test samples instead
    // (e.g. examples/llm-api/quickstart_example.py).
    // trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core/gpt && python3 ../../../generate_checkpoint_config.py --architecture GPTForCausalLM --dtype float16'")
    // trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core//gpt && trtllm-build --model_config config.json --log_level verbose'")
    // trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c 'cd tensorrt_llm/examples/models/core/gpt && python3 ../../../run.py --max_output_len 4 --end_id -1'")
}

def checkStageNameSet(stageNames, jobKeys, paramName) {
    echo "Validate stage names for the passed GitLab bot params [${paramName}]."
    def unmatchedNames = stageNames.findAll { pattern ->
        if (pattern.contains('*')) {
            // Wildcard pattern: check that it matches at least one stage
            !jobKeys.any { key -> stageMatchesPattern(key, pattern) }
        } else {
            // Exact name: check that it exists
            !(pattern in jobKeys)
        }
    }
    if (unmatchedNames) {
        def sortedJobKeys = jobKeys.sort()
        throw new Exception("Cannot find the stage names [${unmatchedNames}] from the passed params [${paramName}]. Available stage names (${sortedJobKeys.size()} total):\n${sortedJobKeys.collect { "    ${it}" }.join('\n')}")
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
                    withCredentials([usernamePassword(
                        credentialsId: ARTIFACTORY_CREDENTIALS_ID,
                        usernameVariable: 'USERNAME',
                        passwordVariable: 'PASSWORD'
                    )]) {
                        sh """
                            set +x
                            echo "\$PASSWORD" | docker login ${ARTIFACTORY_DOCKER_HOST} -u "\$USERNAME" --password-stdin
                        """
                    }
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

// Retry-aware K8s pod launcher. Mirrors the SLURM retry loop's classification
// (classify(ex, InfraFailure.K8S)), budget (K8S_INFRA_RETRY_MAX), and backoff,
// but operates at the pod-launch level so transient pod failures
// (ImagePullBackOff, eviction, OOMKilled, JNLP disconnect, node NotReady) get
// a fresh pod on each retry rather than
// retrying the test body inside a dying pod.
//
// `runner` is invoked with `(attemptTag, isFinalAttempt, retryContext)`. Callers append
// `attemptTag` to any postTag they pass into cacheErrorAndUploadResult so each
// attempt's tar and ensureStageResultNotUploaded guard key are unique;
// `isFinalAttempt` lets cacheErrorAndUploadResult suppress synthetic stage-fail
// XML and junit() for intermediate retryable failures so the per-build test
// report isn't poisoned by transient infra blips that the next attempt
// recovers from.
//
// Options (Map; all optional):
//   singleAttempt (bool, default false) skip the retry loop entirely; run
//                                       once with attemptTag="" and
//                                       isFinalAttempt=true. Used by SLURM
//                                       dispatcher closures so the outer
//                                       pod-level retry does not nest on top
//                                       of the inner SLURM retry and inflate
//                                       the worst-case attempt budget.
def runKubernetesPodWithInfraRetry(Map opts = [:], pipeline, podSpec, containerName, String stageName, Closure runner)
{
    boolean singleAttempt = opts.singleAttempt ?: false
    // SLURM dispatcher pods opt in to off-pod resource reconciliation: on a
    // mid-run pod death their SLURM job / Jenkins node would otherwise leak.
    boolean slurmDispatcher = opts.slurmDispatcher ?: false
    // Per-stage override of the K8s pod-level infra-retry budget (opts.infraRetryMax,
    // 0 = no retry) so resource-scarce pools can cap or disable stage retries. Applies
    // to the outer test-pod retry loop below; the singleAttempt launch-retry keeps the
    // global (relaunching a dispatcher pod doesn't tax the scarce test hardware). SLURM
    // dispatchers carry the same opt through to their inner SLURM retry via retryContext.
    int k8sInfraRetryMax = resolveInfraRetryMax(InfraFailure.K8S, opts.infraRetryMax as Integer)

    // DEBUG_MODE preserves the existing 2-hour-input human-inspection workflow
    // inside runLLMTestlistOnPlatform's finallyRunner: a single attempt only.
    if (testFilter[(DEBUG_MODE)]) {
        trtllm_utils.launchKubernetesPod(pipeline, podSpec, containerName, { runner("", true, null) })
        return
    }

    // singleAttempt opts out of the outer pod-level *execution* retry: the inner
    // retry (e.g. SLURM) owns re-runs, and re-running after work has started
    // could double-submit. But a failure *before* the runner body begins is a
    // pod/agent launch failure (the JNLP agent never came online, the node was
    // pulled during provisioning, etc.): nothing has been dispatched yet, so
    // relaunching a fresh pod -- steered off the bad node by host-node avoidance
    // -- is safe and is the only way these recover. Retry launch failures only;
    // once the runner starts (runnerStarted), honor singleAttempt and rethrow.
    if (singleAttempt) {
        def avoidedKubernetesHostNodes = []
        def launchAttempt = 0
        while (true) {
            launchAttempt++
            Map attemptPlacementContext = [runnerStarted: false]
            // Declared above the try so the catch can reference it when reconciling
            // a dispatcher-pod death: Groovy block-scopes a try-local `def`, so a
            // declaration inside the try is not visible in the catch (it would
            // resolve as an undefined property and throw MissingPropertyException).
            def attemptPodSpec = null
            try {
                if (launchAttempt > 1 && !avoidedKubernetesHostNodes.isEmpty()) {
                    echo "[INFRA-RETRY] ${stageName}: relaunching pod (attempt ${launchAttempt}), avoiding prior host node(s): ${avoidedKubernetesHostNodes.join(', ')}"
                }
                attemptPodSpec = trtllm_utils.withKubernetesHostNodeExclusion(podSpec, avoidedKubernetesHostNodes)
                if (slurmDispatcher) {
                    // Record the dispatcher pod spec so the off-pod finalizer/sweep can
                    // launch a fresh cleanup pod if this pod dies mid-run.
                    registerSlurmResource(stageName, [podSpec: attemptPodSpec, containerName: containerName])
                }
                trtllm_utils.launchKubernetesPodWithPlacement(pipeline, attemptPodSpec, containerName, attemptPlacementContext, {
                    attemptPlacementContext.runnerStarted = true
                    runner("", true, null)
                })
                return
            } catch (InterruptedException e) {
                throw e
            } catch (Exception e) {
                // Once the runner has started, this is an execution failure the
                // inner retry owns -- honor singleAttempt and do not re-run. But if
                // the dispatcher pod itself died mid-run, its in-pod cleanup could not
                // reach the login node, leaking the SLURM job / Jenkins node. We are
                // back on the parent context here, so reconcile them from a fresh
                // cleanup pod before failing closed.
                if (attemptPlacementContext.runnerStarted) {
                    if (slurmDispatcher && isDispatcherPodFailure(e)) {
                        // Pass this attempt's pod spec explicitly so the finalizer
                        // never depends on the registry entry's podSpec surviving a
                        // prior attempt's deregister.
                        finalizeOrphanedSlurmResource(pipeline, stageName, attemptPodSpec)
                    }
                    throw e
                }
                def c = FailureClassifier.classify(e, InfraFailure.K8S)
                if (c instanceof PipelineInterruption) throw e
                if (!(c instanceof InfraFailure)) {
                    // UserFailure -> don't retry, but leave a trace: a pod/agent
                    // launch exception missing from PATTERN_CATALOG (e.g. a
                    // pod-scheduling timeout) lands here and would otherwise
                    // vanish with no log output at all.
                    echo "[INFRA-RETRY] ${stageName}: pod/agent launch failed before execution with no infra pattern matched (classified as user failure); not retrying. Exception: ${e.toString()}"
                    throw e
                }

                rememberAvoidedKubernetesHostNodes(avoidedKubernetesHostNodes, attemptPlacementContext.lastHostNode, stageName)

                if (launchAttempt > K8S_INFRA_RETRY_MAX) {
                    echo "[INFRA-RETRY] ${stageName}: pod launch failed before execution (${c.detectedPattern}), " +
                         "but max launch retries (${K8S_INFRA_RETRY_MAX}) exhausted after ${launchAttempt} attempts. Failing."
                    throw e
                }
                if (!hasBudgetForInfraRetry(pipeline, stageName, InfraFailure.K8S, c, launchAttempt, K8S_INFRA_RETRY_MAX, 60L * 1000L, true)) {
                    echo "[INFRA-RETRY] ${stageName}: pod launch failed (${c.detectedPattern}) is retryable, " +
                         "but remaining CI timeout budget is too small for a relaunch. Failing without retry."
                    throw e
                }

                echo "[INFRA-RETRY] ${stageName}: pod/agent launch failed before execution on attempt ${launchAttempt}: ${c.detectedPattern}"
                echo "[INFRA-RETRY] ${stageName}: Exception: ${e.toString()}"
                echo "[INFRA-RETRY] ${stageName}: Will relaunch (attempt ${launchAttempt + 1} of ${K8S_INFRA_RETRY_MAX + 1}) after 60s cooldown."
                sleep(60)
            }
        }
    }

    def attempt = 0
    // Severity of the previous attempt's classification, or null on first attempt.
    // Used to compute isFinalAttempt for the next attempt: a PERSISTENT prior
    // failure caps the budget at 1 retry (attempt 2 is final), so the next
    // attempt must not suppress synthetic stage-fail XML / junit().
    def lastSeverity = null
    def avoidedKubernetesHostNodes = []
    while (true) {
        attempt++
        Map attemptPlacementContext = [:]
        try {
            if (attempt > 1) {
                echo "[INFRA-RETRY] ${stageName}: Starting attempt ${attempt} of ${k8sInfraRetryMax + 1}"
                if (!avoidedKubernetesHostNodes.isEmpty()) {
                    echo "[INFRA-RETRY] ${stageName}: avoiding prior Kubernetes host node(s): ${avoidedKubernetesHostNodes.join(', ')}"
                }
            }
            // Attempt 1 keeps the caller-supplied postTag verbatim so the
            // canonical artifact name is unchanged for downstream consumers.
            // Retries append "-pod-N" to dodge the upload-once guard and
            // preserve every attempt's tarball in Artifactory. The "-pod-"
            // separator is distinct from the inner SLURM retry's "-attempt-"
            // suffix so the two nest unambiguously: outer-pod 2 / inner-attempt
            // 2 yields "-pod-2-attempt-2" rather than colliding with "-attempt-2".
            def attemptTag = (attempt == 1) ? "" : "-pod-${attempt}"
            // For attempt 1 we don't yet know whether the failure (if any) will
            // be PERSISTENT, so use the worst-case multi-retry budget. From
            // attempt 2 onward we know the prior classification — if it was
            // PERSISTENT, effectiveMax for THIS attempt is 1, meaning attempt
            // 2 IS the final attempt; pass isFinalAttempt=true so the inner
            // cacheErrorAndUploadResult does not suppress synthetic stage-fail
            // XML / junit() on what would otherwise look (to it) like just
            // another intermediate attempt.
            def effectiveMaxThisAttempt = (lastSeverity == InfraFailure.PERSISTENT) ? Math.min(1, k8sInfraRetryMax) : k8sInfraRetryMax
            boolean isFinalAttempt = (attempt > effectiveMaxThisAttempt)
            def retryContext = [
                scope: InfraFailure.K8S,
                stageName: stageName,
                attempt: attempt,
                backoffMs: 60L * 1000L,
                infraRetryMax: opts.infraRetryMax,
                excludedKubernetesHostNodes: avoidedKubernetesHostNodes.collect(),
            ]
            def attemptPodSpec = trtllm_utils.withKubernetesHostNodeExclusion(podSpec, avoidedKubernetesHostNodes)
            trtllm_utils.launchKubernetesPodWithPlacement(pipeline, attemptPodSpec, containerName, attemptPlacementContext, { runner(attemptTag, isFinalAttempt, retryContext) })
            if (attempt > 1) {
                echo "[INFRA-RETRY] ${stageName}: Succeeded on attempt ${attempt}"
            }
            return
        } catch (InterruptedException e) {
            // User abort / pipeline timeout -- never retry
            throw e
        } catch (Exception e) {
            // classify() handles FlowInterruptedException + exit-code-143 +
            // typed throws + cause-chain pattern matching, returning one of
            // PipelineInterruption / InfraFailure / UserFailure. Scope=K8S
            // ensures we only match catalog rows tagged K8S or BOTH; the new
            // scope-isolation guard inside classify() also prevents a typed
            // SLURM-scoped InfraFailure (e.g. from an inner SLURM retry that
            // exhausted its own budget) from being treated as K8s infra here.
            def c = FailureClassifier.classify(e, InfraFailure.K8S)
            if (c instanceof PipelineInterruption) throw e
            if (!(c instanceof InfraFailure)) {
                // UserFailure -> don't retry, but leave a trace so the decline
                // is diagnosable from the console (see the launch-loop twin above).
                echo "[INFRA-RETRY] ${stageName}: stage failed with no infra pattern matched (classified as user failure); not retrying. Exception: ${e.toString()}"
                throw e
            }

            rememberAvoidedKubernetesHostNodes(avoidedKubernetesHostNodes, attemptPlacementContext.lastHostNode, stageName)

            def effectiveMax = (c.severity == InfraFailure.PERSISTENT) ? Math.min(1, k8sInfraRetryMax) : k8sInfraRetryMax

            if (attempt > effectiveMax) {
                echo "[INFRA-RETRY] ${stageName}: Infrastructure failure (${c.detectedPattern}) " +
                     "but max retries (${effectiveMax}) exhausted after ${attempt} attempts. Failing."
                throw e
            }
            if (!hasBudgetForInfraRetry(pipeline, stageName, InfraFailure.K8S, c, attempt, effectiveMax, 60L * 1000L, true)) {
                echo "[INFRA-RETRY] ${stageName}: Infrastructure failure (${c.detectedPattern}) is retryable, " +
                     "but remaining CI timeout budget is too small for another K8s attempt. Failing without retry."
                throw e
            }

            echo "[INFRA-RETRY] ${stageName}: Infrastructure failure detected on attempt ${attempt}: " +
                 "${c.detectedPattern}"
            echo "[INFRA-RETRY] ${stageName}: Exception: ${e.toString()}"
            echo "[INFRA-RETRY] ${stageName}: Will retry (attempt ${attempt + 1} of ${effectiveMax + 1}) after 60s cooldown."

            // Remember severity so the next attempt's isFinalAttempt is correct.
            lastSeverity = c.severity
            sleep(60)
        }
    }
}

def buildStageConfigs(stageName, platform, testlist, testCount, gpuCount, nodeCount, runWithSbatch=false, useClusterDurations=false) {
    def configs = [:]
    for (int k = 1; k <= testCount; k++) {
        def key = "${stageName}-${k}"
        configs[key] = [platform, testlist, k, testCount, gpuCount, nodeCount, runWithSbatch, useClusterDurations]
    }
    return configs
}

// Infra-scoped fail-fast (inner/branch layer). Runs `jobs` under `parallel` so a
// branch whose post-retry failure is a positive infra abort
// (FailureClassifier.isDeferrableInfra) is recorded and swallowed -- its siblings
// keep running instead of being SIGTERMed by failFast. A genuine test/build
// failure (or an unclassified one) is rethrown unchanged, so failFast stays fully
// active for real failures; an interrupt (e.g. a sibling's own fail-fast SIGTERM)
// is also rethrown and never swallowed. After the join, a sub-job that saw ONLY
// infra aborts and no real failure resolves to UNSTABLE (coverage incomplete, not
// a failure) so the parent layer (L0_MergeRequest.launchJob) can spare the healthy
// sibling architecture; a mixed sub-job already threw on its real failure and is
// FAILURE (currentBuild.result worst-of semantics won't downgrade it).
//
// Scope: classify() is scope-filtered, so each branch is classified under its real
// execution scope, passed per-stage in `stageScopes` (built in launchTestJobs from
// opts.slurmDispatcher). Every branch is checked under K8S -- this is where the
// motivating pod-scheduling abort (KubernetesClientTimeoutException) matches, and
// keeps K8s-pod aborts of a SLURM dispatcher pod deferrable exactly as before.
// SLURM dispatcher stages are ADDITIONALLY checked under SLURM so a SLURM-scoped
// abort (SSH outage to the head node, slurm_track ssh exit 255, monitor loss while
// the job is still active) defers too instead of cascading via failFast. The inner
// SLURM retry (runLLMTestlistOnSlurm) has already been exhausted by the time the
// branch body returns here, so this stays post-retry, mirroring the K8s path.
// Stages absent from stageScopes default to K8S-only (phase-1 behavior). Gated on
// ENABLE_INFRA_SCOPED_FAILFAST; off restores today's behavior exactly (plain
// failFast + parallel, no wrapping, no UNSTABLE).
def runBranchesWithInfraDefer(Map jobs, boolean failFast, Map stageScopes = [:]) {
    if (!ENABLE_INFRA_SCOPED_FAILFAST) {
        jobs.failFast = failFast
        parallel jobs
        return
    }
    // CPS serializes parallel-branch continuations onto a single VM thread, so a
    // plain list append from the catch blocks below is safe -- there is no
    // JVM-level concurrency to guard against here.
    def deferred = []
    def wrapped = jobs.collectEntries { stageName, body ->
        // A SLURM dispatcher stage can abort under either scope: its dispatcher pod
        // is a K8s pod (K8S-scoped aborts) that in turn drives the SLURM job
        // (SLURM-scoped aborts). Check K8S for every stage and SLURM in addition
        // for SLURM stages, so neither class of infra abort cascades.
        boolean slurmScoped = (stageScopes[stageName] == InfraFailure.SLURM)
        [(stageName), {
            try {
                body()
            } catch (InterruptedException e) {
                throw e
            } catch (Exception e) {
                if (FailureClassifier.isDeferrableInfra(e, InfraFailure.K8S) ||
                        (slurmScoped && FailureClassifier.isDeferrableInfra(e, InfraFailure.SLURM))) {
                    def scopeTag = slurmScoped ? "SLURM/K8s" : "K8s"
                    deferred.add([stage: stageName])
                    echo "[INFRA-DEFER] ${stageName}: ${scopeTag} infra abort recorded; " +
                         "siblings continue instead of fail-fast. ${e.toString()}"
                    return
                }
                throw e
            }
        }]
    }
    wrapped.failFast = failFast
    parallel wrapped
    if (deferred) {
        echo "[INFRA-DEFER] ${deferred.size()} stage(s) infra-incomplete " +
             "(${deferred.collect { it.stage }.join(', ')}); marking result UNSTABLE " +
             "(coverage incomplete, no genuine test failure)."
        // Distinguish a per-branch infra blip from a cluster-wide outage: when EVERY
        // branch in the group infra-aborted, the shared infra (a SLURM frontend / a
        // whole cluster) is the likely culprit. Flag it loudly so a re-run isn't
        // burned against still-down infra. (A prospective short-circuit that cancels
        // healthy siblings the moment a quorum aborts is deliberately NOT done here:
        // it would reintroduce the cross-branch SIGTERM cascade this seam removes.
        // Tracked as a follow-up.) NB: compare against jobs.size(), not
        // wrapped.size() -- `wrapped.failFast = failFast` above adds a `failFast`
        // key that `parallel` consumes, inflating wrapped's entry count by one.
        if (deferred.size() == jobs.size()) {
            echo "[INFRA-DEFER] ALL ${jobs.size()} branch(es) infra-aborted; " +
                 "suspected cluster-wide / shared-frontend outage rather than isolated blips."
        }
        currentBuild.result = 'UNSTABLE'
    }
}

def launchTestJobs(pipeline, testFilter, globalVars)
{
    def versionOverride = globalVars[TRTLLM_VERSION_OVERRIDE] ?: ""
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
        "CPU-Generic-x86-1": ["cpu", "l0_cpu", 1, 1],
        "DGX_H100-4_GPUs-CPP-1": ["dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "A10-PyTorch-1": ["a10", "l0_a10", 1, 3],
        "A10-PyTorch-2": ["a10", "l0_a10", 2, 3],
        "A10-PyTorch-3": ["a10", "l0_a10", 3, 3],
        "A30-PyTorch-1": ["a30", "l0_a30", 1, 2],
        "A30-PyTorch-2": ["a30", "l0_a30", 2, 2],
        "A30-CPP-1": ["a30", "l0_a30", 1, 1],
        "A100X-PyTorch-1": ["a100x", "l0_a100", 1, 1],
        "L40S-PyTorch-1": ["l40s", "l0_l40s", 1, 2],
        "L40S-PyTorch-2": ["l40s", "l0_l40s", 2, 2],
        "H100_PCIe-PyTorch-Ray-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-CPP-1": ["h100-cr", "l0_h100", 1, 1],
        // platform, test DB, split, splits, GPU count, ModelExpress sidecars
        "DGX_H100-2_GPUs-PyTorch-ModelExpress-1": ["dgx-h100-x4", "l0_model_express", 1, 1, 2, true],
        "DGX_H100-4_GPUs-PyTorch-ModelExpress-OnDemand-1": ["dgx-h100-x4", "l0_model_express", 1, 1, 4, true],
        "RTX5090-PyTorch-1": ["rtx-5090", "l0_gb202", 1, 1],
        "RTX5080-PyTorch-1": ["rtx-5080", "l0_gb203", 1, 2],
        "RTX5080-PyTorch-2": ["rtx-5080", "l0_gb203", 2, 2],
        // Currently post-merge test stages only run tests with "stage: post_merge" mako
        // in the test-db. This behavior may change in the future.
        "A10-PyTorch-Post-Merge-1": ["a10", "l0_a10", 1, 4],
        "A10-PyTorch-Post-Merge-2": ["a10", "l0_a10", 2, 4],
        "A10-PyTorch-Post-Merge-3": ["a10", "l0_a10", 3, 4],
        "A10-PyTorch-Post-Merge-4": ["a10", "l0_a10", 4, 4],
        "A10-FMHA-Post-Merge-1": ["a10", "l0_a10", 1, 1],
        "A30-CPP-Post-Merge-1": ["a30", "l0_a30", 1, 2],
        "A30-CPP-Post-Merge-2": ["a30", "l0_a30", 2, 2],
        // "A30-Triton-Post-Merge-1": ["a30", "l0_a30", 1, 2],
        // "A30-Triton-Post-Merge-2": ["a30", "l0_a30", 2, 2],
        "A100X-PyTorch-Post-Merge-1": ["a100x", "l0_a100", 1, 1],
        "L40S-PyTorch-Post-Merge-1": ["l40s", "l0_l40s", 1, 1],
        "L40S-FMHA-Post-Merge-1": ["l40s", "l0_l40s", 1, 1],
        "H100_PCIe-FMHA-Post-Merge-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-PyTorch-Perf-1": ["h100-cr", "l0_perf", 1, 1],
        "DGX_H200-8_GPUs-PyTorch-Post-Merge-1": ["dgx-h200-x8", "l0_dgx_h200", 1, 1, 8],
        "DGX_H200-4_GPUs-PyTorch-Post-Merge-1": ["dgx-h200-x4", "l0_dgx_h200", 1, 1, 4],
        "DGX_H200-8_GPUs-PyTorch-PerfSanity-Post-Merge-1": ["dgx-h200-x8", "l0_dgx_h200_perf_sanity", 1, 1, 8],
        // Disable RTXPro6000 stages due to nodes will be offline temporarily.
        // [TODO] Split tests between RTXPro6000 and RTXPro6000D and move reasonable mount of tests to pre-merge.
        // "RTXPro6000-PyTorch-Post-Merge-1": ["rtx-pro-6000", "l0_rtx_pro_6000", 1, 1],
        // "RTXPro6000-4_GPUs-PyTorch-Post-Merge-1": ["rtx-pro-6000-x4", "l0_rtx_pro_6000", 1, 2, 4],
        // "RTXPro6000-4_GPUs-PyTorch-Post-Merge-2": ["rtx-pro-6000-x4", "l0_rtx_pro_6000", 2, 2, 4],
        "RTXPro6000D-PyTorch-1": ["rtx-pro-6000d", "l0_rtx_pro_6000", 1, 1],
        "RTXPro6000D-PyTorch-Post-Merge-1": ["rtx-pro-6000d", "l0_rtx_pro_6000", 1, 1],
        // Disable RTXPro6000D-4_GPUs-PyTorch-Post-Merge-1 and RTXPro6000D-4_GPUs-PyTorch-Post-Merge-2 due to some nodes are offline temporarily.
        // "RTXPro6000D-4_GPUs-PyTorch-Post-Merge-1": ["rtx-pro-6000d-x4", "l0_rtx_pro_6000", 1, 2, 4],
        // "RTXPro6000D-4_GPUs-PyTorch-Post-Merge-2": ["rtx-pro-6000d-x4", "l0_rtx_pro_6000", 2, 2, 4],
    ]

    x86TestConfigs = cbtsResizeSplits(x86TestConfigs)
    parallelJobs = x86TestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "amd64", values[4] ?: 1, key.contains("-Perf-"), values.size() > 5 ? values[5] : false), { attemptTag, isFinalAttempt, retryContext = null ->
        def config = VANILLA_CONFIG
        if (key.contains("single-device")) {
            config = SINGLE_DEVICE_CONFIG
        }
        if (key.contains("llvm")) {
            config = LLVM_CONFIG
        }
        runLLMTestlistOnPlatform(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3], false, "cp312", attemptTag, isFinalAttempt, retryContext)
    }]]}
    fullSet = parallelJobs.keySet()

    x86SlurmTestConfigs = [
        "DGX_H100-PyTorch-1": ["auto:dgx-h100-x1", "l0_h100", 1, 6],
        "DGX_H100-PyTorch-2": ["auto:dgx-h100-x1", "l0_h100", 2, 6],
        "DGX_H100-PyTorch-3": ["auto:dgx-h100-x1", "l0_h100", 3, 6],
        "DGX_H100-PyTorch-4": ["auto:dgx-h100-x1", "l0_h100", 4, 6],
        "DGX_H100-PyTorch-5": ["auto:dgx-h100-x1", "l0_h100", 5, 6],
        "DGX_H100-PyTorch-6": ["auto:dgx-h100-x1", "l0_h100", 6, 6],
        "DGX_H100-PyTorch-Post-Merge-1": ["auto:dgx-h100-x1", "l0_h100", 1, 2],
        "DGX_H100-PyTorch-Post-Merge-2": ["auto:dgx-h100-x1", "l0_h100", 2, 2],
        "DGX_A100-FMHA-Post-Merge-1": ["auto:dgx-a100-x1", "l0_a100", 1, 1],
        "DGX_H100-2_GPUs-PyTorch-Others-1": ["auto:dgx-h100-x2", "l0_dgx_h100", 1, 2, 2],
        "DGX_H100-2_GPUs-PyTorch-Others-2": ["auto:dgx-h100-x2", "l0_dgx_h100", 2, 2, 2],
        "DGX_H100-2_GPUs-PyTorch-GptOss-1": ["auto:dgx-h100-x2", "l0_dgx_h100", 1, 1, 2],
        "DGX_H100-2_GPUs-PyTorch-Ray-1": ["auto:dgx-h100-x2", "l0_dgx_h100", 1, 1, 2],
        "DGX_H100-4_GPUs-PyTorch-DeepSeek-1": ["auto:dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-PyTorch-GptOss-1": ["auto:dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-PyTorch-Others-1": ["auto:dgx-h100-x4", "l0_dgx_h100", 1, 2, 4],
        "DGX_H100-4_GPUs-PyTorch-Others-2": ["auto:dgx-h100-x4", "l0_dgx_h100", 2, 2, 4],
        "DGX_H100-4_GPUs-PyTorch-Ray-1": ["auto:dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_H100-4_GPUs-PyTorch-Post-Merge-1": ["auto:dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
        "DGX_B200-CPP-1": ["auto:dgx-b200-flex", "l0_b200", 1, 1, 1, 1, true],
        "DGX_B200-PyTorch-1": ["auto:dgx-b200-flex", "l0_b200", 1, 9, 1, 1, true],
        "DGX_B200-PyTorch-2": ["auto:dgx-b200-flex", "l0_b200", 2, 9, 1, 1, true],
        "DGX_B200-PyTorch-3": ["auto:dgx-b200-flex", "l0_b200", 3, 9, 1, 1, true],
        "DGX_B200-PyTorch-4": ["auto:dgx-b200-flex", "l0_b200", 4, 9, 1, 1, true],
        "DGX_B200-PyTorch-5": ["auto:dgx-b200-flex", "l0_b200", 5, 9, 1, 1, true],
        "DGX_B200-PyTorch-6": ["auto:dgx-b200-flex", "l0_b200", 6, 9, 1, 1, true],
        "DGX_B200-PyTorch-7": ["auto:dgx-b200-flex", "l0_b200", 7, 9, 1, 1, true],
        "DGX_B200-PyTorch-8": ["auto:dgx-b200-flex", "l0_b200", 8, 9, 1, 1, true],
        "DGX_B200-PyTorch-9": ["auto:dgx-b200-flex", "l0_b200", 9, 9, 1, 1, true],
        "DGX_B200-PyTorch-Post-Merge-1": ["auto:dgx-b200-flex", "l0_b200", 1, 2, 1, 1, true],
        "DGX_B200-PyTorch-Post-Merge-2": ["auto:dgx-b200-flex", "l0_b200", 2, 2, 1, 1, true],
        "DGX_B200-2_GPUs-PyTorch-1": ["auto:dgx-b200-flex", "l0_dgx_b200", 1, 1, 2, 1, true],
        "DGX_B200-4_GPUs-PyTorch-1": ["auto:dgx-b200-flex", "l0_dgx_b200", 1, 3, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-2": ["auto:dgx-b200-flex", "l0_dgx_b200", 2, 3, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-3": ["auto:dgx-b200-flex", "l0_dgx_b200", 3, 3, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Ray-1": ["auto:dgx-b200-flex", "l0_dgx_b200", 1, 1, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Post-Merge-1": ["auto:dgx-b200-flex", "l0_dgx_b200", 1, 4, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Post-Merge-2": ["auto:dgx-b200-flex", "l0_dgx_b200", 2, 4, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Post-Merge-3": ["auto:dgx-b200-flex", "l0_dgx_b200", 3, 4, 4, 1, true],
        "DGX_B200-4_GPUs-PyTorch-Post-Merge-4": ["auto:dgx-b200-flex", "l0_dgx_b200", 4, 4, 4, 1, true],
        "DGX_B200-8_GPUs-PyTorch-1": ["auto:dgx-b200-flex", "l0_dgx_b200", 1, 4, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-2": ["auto:dgx-b200-flex", "l0_dgx_b200", 2, 4, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-3": ["auto:dgx-b200-flex", "l0_dgx_b200", 3, 4, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-4": ["auto:dgx-b200-flex", "l0_dgx_b200", 4, 4, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-Ray-1": ["auto:dgx-b200-flex", "l0_dgx_b200", 1, 1, 8, 1, true],
        "DGX_B200-4_GPUs-Verl-Post-Merge-1": ["auto:dgx-b200-flex", "l0_verl", 1, 1, 4, 1, true],
        "B300-PyTorch-1": ["auto:dgx-b300-flex", "l0_b300", 1, 2, 1, 1, true],
        "B300-PyTorch-2": ["auto:dgx-b300-flex", "l0_b300", 2, 2, 1, 1, true],
        "B300-PyTorch-Post-Merge-1": ["auto:dgx-b300-flex", "l0_b300", 1, 1, 1, 1, true],
        "DGX_B300-4_GPUs-PyTorch-1": ["auto:dgx-b300-flex", "l0_dgx_b300", 1, 1, 4, 1, true],
        "DGX_B300-4_GPUs-PyTorch-Post-Merge-1": ["auto:dgx-b300-flex", "l0_dgx_b300", 1, 2, 4, 1, true],
        "DGX_B300-4_GPUs-PyTorch-Post-Merge-2": ["auto:dgx-b300-flex", "l0_dgx_b300", 2, 2, 4, 1, true],
        // VisualGen PerfSanity post-merge test
        "DGX_B200-8_GPUs-PyTorch-VisualGen-PerfSanity-Post-Merge-1": ["auto:dgx-b200-flex", "l0_b200_visual_gen_perf_sanity", 1, 1, 8, 1, true],
        // Single-GPU Gemma4 PerfSanity post-merge baseline
        "DGX_B200-PyTorch-PerfSanity-Post-Merge-1": ["auto:dgx-b200-flex", "l0_b200_perf_sanity", 1, 1, 1, 1, true],
        // PerfSanity post-merge tests
        "DGX_B200-8_GPUs-PyTorch-PerfSanity-Post-Merge-1": ["auto:dgx-b200-flex", "l0_b200_multi_gpus_perf_sanity", 1, 4, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-PerfSanity-Post-Merge-2": ["auto:dgx-b200-flex", "l0_b200_multi_gpus_perf_sanity", 2, 4, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-PerfSanity-Post-Merge-3": ["auto:dgx-b200-flex", "l0_b200_multi_gpus_perf_sanity", 3, 4, 8, 1, true],
        "DGX_B200-8_GPUs-PyTorch-PerfSanity-Post-Merge-4": ["auto:dgx-b200-flex", "l0_b200_multi_gpus_perf_sanity", 4, 4, 8, 1, true],
    ]
    // B200 PerfSanity pre-merge disaggregated (functional-only: perf regressions do not fail CI)
    // 2 Nodes
    x86SlurmTestConfigs += buildStageConfigs(
        "DGX_B200-16_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-FUNCTIONAL-ONLY-CTX1-NODE1-GPU4-GEN1-NODE1-GPU8",
        "auto:dgx-b200-flex",
        "l0_b200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node1_gpu8",
        1,
        16,
        2
    )
    // B200 PerfSanity post-merge disaggregated
    // 2 Nodes
    x86SlurmTestConfigs += buildStageConfigs(
        "DGX_B200-16_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE1-GPU8-Post-Merge",
        "auto:dgx-b200-flex",
        "l0_b200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node1_gpu8",
        2,
        16,
        2
    )
    x86SlurmTestConfigs = cbtsResizeSplits(x86SlurmTestConfigs)
    fullSet += x86SlurmTestConfigs.keySet()

    parallelSlurmJobs = x86SlurmTestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(X86_64_DOCKER_IMAGE, "slurm", "amd64"), { attemptTag, isFinalAttempt, retryContext = null ->
        // attemptTag comes from runKubernetesPodWithInfraRetry for the outer
        // dispatcher pod (when retry is enabled — see opts below) and is
        // threaded into runLLMTestlistOnSlurm so a future re-enable of outer
        // pod retry composes a unique postTag instead of colliding with the
        // dead pod's recorded upload. With singleAttempt:true (default for
        // SLURM stages) the outer retry is off and attemptTag is always "".
        def config = VANILLA_CONFIG
        if (key.contains("single-device")) {
            config = SINGLE_DEVICE_CONFIG
        }
        if (key.contains("llvm")) {
            config = LLVM_CONFIG
        }
        runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3], values[4] ?: 1, values[5] ?: 1, values[6] ?: false, false, "cp312", attemptTag, false, retryContext?.infraRetryMax)
    }, [singleAttempt: true, slurmDispatcher: true]]]}
    // SLURM dispatcher pods run their own inner retry loop
    // (runLLMTestlistOnSlurm with SLURM_INFRA_RETRY_MAX). Disabling the outer
    // K8s pod retry (singleAttempt:true) here caps total attempts at
    // SLURM_INFRA_RETRY_MAX+1 instead of (SLURM_INFRA_RETRY_MAX+1) *
    // (K8S_INFRA_RETRY_MAX+1). Each SLURM attempt that hits the partition
    // walltime burns ~4h, so nesting the two layers cost up to ~36h per stage
    // on consistently-timing-out tests before this cap.

    parallelJobs += parallelSlurmJobs

    // SBSA machines from the Blossom machine pool
    SBSATestConfigs = [
        "CPU-Generic-arm-1": ["cpu", "l0_cpu", 1, 1],
        "GH200-PyTorch-Post-Merge-1": ["gh200", "l0_gh200", 1, 1],
        // DGX Spark is also named as GB10 Grace Blackwell Superchip.
        "GB10-PyTorch-1": ["gb10x", "l0_gb10", 1, 1],
    ]
    SBSATestConfigs = cbtsResizeSplits(SBSATestConfigs)
    fullSet += SBSATestConfigs.keySet()

    SBSASlurmTestConfigs = [
        // [platform, testList, splitId, splits, gpuCount, nodeCount?, runWithSbatch?, useClusterDurations?]
        // useClusterDurations=true: record actual test times so each cluster builds its own
        // .test_durations_<clusterName> baseline for load-balanced sharding.
        "GB200-4_GPUs-PyTorch-1": ["auto:gb200-x4", "l0_gb200_multi_gpus", 1, 5, 4, 1, false, true],
        "GB200-4_GPUs-PyTorch-2": ["auto:gb200-x4", "l0_gb200_multi_gpus", 2, 5, 4, 1, false, true],
        "GB200-4_GPUs-PyTorch-3": ["auto:gb200-x4", "l0_gb200_multi_gpus", 3, 5, 4, 1, false, true],
        "GB200-4_GPUs-PyTorch-4": ["auto:gb200-x4", "l0_gb200_multi_gpus", 4, 5, 4, 1, false, true],
        "GB200-4_GPUs-PyTorch-5": ["auto:gb200-x4", "l0_gb200_multi_gpus", 5, 5, 4, 1, false, true],
        "GB200-4_GPUs-PyTorch-Post-Merge-1": ["auto:gb200-x4", "l0_gb200_multi_gpus", 1, 1, 4, 1, false, true],
        "GB10-PyTorch-Post-Merge-1": ["gb10x-single", "l0_gb10", 1, 1],
        "GB300-4_GPUs-PyTorch-1": ["auto:gb300-x4", "l0_gb300", 1, 1, 4, 1, true, false],
        "GB300-4_GPUs-PyTorch-Post-Merge-1": ["auto:gb300-x4", "l0_gb300_multi_gpus", 1, 3, 4, 1, true, false],
        "GB300-4_GPUs-PyTorch-Post-Merge-2": ["auto:gb300-x4", "l0_gb300_multi_gpus", 2, 3, 4, 1, true, false],
        "GB300-4_GPUs-PyTorch-Post-Merge-3": ["auto:gb300-x4", "l0_gb300_multi_gpus", 3, 3, 4, 1, true, false],
        // PerfSanity pre-merge tests
        "GB200-4_GPUs-PyTorch-PerfSanity-1": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 1, 2, 4],
        "GB200-4_GPUs-PyTorch-PerfSanity-2": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 2, 2, 4],
        // PerfSanity post-merge tests
        "GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-1": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 1, 4, 4],
        "GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-2": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 2, 4, 4],
        "GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-3": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 3, 4, 4],
        "GB200-4_GPUs-PyTorch-PerfSanity-Post-Merge-4": ["auto:gb200-x4", "l0_gb200_multi_gpus_perf_sanity", 4, 4, 4],
        "GB300-4_GPUs-PyTorch-PerfSanity-Post-Merge-1": ["auto:gb300-x4", "l0_gb300_multi_gpus_perf_sanity", 1, 5, 4, 1, true, false],
        "GB300-4_GPUs-PyTorch-PerfSanity-Post-Merge-2": ["auto:gb300-x4", "l0_gb300_multi_gpus_perf_sanity", 2, 5, 4, 1, true, false],
        "GB300-4_GPUs-PyTorch-PerfSanity-Post-Merge-3": ["auto:gb300-x4", "l0_gb300_multi_gpus_perf_sanity", 3, 5, 4, 1, true, false],
        "GB300-4_GPUs-PyTorch-PerfSanity-Post-Merge-4": ["auto:gb300-x4", "l0_gb300_multi_gpus_perf_sanity", 4, 5, 4, 1, true, false],
        "GB300-4_GPUs-PyTorch-PerfSanity-Post-Merge-5": ["auto:gb300-x4", "l0_gb300_multi_gpus_perf_sanity", 5, 5, 4, 1, true, false],
    ]
    SBSASlurmTestConfigs = cbtsResizeSplits(SBSASlurmTestConfigs)
    fullSet += SBSASlurmTestConfigs.keySet()

    multiNodesSBSAConfigs = [
        // Each GB200 testcase below uses 8 GPUs and 2 nodes.
        // https://nvbugs/5598863 (uncorrectable NVLink error detected during the execution) may not exist in OCI machines.
        "GB200-8_GPUs-2_Nodes-PyTorch-1": ["auto:gb200-flex", "l0_gb200_multi_nodes", 1, 2, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-2": ["auto:gb200-flex", "l0_gb200_multi_nodes", 2, 2, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-1": ["auto:gb200-flex", "l0_gb200_multi_nodes", 1, 3, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-2": ["auto:gb200-flex", "l0_gb200_multi_nodes", 2, 3, 8, 2],
        "GB200-8_GPUs-2_Nodes-PyTorch-Post-Merge-3": ["auto:gb200-flex", "l0_gb200_multi_nodes", 3, 3, 8, 2],
        // GB300 accuracy post-merge aggregated (4 GPUs per node). One test list per topology,
        // spelled out here rather than via buildStageConfigs: test_to_stage_mapping.py resolves
        // stage <-> test by list name with a line-based parser, so a shared list or a helper's
        // output breaks the mapping. For SingleNvlinkDomain see singleNvlinkDomainMode.
        "GB300-8_GPUs-2_Nodes-PyTorch-SingleNvlinkDomain-Post-Merge-1": ["auto:gb300-flex", "l0_gb300_multi_nodes_node2_gpu8", 1, 1, 8, 2],
        "GB300-16_GPUs-4_Nodes-PyTorch-SingleNvlinkDomain-Post-Merge-1": ["auto:gb300-flex", "l0_gb300_multi_nodes_node4_gpu16", 1, 1, 16, 4],
    ]
    // PerfSanity post-merge aggregated
    // 2 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-8_GPUs-2_Nodes-PyTorch-PerfSanity-Node2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_node2_gpu8",
        6,
        8,
        2
    )
    // PerfSanity pre-merge disaggregated (functional-only: perf regressions do not fail CI)
    // 2 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-FUNCTIONAL-ONLY-CTX1-NODE1-GPU1-GEN1-NODE1-GPU4",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node1_gpu4",
        1,
        8,
        2
    )
    // PerfSanity post-merge disaggregated
    // 2 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU1-GEN1-NODE1-GPU2-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node1_gpu2",
        1,
        8,
        2
    )
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU1-GEN1-NODE1-GPU4-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node1_gpu4",
        5,
        8,
        2
    )
    // 3 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-12_GPUs-3_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU1-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu1_gen1_node2_gpu8",
        2,
        12,
        3
    )
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-12_GPUs-3_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node2_gpu8",
        1,
        12,
        3
    )
    // 4 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-16_GPUs-4_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE2-GPU8-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node2_gpu8_gen1_node2_gpu8",
        1,
        16,
        4
    )
    // 5 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-20_GPUs-5_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE4-GPU16-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node4_gpu16",
        2,
        20,
        5
    )
    // 6 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-24_GPUs-6_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE2-GPU8-GEN1-NODE4-GPU16-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node2_gpu8_gen1_node4_gpu16",
        2,
        24,
        6
    )
    // 9 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB200-36_GPUs-9_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE8-GPU32-Post-Merge",
        "auto:gb200-flex",
        "l0_gb200_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node8_gpu32",
        1,
        36,
        9
    )
    // GB300 PerfSanity post-merge aggregated
    // 2 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-8_GPUs-2_Nodes-PyTorch-PerfSanity-Node2-GPU8-Post-Merge",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_node2_gpu8",
        2,
        8,
        2
    )
    // GB300 PerfSanity post-merge disaggregated
    // 3 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-12_GPUs-3_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node2_gpu8",
        4,
        12,
        3
    )
    // 5 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-20_GPUs-5_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE4-GPU16-Post-Merge",
        "gb300-flex-aws-cmh",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node4_gpu16",
        2,
        20,
        5
    )
    // GB300 GLM-5 disaggregated (ctx DEP2)
    // 3 Nodes (pre-merge, functional-only)
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-12_GPUs-3_Nodes-PyTorch-Disagg-PerfSanity-FUNCTIONAL-ONLY-CTX1-NODE1-GPU2-GEN1-NODE2-GPU8",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu2_gen1_node2_gpu8",
        1,
        12,
        3
    )
    // 3 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-12_GPUs-3_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU2-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu2_gen1_node2_gpu8",
        2,
        12,
        3
    )
    // 9 Nodes
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-36_GPUs-9_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU2-GEN1-NODE8-GPU32-Post-Merge",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu2_gen1_node8_gpu32",
        1,
        36,
        9
    )
    // 9 Nodes: ctx1 (1 node, 4 GPUs) + gen4 (2 nodes, 8 GPUs each) = 36 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-36_GPUs-9_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN4-NODE2-GPU8-Post-Merge",
        "gb300-flex-aws-cmh",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen4_node2_gpu8",
        2,
        36,
        9
    )
    // 10 Nodes: ctx6 (1 node, 4 GPUs each) + gen1 (4 nodes, 16 GPUs) = 40 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-40_GPUs-10_Nodes-PyTorch-Disagg-PerfSanity-CTX6-NODE1-GPU4-GEN1-NODE4-GPU16-Post-Merge",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_ctx6_node1_gpu4_gen1_node4_gpu16",
        2,
        40,
        10
    )
    // 11 Nodes: ctx3 (1 node, 4 GPUs each) + gen1 (8 nodes, 32 GPUs) = 44 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-44_GPUs-11_Nodes-PyTorch-Disagg-PerfSanity-CTX3-NODE1-GPU4-GEN1-NODE8-GPU32-Post-Merge",
        "gb300-flex-aws-cmh",
        "l0_gb300_multi_nodes_perf_sanity_ctx3_node1_gpu4_gen1_node8_gpu32",
        2,
        44,
        11
    )
    // 14 Nodes: ctx12 (1 node, 4 GPUs each) + gen1 (2 nodes, 8 GPUs) = 56 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-56_GPUs-14_Nodes-PyTorch-Disagg-PerfSanity-CTX12-NODE1-GPU4-GEN1-NODE2-GPU8-Post-Merge",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_ctx12_node1_gpu4_gen1_node2_gpu8",
        2,
        56,
        14
    )
    // Nemotron-Ultra-V3 8k64k con1: ctx1 (1 node, 4 GPUs) + gen1 tep4 (1 node, 4 GPUs) = 8 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-8_GPUs-2_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN1-NODE1-GPU4-Post-Merge",
        "auto:gb300-flex",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen1_node1_gpu4",
        2,
        8,
        2
    )
    // Nemotron-Ultra-V3 50k2k con12: ctx1 (1 node, 4 GPUs) + gen6 (6 nodes, 4 GPUs each) = 28 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-28_GPUs-7_Nodes-PyTorch-Disagg-PerfSanity-CTX1-NODE1-GPU4-GEN6-NODE1-GPU4-Post-Merge",
        "gb300-flex-aws-cmh",
        "l0_gb300_multi_nodes_perf_sanity_ctx1_node1_gpu4_gen6_node1_gpu4",
        2,
        28,
        7
    )
    // Nemotron-Ultra-V3 50k2k con178: ctx5 (5 nodes, 4 GPUs each) + gen1 dep4 (1 node, 4 GPUs) = 24 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-24_GPUs-6_Nodes-PyTorch-Disagg-PerfSanity-CTX5-NODE1-GPU4-GEN1-NODE1-GPU4-Post-Merge",
        "gb300-flex-aws-cmh",
        "l0_gb300_multi_nodes_perf_sanity_ctx5_node1_gpu4_gen1_node1_gpu4",
        2,
        24,
        6
    )
    // Nemotron-Ultra-V3 con9832 (8k64k) and con1197 (50k2k) are ctx_only-only:
    // their full 68-/72-GPU e2e+gen_only disagg topologies are intentionally not
    // created; the ctx_only ids run in the 4-GPU multi_gpus post-merge stage.
    // GB300 DeepSeek-V4-Pro-DSpark, AgentX agentic trace replay.
    // These lanes replay a ~1M-token multi-turn conversation trace for a fixed
    // wall-clock duration instead of a fixed prompt count, so they are pinned to
    // aws-cmh where the DSpark checkpoint and the trace corpus are staged.
    // 6 Nodes: ctx2 (2 nodes, 8 GPUs each) + gen1 (2 nodes, 8 GPUs) = 24 GPUs
    multiNodesSBSAConfigs += buildStageConfigs(
        "GB300-24_GPUs-6_Nodes-PyTorch-Disagg-PerfSanity-AgentX-CTX2-NODE2-GPU8-GEN1-NODE2-GPU8-Post-Merge",
        "gb300-flex-aws-cmh",
        "l0_gb300_multi_nodes_perf_sanity_ctx2_node2_gpu8_gen1_node2_gpu8",
        1,
        24,
        6
    )
    multiNodesSBSAConfigs = cbtsResizeSplits(multiNodesSBSAConfigs)
    fullSet += multiNodesSBSAConfigs.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        parallelJobs = SBSATestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(LLM_DOCKER_IMAGE, values[0], "arm64"), { attemptTag, isFinalAttempt, retryContext = null ->
            runLLMTestlistOnPlatform(pipeline, values[0], values[1], LINUX_AARCH64_CONFIG, false, key, values[2], values[3], false, "cp312", attemptTag, isFinalAttempt, retryContext, values[4] ?: false)
        }]]}

        // Add SBSA Slurm jobs
        // singleAttempt:true disables the outer K8s pod retry; see the x86
        // SLURM closure above for the full rationale (cap nested retry budget
        // so consistently-timing-out tests don't burn ~36h on retry cascades).
        parallelSlurmJobs = SBSASlurmTestConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(X86_64_DOCKER_IMAGE, "slurm", "amd64"), { attemptTag, isFinalAttempt, retryContext = null ->
            // attemptTag is threaded into runLLMTestlistOnSlurm as the outer
            // dispatcher pod's tag so the inner SLURM retry's postTag can't
            // collide with a previous dispatcher pod's upload. See the x86
            // SLURM closure for the full rationale.
            def config = LINUX_AARCH64_CONFIG
            if (key.contains("single-device")) {
                config = SINGLE_DEVICE_CONFIG
            }
            if (key.contains("llvm")) {
                config = LLVM_CONFIG
            }
            runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3], values[4] ?: 1, values[5] ?: 1, values[6] ?: false, false, "cp312", attemptTag, values[7] ?: false, retryContext?.infraRetryMax)
        }, [singleAttempt: true, slurmDispatcher: true]]]}
        parallelJobs += parallelSlurmJobs

        // Add SBSA multi node Slurm jobs
        // singleAttempt:true disables the outer K8s pod retry; see above.
        parallelMultiNodesSBSAJobs = multiNodesSBSAConfigs.collectEntries{key, values -> [key, [createKubernetesPodConfig(X86_64_DOCKER_IMAGE, "slurm", "amd64"), { attemptTag, isFinalAttempt, retryContext = null ->
            def config = LINUX_AARCH64_CONFIG
            if (key.contains("single-device")) {
                config = SINGLE_DEVICE_CONFIG
            }
            if (key.contains("llvm")) {
                config = LLVM_CONFIG
            }
            runLLMTestlistOnSlurm(pipeline, values[0], values[1], config, key.contains("-Perf-"), key, values[2], values[3], values[4] ?: 1, values[5] ?: 2, values[6] ?: false, false, "cp312", attemptTag, values[7] ?: false, retryContext?.infraRetryMax)
        }, [singleAttempt: true, slurmDispatcher: true]]]}

        parallelJobs += parallelMultiNodesSBSAJobs
    }

    // Doc build is pure CPU work (checkout + wheel install + doxygen/sphinx
    // `make html` + upload); it never touches a GPU, so run it on the CPU
    // "build" pod. Mirrors the CPU-only agent-flow job below.
    docBuildSpec = createKubernetesPodConfig(LLM_DOCKER_IMAGE, "build")
    docBuildConfigs = [
        "CPU-Build_Docs": [docBuildSpec, {
            sh "rm -rf **/*.xml *.tar.gz"
            runLLMDocBuild(pipeline, config=VANILLA_CONFIG)
        }],
    ]

    fullSet += docBuildConfigs.keySet()

    if (env.targetArch == AARCH64_TRIPLE) {
        docBuildConfigs = [:]
    }

    docBuildJobs = docBuildConfigs.collectEntries{key, values -> [key, [values[0], { attemptTag, isFinalAttempt, retryContext = null ->
        // attemptTag uniquifies the upload-once guard key and tar filename per
        // pod-launch attempt; isFinalAttempt suppresses synthetic stage-fail XML
        // and junit() on intermediate retryable infra failures.
        stage("[${key}] Run") {
            cacheErrorAndUploadResult("${key}", values[1], {}, true, attemptTag, isFinalAttempt, retryContext)
        }
    }]]}

    // agent-flow: pure-CPU pytest suite for the imported agent-flow sub-project.
    // Runs on a CPU node ("build" pod type, kubernetes-cpu cloud) — it needs no
    // GPU and no TRT-LLM wheel, only its own `pip install -e agent-flow[test]`.
    // CBTS narrows this to agent-flow-only changes via
    // jenkins/scripts/cbts/rules/agent_flow_rule.py; AGENT_FLOW_STAGE in that
    // rule MUST equal the stage name below or CBTS Layer 2 will drop it.
    agentFlowTestSpec = createKubernetesPodConfig(LLM_DOCKER_IMAGE, "build")
    // Single source of truth for the stage name: it is both the map key (which
    // cacheErrorAndUploadResult uses for the results dir it tars and feeds to
    // junit) and the name passed to runLLMAgentFlowTest (which writes
    // results.xml into that dir). If the two ever diverge, junit finds no
    // report ("No test report files were found").
    def agentFlowStageName = "CPU-AgentFlow-UnitTest"
    agentFlowTestConfigs = [
        (agentFlowStageName): [agentFlowTestSpec, {
            sh "rm -rf **/*.xml *.tar.gz"
            runLLMAgentFlowTest(pipeline, agentFlowStageName)
        }],
    ]

    fullSet += agentFlowTestConfigs.keySet()

    // agent-flow tests are architecture-independent; run them once on x86 only.
    if (env.targetArch == AARCH64_TRIPLE) {
        agentFlowTestConfigs = [:]
    }

    agentFlowTestJobs = agentFlowTestConfigs.collectEntries{key, values -> [key, [values[0], { attemptTag, isFinalAttempt, retryContext = null ->
        stage("[${key}] Run") {
            cacheErrorAndUploadResult("${key}", values[1], {}, false, attemptTag, isFinalAttempt, retryContext)
        }
    }]]}

    // Python version and OS for sanity check
    // Slots: [buildImage, gpuType, cpuArch, reinstallDependencies, isDlfw, pipInstallImage, extraPytorchInstall, platName]
    x86SanityCheckConfigs = [
        "PY312-DLFW": [
            LLM_DOCKER_IMAGE,
            "B200_PCIe",
            X86_64_TRIPLE,
            false,
            true,
            DLFW_IMAGE,
            false,
            'manylinux_2_39_x86_64',
        ],
        "PY310-UB2204": [
            LLM_ROCKYLINUX8_PY310_DOCKER_IMAGE,
            "A10",
            X86_64_TRIPLE,
            true,
            false,
            UBUNTU_22_04_IMAGE,
            true, // Extra install PyTorch CUDA 13.0 package to align with the CUDA version used for building TensorRT LLM wheels.
            'manylinux_2_28_x86_64',
        ],
        "PY312-UB2404": [
            LLM_ROCKYLINUX8_PY312_DOCKER_IMAGE,
            "A100X",
            X86_64_TRIPLE,
            true,
            false,
            UBUNTU_24_04_IMAGE,
            true, // Extra PyTorch CUDA 13.0 install
            'manylinux_2_28_x86_64',
        ],
    ]

    aarch64SanityCheckConfigs = [
        "PY312-UB2404": [
            LLM_WHEEL_DOCKER_IMAGE,
            "GH200",
            AARCH64_TRIPLE,
            false,
            false,
            UBUNTU_24_04_IMAGE,
            true, // Extra PyTorch CUDA 13.0 install
            'manylinux_2_39_aarch64',
        ],
        "PY312-DLFW": [
            LLM_DOCKER_IMAGE,
            "GH200",
            AARCH64_TRIPLE,
            false,
            true,
            DLFW_IMAGE,
            false,
            'manylinux_2_39_aarch64',
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

            def isDlfw = values[4]
            def packageVersionOverride = versionOverride
            if (isDlfw) {
                // Extract PyTorch version from LLM_DOCKER_IMAGE. e.g. pytorch-26.02 -> 2602
                def matcher = LLM_DOCKER_IMAGE =~ /:pytorch-(\d+)\.(\d+)-/
                if (!matcher.find()) {
                    error "Failed to extract PyTorch version from LLM_DOCKER_IMAGE: ${LLM_DOCKER_IMAGE}"
                }
                packageVersionOverride +=
                    "+ngcpytorch${matcher.group(1)}${matcher.group(2)}"
            }
            def wheelName = ""
            def cpver = "cp312"
            def pyver = "3.12"
            if (key.contains("PY310")) {
                cpver = "cp310"
                pyver = "3.10"
            }

            buildRunner("[${toStageName(values[1], key)}] Build") {
                wheelName = runLLMBuild(pipeline, cpu_arch, values[3], "", packageVersionOverride, cpver, values[7])
            }

            // TODO: Re-enable the sanity check after updating GPU testers' driver version.
            // def fullWheelPath = "${cpu_arch}/${wheelName}"
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
                    runKubernetesPodWithInfraRetry(pipeline, pipInstallSanitySpec, "trt-llm", toStageName(values[1], key), { attemptTag, isFinalAttempt, retryContext = null ->
                        echo "###### Prerequisites Start ######"
                        echoNodeAndGpuInfo(pipeline, toStageName(values[1], key))
                        // Clean up the pip constraint file from the base NGC PyTorch image.
                        if (values[5] == DLFW_IMAGE) {
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "[ -f /etc/pip/constraint.txt ] && : > /etc/pip/constraint.txt || true")
                            // Remove the python3-pygments pip package because the dlfw image already includes a Debian pygments package, which conflicts with the pip-installed version.
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get remove -y python3-pygments")
                            // Remove stale nvidia-cutlass-dsl from the base image to prevent namespace
                            // directory corruption when pip upgrades to the version required by tensorrt_llm.
                            trtllm_utils.llmExecStepWithRetry(
                                pipeline,
                                script: "pip3 uninstall -y nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base " +
                                    "nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 " +
                                    "nvidia-cutlass-dsl-libs-cu13 || true")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: 'rm -rf $(python3 -c "import site; print(site.getsitepackages()[0])")/nvidia_cutlass_dsl*')
                        }
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y python3-pip git rsync curl wget")
                        trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 config set global.break-system-packages true")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install 'requests>=2.32.4,<3'")
                        trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 uninstall -y tensorrt")
                        if (values[5] != DLFW_IMAGE) {
                            def ubuntu_version = key.contains("UB2404") ? "ubuntu2404" : "ubuntu2204"
                            def platform = cpu_arch == X86_64_TRIPLE ? "x86_64" : "sbsa"
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "wget https://developer.download.nvidia.com/compute/cuda/repos/${ubuntu_version}/${platform}/cuda-keyring_1.1-1_all.deb")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "dpkg -i cuda-keyring_1.1-1_all.deb")
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y cuda-toolkit-13-2")
                        }
                        // Extra PyTorch CUDA 13.2 install for all bare-metal environments (Default PyTorch is for CUDA 12.8)
                        if (values[6]) {
                            echo "###### Extra PyTorch CUDA 13.2 install Start ######"
                            // Use internal mirror instead of https://download.pytorch.org/whl/cu130 for better network stability.
                            // PyTorch CUDA 13.0 package and torchvision package can be installed as expected.
                            trtllm_utils.llmExecStepWithRetry(pipeline, script: "pip3 install torch==2.12.0+cu130 torchvision==0.27.0+cu130 --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/pytorch-cu128-remote/simple --extra-index-url https://download.pytorch.org/whl/cu130")
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
                                    checkPipInstall(pipeline, "${cpu_arch}", packageVersionOverride)
                                }
                            })
                        }
                        echo "###### Run LLMAPI tests Start ######"

                        // Resolve the real tensorrt_llm install location after pip install,
                        // and expose UCX shared libraries shipped inside the wheel
                        // (tensorrt_llm/libs/ucx/*.so and libtensorrt_llm_ucx_wrapper.so)
                        // so dlopen can find them at test runtime.
                        // Use `pip3 show` (metadata only) instead of `import tensorrt_llm`,
                        // because importing executes tensorrt_llm/__init__.py which prints a
                        // version banner to stdout and would pollute the captured path.
                        def trtllmLibsDir = sh(
                            script: "pip3 show tensorrt_llm | grep \"Location\" | awk -F\":\" '{ gsub(/ /, \"\", \$2); print \$2\"/tensorrt_llm/libs\"}'",
                            returnStdout: true,
                        ).replaceAll("\\s","")
                        libEnv += ["LD_LIBRARY_PATH+trtllm_ucx=${trtllmLibsDir}/ucx"]
                        libEnv += ["LD_LIBRARY_PATH+trtllm_libs=${trtllmLibsDir}"]

                        def config = VANILLA_CONFIG
                        if (cpu_arch == AARCH64_TRIPLE) {
                            config = LINUX_AARCH64_CONFIG
                        }
                        withEnv(libEnv) {
                            sh "env | sort"
                            runLLMTestlistOnPlatform(pipeline, gpu_type, "l0_sanity_check", config, false, toStageName(values[1], key), 1, 1, true, cpver, "-SubJob-RunTest" + attemptTag, isFinalAttempt, retryContext)
                        }
                    })
                }
            }
        }, {}, true)
    }]}

    // OnDemand stages are available through --stage-list/--extra-stage only.
    multiGpuJobs = parallelJobs.findAll{(it.key =~ /\d+_GPUs/) && !it.key.contains("Post-Merge") && !it.key.contains("-OnDemand-")}
    println multiGpuJobs.keySet()
    multiGpuJobsPostMerge = parallelJobs.findAll{(it.key =~ /\d+_GPUs/) && it.key.contains("Post-Merge")}

    parallelJobs += docBuildJobs
    parallelJobs += sanityCheckJobs
    parallelJobs += agentFlowTestJobs

    onDemandJobs = parallelJobs.findAll {it.key.contains("-OnDemand-")}
    postMergeJobs = parallelJobs.findAll {it.key.contains("Post-Merge")}

    // Start as a normal pre-merge job
    parallelJobsFiltered = parallelJobs - multiGpuJobs - postMergeJobs - onDemandJobs

    // Check if the multi GPU related file has changed or not. If changed, add multi GPU test stages.
    if (testFilter[(MULTI_GPU_FILE_CHANGED)]) {
        parallelJobsFiltered += multiGpuJobs
    }

    if (testFilter[(AUTO_TRIGGER_TAG_LIST)]) {
        echo "AUTO_TRIGGER_TAG_LIST mode is true. Auto trigger tags: ${testFilter[(AUTO_TRIGGER_TAG_LIST)].join(', ')}."
        def autoTriggerTagStages = [:]
        for (tag in testFilter[(AUTO_TRIGGER_TAG_LIST)]) {
            autoTriggerTagStages += (parallelJobs - onDemandJobs).findAll { it.key.contains(tag) }
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
            excludedBackends["PyTorch"] = ["-CPP-", "-FMHA-"]     // Only pytorch file change also need to run triton tests
            excludedBackends["Triton"] = ["-PyTorch-", "-CPP-", "-FMHA-"]
            excludedBackends["FMHA"] = ["-PyTorch-", "-CPP-", "-Triton-"]
            def group = testFilter[(ONLY_ONE_GROUP_CHANGED)]
            if (excludedBackends.containsKey(group)) {
                parallelJobsFiltered = parallelJobsFiltered.findAll { key, value ->
                    !excludedBackends[group].any { backend -> key.contains(backend) }
                }
            }
            println parallelJobsFiltered.keySet()
        }
    }

    // Keep manually triggered stages out of every automatic selection path.
    parallelJobsFiltered -= onDemandJobs

    // Check --stage-list, only run the stages in stage-list. Supports wildcard '*'.
    if (testFilter[TEST_STAGE_LIST] != null) {
        echo "Use TEST_STAGE_LIST for filtering. Stages: ${testFilter[(TEST_STAGE_LIST)]}."
        parallelJobsFiltered = parallelJobs.findAll { stageMatchesAnyPattern(it.key, testFilter[(TEST_STAGE_LIST)]) }
        println parallelJobsFiltered.keySet()
    }

    // Check --extra-stage, add the stages in extra-stage. Supports wildcard '*'.
    if (testFilter[EXTRA_STAGE_LIST] != null) {
        echo "Use EXTRA_STAGE_LIST for filtering. Stages: ${testFilter[(EXTRA_STAGE_LIST)]}."
        parallelJobsFiltered += parallelJobs.findAll { stageMatchesAnyPattern(it.key, testFilter[(EXTRA_STAGE_LIST)]) }
        println parallelJobsFiltered.keySet()
    }

    checkStageName(fullSet)

    if (testFilter[(TEST_STAGE_LIST)] != null) {
        checkStageNameSet(testFilter[(TEST_STAGE_LIST)], fullSet, TEST_STAGE_LIST)
    }
    if (testFilter[(EXTRA_STAGE_LIST)] != null) {
        checkStageNameSet(testFilter[(EXTRA_STAGE_LIST)], fullSet, EXTRA_STAGE_LIST)
    }

    parallelJobsFiltered = filterCbtsStageJobs(
        parallelJobs, parallelJobsFiltered, multiGpuJobs, testFilter)

    if (globalVars[RUN_MODE] == "nightly_release") {
        parallelJobsFiltered = sanityCheckJobs
    }

    echo "Check the passed GitLab bot testFilter parameters."
    def keysStr = parallelJobsFiltered.keySet().join(",\n")
    pipeline.echo "Now we will run stages: [\n${keysStr}\n]"

    // Per-stage execution scope for infra-scoped fail-fast (runBranchesWithInfraDefer).
    // A stage carrying opts.slurmDispatcher runs its work through a SLURM dispatcher
    // pod, so its post-retry failures can be SLURM-scoped; everything else is K8s.
    // Built here, where the config tuple's opts (3rd element) is still visible, and
    // keyed by stage name so the K8s/SLURM group subsets can look each branch up.
    stageInfraScope = [:]
    parallelJobsFiltered = parallelJobsFiltered.collectEntries { key, values ->
        def stageOpts = (values instanceof List && values.size() >= 3 && values[2] instanceof Map) ? values[2] : [:]
        stageInfraScope[key] = stageOpts.slurmDispatcher ? InfraFailure.SLURM : InfraFailure.K8S
        [key, {
        stage(key) {
            if (key in testFilter[REUSE_STAGE_LIST]) {
                stage("Skip - Reused") {
                    echo "Skip - Passed in the previous pipelines."
                }
            } else if (values instanceof List) {
                // parallelJobs entries are either [podSpec, runner] or
                // [podSpec, runner, opts] where opts is a Map passed through
                // to runKubernetesPodWithInfraRetry (e.g. singleAttempt:true
                // for SLURM dispatcher pods to disable nested pod retry).
                def opts = (values.size() >= 3 && values[2] instanceof Map) ? values[2] : [:]
                runKubernetesPodWithInfraRetry(opts, pipeline, values[0], "trt-llm", key, { attemptTag, isFinalAttempt, retryContext = null ->
                    // Carry a per-stage infra-retry override (opts.infraRetryMax) to the
                    // inner runner via retryContext -- the SLURM dispatcher's runner reads
                    // it and passes it to runLLMTestlistOnSlurm. Preserve the null default
                    // when no override is set so non-SLURM runners are unaffected.
                    def innerRetryContext = retryContext
                    if (opts.infraRetryMax != null) {
                        innerRetryContext = (retryContext ?: [:]) + [infraRetryMax: opts.infraRetryMax]
                    }
                    values[1](attemptTag, isFinalAttempt, innerRetryContext)
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
                runKubernetesPodWithInfraRetry(pipeline, imageSanitySpec, "trt-llm", values.name, { attemptTag, isFinalAttempt, retryContext = null ->
                    sh "env | sort"
                    trtllm_utils.llmExecStepWithRetry(pipeline, script: "apt-get update && apt-get install -y git rsync curl")
                    runLLMTestlistOnPlatform(pipeline, values.gpuType, "l0_sanity_check", values.config, false, values.name, 1, 1, true, null, "-SubJob-TestImage" + attemptTag, isFinalAttempt, retryContext)
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
                    globalVars = trtllm_utils.initializeCiBudget(this, globalVars, 24, 'HOURS', 'L0_Test')
                    globalVars[ACTION_INFO] = trtllm_utils.setupPipelineDescription(this, globalVars[ACTION_INFO])
                }
            }
        }
        stage("Check Test List")
        {
            when {
                expression {
                    // Only run the test list validation when necessary
                    globalVars[RUN_MODE] != "nightly_release" &&
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
                    // Default scope map so the image-sanity path (which does not
                    // build one) still has a value for runBranchesWithInfraDefer;
                    // launchTestJobs overwrites this with per-stage scopes.
                    stageInfraScope = [:]
                    try {
                        if (env.JOB_NAME ==~ /.*BuildDockerImageSanityTest.*/) {
                            parallelJobs = launchTestJobsForImagesSanityCheck(this, globalVars)
                        } else {
                            parallelJobs = launchTestJobs(this, testFilter, globalVars)
                        }

                        singleGpuJobs = parallelJobs
                        dgxJobs = [:]

                        def testPhase2StageName = env.testPhase2StageName
                        if (testPhase2StageName) {
                            def multiGpuPattern = /\d+_GPUs/
                            singleGpuJobs = parallelJobs.findAll{!(it.key =~ multiGpuPattern)}
                            dgxJobs = parallelJobs.findAll{it.key =~ multiGpuPattern}

                            // Move approval-exempt multi-GPU stages into singleGpuJobs so they
                            // run without waiting for the multi-GPU dispatch (which requires
                            // the 'ci: full pre-merge approved' label).
                            def exemptJobs = dgxJobs.findAll { stageName, stageValue ->
                                MULTI_GPU_RUN_WITH_SINGLE.any { pattern ->
                                    stageMatchesPattern(stageName, pattern)
                                }
                            }
                            if (exemptJobs) {
                                echo "[Multi-GPU split] Moving ${exemptJobs.keySet()} to single-GPU job (approval-exempt)"
                                singleGpuJobs += exemptJobs
                                dgxJobs -= exemptJobs
                            }
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
                                runBranchesWithInfraDefer(singleGpuJobs, params.enableFailFast, stageInfraScope)
                            } else if (isInfraDryRun()) {
                                error "Skip single-GPU testing. No test to run for infrastructure dry run."
                            } else {
                                echo "Skip single-GPU testing. No test to run."
                            }
                        } else if (env.JOB_NAME ==~ /.*Multi-GPU.*/) {
                            echo "Only run multi-GPU tests."
                            if (dgxJobs.size() > 0) {
                                runBranchesWithInfraDefer(dgxJobs, params.enableFailFast, stageInfraScope)
                            } else {
                                error "Skip multi-GPU testing. No test to run."
                            }
                        } else {
                            if (singleGpuJobs.size() > 0) {
                                runBranchesWithInfraDefer(singleGpuJobs, params.enableFailFast, stageInfraScope)
                            } else {
                                echo "Skip single-GPU testing. No test to run."
                            }

                            if (dgxJobs.size() > 0) {
                                stage(testPhase2StageName) {
                                    runBranchesWithInfraDefer(dgxJobs, params.enableFailFast, stageInfraScope)
                                }
                            }
                        }
                    } finally {
                        // Backstop: reclaim any SLURM job / Jenkins node left orphaned
                        // by a dispatcher-pod death whose in-catch finalize did not run
                        // or failed. Best-effort; never fails the build.
                        try {
                            sweepOrphanedSlurmResources(this)
                        } catch (Exception sweepErr) {
                            echo "[SLURM-FINALIZER] post-build sweep error: ${sweepErr}"
                        }
                    }
                }
            }
        } // Test stage
    } // stages
} // pipeline
