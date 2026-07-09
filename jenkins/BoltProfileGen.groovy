@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

// =============================================================================
// BoltProfileGen.groovy - helper job: BOLT profile generation (phase 2).
//
// Pulls the phase-1 BOLT-compatible tarball produced by Build.groovy, then fans
// out one perf-sanity run per workload: each drives the perf harness
// (jenkins/scripts/perf/local/run_disagg.sh) for an existing test id, with the
// generic POST_INSTALL_HOOK (scripts/bolt/internal/perf_instrument_hook.sh)
// swapping in BOLT-instrumented libs so the run emits .fdata. A single merge job
// (scripts/bolt/internal/slurm_merge.sh) gathers every workload's .fdata, merges
// + packages the promotable bundle.
//
// Run on-demand today (manual / bot-triggered); there is no auto-trigger.
// Promotion of the packaged bundle to the branch-keyed Artifactory path (and the
// downstream apply/consume) is deferred to the follow-on deployment PR.
// =============================================================================

import groovy.transform.Field
import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.SlurmCluster
import com.nvidia.bloom.SlurmPartition
import com.nvidia.bloom.Utils

LLM_ROOT = "llm"

// Phase-1 tarball location (passed from the parent via artifactPath, same as
// Build/L0_Test). The promoted bundle goes to a stable BRANCH-keyed path.
ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
URM_ARTIFACTORY_BASE = "https://urm.nvidia.com/artifactory"

X86_64_TRIPLE = "x86_64-linux-gnu"
AARCH64_TRIPLE = "aarch64-linux-gnu"

LLM_DOCKER_IMAGE = env.dockerImage
AGENT_IMAGE = env.dockerImage ? env.dockerImage.replace("aarch64", "x86_64") : env.dockerImage

// ---- bolt-specific params (passed by launchJob additionalParameters) --------
// targetArch        : aarch64-linux-gnu | x86_64-linux-gnu
// boltRef           : source ref/commit the phase-1 tarball was built from
// branch            : branch name for the branch-keyed promote path
// slurmPlatform     : SlurmConfig platform string (GPU + multi-node for sbsa)
// boltTarName       : phase-1 TARNAME to profile (e.g. TensorRT-LLM-GH200.tar.gz)
TARGET_ARCH   = params.targetArch   ?: env.targetArch ?: AARCH64_TRIPLE
BOLT_REF      = params.boltRef      ?: (env.artifactCommit ?: env.gitlabCommit ?: "unknown")
BRANCH        = params.branch       ?: (env.gitlabTargetBranch ?: "main")
// SBSA multi-node on oci-hsg: flexible node count (sbatch sets --nodes itself).
// auto:gb200-flex -> gb200-flex-oci-hsg -> gb200-oci-trtllm (clusterName oci-hsg).
SLURM_PLATFORM= params.slurmPlatform?: (TARGET_ARCH == AARCH64_TRIPLE ? "auto:gb200-flex" : "")
BOLT_TARNAME  = params.boltTarName  ?: (TARGET_ARCH == AARCH64_TRIPLE ? "TensorRT-LLM-GH200.tar.gz" : "TensorRT-LLM.tar.gz")
NUM_NODES     = params.numNodes     ?: "2"   // legacy single-workload wiring (unused by fan-out)

TRIPLE = TARGET_ARCH

// Fan-out workload set. Each entry is an EXISTING perf-sanity test id (the
// cluster-tuned config lives in tests/scripts/perf-sanity/{aggregated,
// disaggregated}/); the perf harness (jenkins/scripts/perf) runs it, and the
// generic POST_INSTALL_HOOK (scripts/bolt/internal/perf_instrument_hook.sh)
// swaps in BOLT-instrumented libs so the run emits .fdata. submit.py derives
// runtime mode + SLURM node count from the test id / config, so we don't size
// allocations here.
//   name   : short BOLT label -> $FDATA_ROOT/<name>/<host> and manifest workload
//   testId : pytest node id: perf/test_perf_sanity.py::test_e2e[<case>]
// NOTE: entries mirror perf-sanity test-db cases and must be kept in sync with
// them. The 3 GB200 agg entries are the validated set; disagg is the follow-on scope.
BOLT_WORKLOADS = [
    [name: "dsr1_agg_1k1k_c2",     testId: "perf/test_perf_sanity.py::test_e2e[aggr-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_tp4_mtp3_1k1k]"],
    [name: "dsr1_agg_8k1k_c2",     testId: "perf/test_perf_sanity.py::test_e2e[aggr-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_tp4_mtp3_8k1k]"],
    [name: "dsr1_agg_1k1k_c1024",  testId: "perf/test_perf_sanity.py::test_e2e[aggr-deepseek_r1_fp4_v2_grace_blackwell-r1_fp4_v2_dep4_mtp1_1k1k]"],
    // --- disagg (multi-node): deferred to the follow-on PR. Disagg gen-worker
    //     bring-up is still flaky under BOLT instrumentation (servers hang or a
    //     gen worker crashes at startup), so this PR ships only the validated
    //     single-node agg set. Re-enable per-entry once each is confirmed passing.
    // [name: "dsr1_disagg_128k8k_c1",  testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-r1-fp4_128k8k_con1_ctx1_pp8_gen1_tep8_eplb0_mtp3_ccb-NIXL]"],
    // [name: "k2_disagg_1k1k_c4",      testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL]"],
    // [name: "dsr1_disagg_1k1k_c1",    testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL]"],
    // [name: "dsr1_disagg_128k8k_c128",testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL]"],
    // [name: "dsv32_disagg_32k4k_c1",  testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-v32-fp4_32k4k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL]"],
]

POD_TIMEOUT_SECONDS = env.podTimeoutSeconds ? env.podTimeoutSeconds : "43200"

// Lightweight CPU dispatcher pod: it only SSHes to the SLURM frontend and polls
// sacct; the heavy work runs on the cluster. Mirrors the "agent" pod type in
// L0_MergeRequest.groovy::createKubernetesPodConfig (cloud, nodeSelector,
// nodeAffinity, resources, PVC), with an `alpine` work container so BSL's
// withSlurmSshCredentials can apk-install ssh as root (the devel image can't).
def createKubernetesPodConfig(image, arch = "amd64")
{
    // amd64 is hardcoded: the jnlp + alpine images are arch-specific, so the pod
    // must land on an amd64 node (the aarch64/SBSA work runs on the SLURM cluster,
    // not on this pod). The `arch` arg is intentionally ignored.
    def jnlpImage = "urm.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_amd_linux:jdk17"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/arch: amd64
                  kubernetes.io/os: linux"""
    def containerConfig = """
                  - name: alpine
                    image: urm.nvidia.com/docker/alpine:latest
                    command: ['cat']
                    tty: true
                    resources:
                      requests:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                    imagePullPolicy: Always"""
    def nodeLabel = trtllm_utils.generateNodeLabel("cpu")
    def podConfig = [
        cloud: "kubernetes-cpu",
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
                nodeSelector: ${selectors}
                containers:
                  ${containerConfig}
                    env:
                    - name: HOST_NODE_NAME
                      valueFrom:
                        fieldRef:
                          fieldPath: spec.nodeName
                  - name: jnlp
                    image: ${jnlpImage}
                    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
                    resources:
                      requests:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                volumes:
                - name: sw-tensorrt-pvc
                  persistentVolumeClaim:
                    claimName: sw-tensorrt-pvc
        """.stripIndent(),
    ]
    return podConfig
}


// ---------------------------------------------------------------------------
// SLURM profile generation: fan out perf-harness runs (one per workload, BOLT
// hook enabled) then a single cross-workload merge (slurm_merge.sh).
// ---------------------------------------------------------------------------
def submitProfileGen(pipeline)
{
    // The phase-1 tarball to profile:
    def llmTarfile = "${URM_ARTIFACTORY_BASE}/${ARTIFACT_PATH}/${BOLT_TARNAME}"
    pipeline.echo("Phase-1 tarball: ${llmTarfile}")

    SlurmPartition partition = SlurmConfig.resolvePlatform(SLURM_PLATFORM)
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]
    def scratch = cluster.scratchPath ?: "/lustre/fs1/portfolios/coreai/projects/coreai_tensorrt_ci"
    // Ephemeral workspaces live under the service-user dir on the lustre root
    // (NOT the project root directly, and NOT $HOME which is size-limited).
    def ws = "${scratch}/users/svc_tensorrt/bolt-ci/${BRANCH}/${TRIPLE}/${env.BUILD_TAG}"
    // Run-level shared fdata root: each workload writes $fdataRoot/<workload>/<host>,
    // the single merge job globs across all of them and packages into _bundle.
    def fdataRoot = "${ws}/runs/profile-fdata"
    def outDir = "${fdataRoot}/_bundle"
    def bundle = "${outDir}/bolt-profile-${BOLT_REF}-${TRIPLE}.tar.gz"

    // Bootstrap on the frontend entirely from the phase-1 tarball (no agent->
    // cluster file copy): stage the toolkit (scripts/bolt), the full source tree
    // + wheel (the perf harness runs from the checkout, install_mode=wheel), and
    // llvm-bolt once (shared, reused by the per-node instrument hook + merge).
    // Then fan out one perf-harness run per workload, and merge once.
    CloudManager.withSlurmSshCredentials(pipeline, partition.clusterName, cluster) { remote ->
        // 1) Bootstrap on the frontend as SEPARATE commands with NO timeout: the
        //    tarball download is a multi-GB cross-region transfer, and Utils.exec
        //    defaults to a 10-min timeout, so pass timeout:false on each.
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, "\"mkdir -p ${ws}/builds ${ws}/runs ${ws}/toolkit\""))

        // Download the phase-1 tarball from Artifactory to the cluster frontend.
        // curl (not wget): --speed-time/--speed-limit aborts a STALLED transfer
        // (< ~10KB/s for 120s) and --retry restarts it, so a flaky cross-region
        // link can't hang the job (Utils.exec runs with timeout:false). NO -C -:
        // a resumed/appended transfer is a corruption vector; a clean restart on
        // retry is safer. gzip -t verifies integrity so a truncated/corrupt
        // download fails fast rather than poisoning the extract step.
        def tarStage = """
            set -e
            curl -fSL --retry 10 --retry-all-errors --retry-delay 15 \
                 --connect-timeout 60 --speed-time 120 --speed-limit 10000 \
                 -o ${ws}/builds/${BOLT_TARNAME} ${llmTarfile}
            if ! gzip -t ${ws}/builds/${BOLT_TARNAME}; then
                echo '[ERROR] downloaded tarball failed gzip -t (corrupt/truncated)'
                rm -f ${ws}/builds/${BOLT_TARNAME}
                exit 1
            fi
        """.stripIndent()
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, "\"${tarStage}\""))

        // Extract the full source tree + wheel from the tarball (which packs the
        // build commit's TensorRT-LLM/src). The perf harness runs from this
        // checkout (jenkins/scripts/perf + tests/scripts/perf-sanity), and
        // install_mode=wheel uses the bundled TensorRT-LLM/tensorrt_llm-*.whl;
        // scripts/bolt (the toolkit) lives under src/ too.
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, "\"tar -xf ${ws}/builds/${BOLT_TARNAME} -C ${ws} TensorRT-LLM\""))
        // Keep a toolkit copy for the merge job (slurm_merge.sh TOOLKIT_HOST).
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, "\"cp -r ${ws}/TensorRT-LLM/src/scripts/bolt/. ${ws}/toolkit/\""))

        // Stage llvm-bolt ONCE here (shared ${ws}/builds/llvm), before the fan-out.
        // The per-node instrument hook (perf_instrument_hook.sh, via BOLT_LLVM_DIR)
        // and the merge job both reuse it, so no worker re-downloads llvm and
        // parallel runs can't race extracting into the same dir.
        def llvmArch = (TARGET_ARCH == AARCH64_TRIPLE) ? "ARM64" : "X64"
        def llvmVer  = "21.1.5"   // keep in sync with internal/slurm_merge.sh LLVM_BOLT_VERSION
        def llvmTb   = "LLVM-${llvmVer}-Linux-${llvmArch}.tar.xz"
        def llvmStage = """
            set -e
            if [ ! -x ${ws}/builds/llvm/bin/llvm-bolt ]; then
                echo '[INFO] staging llvm-bolt ${llvmVer} once (shared by all workloads)'
                mkdir -p ${ws}/builds/llvm
                curl -fSL --retry 10 --retry-all-errors --retry-delay 15 --connect-timeout 60 \
                     -o /tmp/${llvmTb} https://github.com/llvm/llvm-project/releases/download/llvmorg-${llvmVer}/${llvmTb}
                tar -xJf /tmp/${llvmTb} -C ${ws}/builds/llvm --strip-components=1
                rm -f /tmp/${llvmTb}
            else
                echo '[INFO] llvm-bolt already staged'
            fi
        """.stripIndent()
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, "\"${llvmStage}\""))

        // 2) Fan-out: ONE perf-sanity run per workload (Jenkins parallel{}).
        //    Each drives the perf harness (run_disagg.sh) for its test id, with
        //    the BOLT POST_INSTALL_HOOK swapping in instrumented libs so the run
        //    emits .fdata under $FDATA_ROOT/<workload>/<host>. submit.py sizes the
        //    SLURM allocation from the config, so we don't pass node counts.
        def modelsRoot = env.boltModelsRoot ?: '/lustre/fs1/portfolios/coreai/projects/coreai_tensorrt_ci/llm-models'
        def trtllmSrc = "${ws}/TensorRT-LLM/src"
        def imageEnroot = (LLM_DOCKER_IMAGE ?: "").replace("urm.nvidia.com/", "urm.nvidia.com#")
        // Cluster values for the harness .conf, AUTO-DERIVED from the same resolved
        // SLURM partition the merge job uses -- so a plain `/bot run` works with
        // nothing to set by hand. `partition.name` is the SLURM partition name
        // (getPartitionArgs builds `--partition=<name>` from it); mounts default to
        // the workspace + models (covers hook, llvm, fdata, toolkit -- all under ws).
        // Overridable via param/env if a run ever needs a different partition.
        def harnessPartition = params.boltHarnessPartition ?: env.boltHarnessPartition ?: partition.name
        def harnessMounts = params.boltHarnessMounts ?: env.boltHarnessMounts ?: "${ws}:${ws},${modelsRoot}:${modelsRoot}"
        // The merge job still uses our own slurm_merge.sh (not the harness).
        def partArgs = "${partition.additionalArgs} ${SlurmConfig.getTimeArgs(partition)} ${SlurmConfig.getPartitionArgs(partition)}"

        // Wrap each branch in a stage() so Blue Ocean renders one parallel stage
        // per workload (named "Collect: <workload>").
        def branches = [:]
        BOLT_WORKLOADS.each { wl ->
            branches["Collect: ${wl.name}"] = {
                stage("Collect: ${wl.name}") {
                    def jid = submitHarnessWorkload(pipeline, remote, ws, fdataRoot, trtllmSrc,
                                                    modelsRoot, imageEnroot, harnessPartition, harnessMounts, wl)
                    pipeline.echo("workload ${wl.name}: submitted perf-harness job ${jid}")
                    pollSlurm(pipeline, remote, jid, "collect:${wl.name}")
                }
            }
        }
        parallel(branches)
        pipeline.echo("All ${BOLT_WORKLOADS.size()} collect job(s) COMPLETED; starting cross-workload merge.")

        // 3) Single merge job: gather every workload's .fdata -> merge -> package.
        //    Wrapped in its own stage() so it shows as a distinct marker in
        //    Blue Ocean after the parallel collect fan-out.
        stage("Merge + Package") {
            def mid = submitMerge(pipeline, remote, ws, fdataRoot, outDir, partArgs)
            pipeline.echo("submitted merge job ${mid}")
            pollSlurm(pipeline, remote, mid, "merge")
            pipeline.echo("Merge COMPLETED. Bundle: ${bundle}")
        }
        // Promote of the packaged bundle to Artifactory is deferred to the
        // follow-on deployment PR.
    }
    return bundle
}

// ---------------------------------------------------------------------------
// Fan-out helpers. Top-level methods (not closures) so they are CPS-safe to call
// from parallel{} branches. Each submits via the BSL SSH primitives and returns
// the parsed SLURM job id; pollSlurm blocks until the job reaches a terminal
// state and FAILS on anything other than COMPLETED|0:0 (no fallback by design).
// ---------------------------------------------------------------------------
// Drive the perf-sanity harness (run_disagg.sh) for one workload's test id, with
// the BOLT POST_INSTALL_HOOK enabled so the run's ctx/gen worker(s) load
// instrumented libs and emit .fdata to $FDATA_ROOT/<name>/<host>. Writes a
// per-workload .conf, runs the harness (which sbatches), and returns the SLURM
// job id it recorded. submit.py derives runtime mode + node count from the config.
def submitHarnessWorkload(pipeline, remote, String ws, String fdataRoot, String trtllmSrc,
                          String modelsRoot, String imageEnroot, String harnessPartition,
                          String harnessMounts, Map wl)
{
    def workDir = "${ws}/harness/${wl.name}"
    def conf = "${workDir}/bolt.conf"
    def hook = "${trtllmSrc}/scripts/bolt/internal/perf_instrument_hook.sh"
    def runDisagg = "${trtllmSrc}/jenkins/scripts/perf/local/run_disagg.sh"
    // SLURM wall-time for each collect job. Default 4h (this cluster's max): the
    // instrumented workload runs much slower than an uninstrumented one, so the
    // old run_disagg.sh default of 2h is too tight for the heavier cases.
    // Overridable via param/env if a cluster allows more/less.
    def harnessTimeLimit = params.boltHarnessTimeLimit ?: env.boltHarnessTimeLimit ?: '04:00:00'

    // Resolve the wheel path up front (separate ssh) so we bake a literal path
    // into the conf -- avoids a $(...) in the heredoc that the agent shell would
    // otherwise evaluate before the command reaches the cluster.
    def wheel = Utils.exec(pipeline, timeout: false, returnStdout: true, numRetries: 1,
        script: Utils.sshUserCmd(remote,
            "\"ls ${ws}/TensorRT-LLM/tensorrt_llm-*.whl | head -1\"")).trim()

    // Generate the .conf, then run the harness with the BOLT hook wired in.
    // Single-quote EXTRA_CONTAINER_EXPORTS (its value has no single quotes) so
    // the nested value doesn't collide with sshUserCmd's outer double quotes.
    def script = """
        set -e
        mkdir -p ${workDir}
        cat > ${conf} <<'CONF'
trtllm=${trtllmSrc}
work_dir=${workDir}
partition=${harnessPartition}
account=coreai_tensorrt_ci
image=${imageEnroot}
mounts=${harnessMounts}
llm_models_path=${modelsRoot}
install_mode=wheel
wheel_path=${wheel}
test_id=${wl.testId}
time_limit=${harnessTimeLimit}
CONF
        export POST_INSTALL_HOOK=${hook}
        # The hook instruments the installed TRT-LLM libs at install time; the
        # workload then runs once and emits .fdata under \$BOLT_FDATA_DIR/<host>.
        export EXTRA_CONTAINER_EXPORTS='BOLT_FDATA_DIR=${fdataRoot}/${wl.name};BOLT_LLVM_DIR=${ws}/builds/llvm'
        bash ${runDisagg} -c ${conf}
        # run_disagg records '<jobid>|<test_id>' lines; emit the (single) job id.
        cut -d'|' -f1 ${workDir}/slurm_jobs.txt | head -1
    """.stripIndent()
    return Utils.exec(pipeline, timeout: false, returnStdout: true, numRetries: 1,
                      script: Utils.sshUserCmd(remote, "\"${script}\"")).trim().readLines().last().trim()
}

def submitMerge(pipeline, remote, String ws, String fdataRoot, String outDir, String partArgs)
{
    def cmd = "cd ${ws}/toolkit && " +
        "CONTAINER_IMAGE=${LLM_DOCKER_IMAGE} " +
        "WORKSPACE=${ws} TOOLKIT_HOST=${ws}/toolkit BUILDS_HOST=${ws}/builds " +
        "BOLT_REF=${BOLT_REF} TRIPLE=${TRIPLE} TARBALL_NAME=${BOLT_TARNAME} " +
        "FDATA_ROOT=${fdataRoot} OUT_DIR=${outDir} " +
        "sbatch --parsable --nodes=1 ${partArgs} internal/slurm_merge.sh"
    return Utils.exec(pipeline, timeout: false, returnStdout: true, numRetries: 1,
                      script: Utils.sshUserCmd(remote, "\"${cmd}\"")).trim().tokenize(';')[0].trim()
}

def pollSlurm(pipeline, remote, String jobId, String label)
{
    // No timeout: SLURM enforces the job time limit; the outer stage timeout is
    // the backstop. FAIL on any non-COMPLETED terminal state (no fallback).
    waitUntil(initialRecurrencePeriod: 60000) {
        def st = Utils.exec(pipeline, returnStdout: true, numRetries: 3, timeout: false,
            script: Utils.sshUserCmd(remote,
                "\"sacct -j ${jobId} --format=State,ExitCode -Pn --allocations | head -1\"")).trim()
        pipeline.echo("[${label}] job ${jobId}: ${st}")
        if (st ==~ /(?i).*(RUNNING|PENDING|CONFIGURING).*/ || st.isEmpty()) { return false }
        if (!(st ==~ /(?i)COMPLETED\|0:0.*/)) {
            error("BoltProfileGen: SLURM job ${jobId} (${label}) did not complete cleanly: ${st}")
        }
        return true
    }
}


pipeline {
    agent {
        // Lightweight x86 CPU dispatcher pod; the heavy work runs on SLURM nodes.
        kubernetes createKubernetesPodConfig(AGENT_IMAGE, "amd64")
    }
    options {
        skipDefaultCheckout()
        timestamps()
        timeout(time: 8, unit: 'HOURS')
    }
    stages {
        stage("BOLT Profile Generation") {
            steps {
                script {
                    container('alpine') {
                        def bundle = submitProfileGen(this)
                        echo("BOLT profile bundle: ${bundle}")
                    }
                }
            }
        }
    }
}
