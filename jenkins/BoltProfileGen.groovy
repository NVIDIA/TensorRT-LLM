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

// =============================================================================
// BoltProfileGen.groovy - helper job: BOLT profile generation.
//
// Pulls the BOLT-compatible tarball produced by Build.groovy, then fans
// out one perf-sanity run per workload: each drives the perf harness
// (jenkins/scripts/perf/local/run_disagg.sh) for an existing test id, with the
// generic POST_INSTALL_HOOK (scripts/bolt/internal/perf_instrument_hook.sh)
// swapping in BOLT-instrumented libs so the run emits .fdata. A single merge job
// (scripts/bolt/internal/slurm_merge.sh) gathers every workload's .fdata, merges
// + packages the promotable bundle.
//
// Launched from the postmerge pipeline's SBSA branch with promote=true, so it
// runs on that pipeline's cadence; also runnable on demand with explicit params.
// APPLY_PROFILES (default on) re-BOLTs that tarball in the same run
// ("consume immediately after generating"); PROMOTE (set by the postmerge launch,
// opt-in elsewhere) publishes the packaged bundle to the branch-keyed Artifactory
// path so premerge can pull `latest` (apply_latest.sh).
// =============================================================================

import groovy.transform.Field
import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.SlurmCluster
import com.nvidia.bloom.SlurmPartition
import com.nvidia.bloom.Utils

LLM_ROOT = "llm"

// Input tarball location (passed from the parent via artifactPath, same as
// Build/L0_Test). The promoted bundle goes to a stable BRANCH-keyed path.
ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
URM_ARTIFACTORY_BASE = "https://urm.nvidia.com/artifactory"

X86_64_TRIPLE = "x86_64-linux-gnu"
AARCH64_TRIPLE = "aarch64-linux-gnu"

LLM_DOCKER_IMAGE = env.dockerImage
AGENT_IMAGE = env.dockerImage ? env.dockerImage.replace("aarch64", "x86_64") : env.dockerImage

// ---- bolt-specific params ---------------------------------------------------
// Declared in the parameters{} block below and/or passed by launchJob; each
// resolves as `params.X ?: env.X ?: default`, so empty keeps the default here.
// targetArch        : aarch64-linux-gnu | x86_64-linux-gnu
// boltRef           : source ref/commit the tarball was built from
// branch            : branch name for the branch-keyed promote path
// slurmPlatform     : SlurmConfig platform string (GPU + multi-node for sbsa)
// boltTarName       : TARNAME to profile (e.g. TensorRT-LLM-GH200.tar.gz)
// promote           : "true" to publish the bundle to the branch `latest` path
// applyProfiles     : "true" to re-BOLT the tarball in the same run
TARGET_ARCH   = params.targetArch   ?: env.targetArch ?: AARCH64_TRIPLE
BOLT_REF      = params.boltRef      ?: (env.artifactCommit ?: env.gitlabCommit ?: "unknown")
BRANCH        = params.branch       ?: (env.gitlabTargetBranch ?: "main")
// SBSA multi-node: flexible node count (sbatch sets --nodes itself).
// gb300-flex-aws-cmh -> gb300-aws-trtllm-cmh (clusterName aws-cmh). We default to
// aws-cmh (GB300): in cross-cluster comparison it had the best disagg completion
// rate (7/8), vs GB200/aws-dfw and oci-aga (whose disagg all failed the
// cache-transceiver network precheck -- a cluster infra issue, not ours). The
// intermittent GEN IPC-spawn hang is hardware-independent (seen on both GB200 and
// GB300), so this is about picking the healthiest cluster, not fixing the hang.
// Override via params.slurmPlatform for a different cluster.
SLURM_PLATFORM= params.slurmPlatform?: (TARGET_ARCH == AARCH64_TRIPLE ? "gb300-flex-aws-cmh" : "")
BOLT_TARNAME  = params.boltTarName  ?: (TARGET_ARCH == AARCH64_TRIPLE ? "TensorRT-LLM-GH200.tar.gz" : "TensorRT-LLM.tar.gz")
NUM_NODES     = params.numNodes     ?: "2"   // legacy single-workload wiring (unused by fan-out)
// promote: publish the packaged bundle to the branch-keyed Artifactory path
// (PROFILE_PROMOTE_DIR) as both a versioned copy and latest.tar.gz. Default OFF
// (opt-in): the postmerge trigger sets promote=true; the premerge "generate +
// consume without promoting" override leaves it false. See promoteBundle().
PROMOTE       = (params.promote ?: "false").toString()
// applyProfiles: after merge, re-BOLT the input tarball with the just-generated
// bundle -> bolted tarball on the cluster ("consume immediately after generating").
// The merge job (slurm_merge.sh) runs apply_bolt.py when BOLT_APPLY=1.
APPLY_PROFILES = (params.applyProfiles ?: "true").toString()
// Multiply each workload's client `iterations` (num_requests = concurrency *
// iterations) to lengthen the measured serving window without editing the shared
// perf-sanity configs. Default 64: the point where the aggregated workloads'
// steady-state function set converges. Disagg wants its own value, so this should
// split into per-mode knobs once those workloads are re-enabled. Consumed by
// tests/integration/defs/perf/test_perf_sanity.py via EXTRA_CONTAINER_EXPORTS.
BOLT_ITER_MULT = (params.boltIterMult ?: env.boltIterMult ?: "64").toString()

TRIPLE = TARGET_ARCH

// Branch-keyed Artifactory dir the packaged bundle is promoted to, and that
// premerge consumption (apply_latest.sh -> artifactory.sh pull-latest) reads
// back. MUST stay identical to promote_dir() in scripts/bolt/internal/
// artifactory.sh -- "<REPO>/<PREFIX>/<branch>/<triple>" with the same defaults
// (BOLT_ARTIFACTORY_REPO, BOLT_PROFILE_PREFIX) -- or promote writes somewhere
// pull-latest never looks. Combined with URM_ARTIFACTORY_BASE above, this
// reproduces the URL cmd_pull_latest builds.
PROFILE_PROMOTE_DIR = "sw-tensorrt-generic/llm-artifacts/bolt-profiles/${BRANCH}/${TRIPLE}"

// SLURM states in which the job is still alive, i.e. no terminal verdict yet.
// COMPLETING (CG) is the one that bites: the tasks have exited but slurmctld is
// still running the epilog and reclaiming nodes, which on a large allocation
// outlasts the poll interval below. ExitCode is not final in that window either
// (accounting lags the state transition), so waiting for a genuinely terminal
// state is what makes the 0:0 check meaningful.
// Keep in sync with SLURM_NON_TERMINAL_STATES in jenkins/L0_Test.groovy.
SLURM_NON_TERMINAL_STATES = [
    "RUNNING", "PENDING", "CONFIGURING", "COMPLETING",
    "REQUEUED", "RESIZING", "SUSPENDED", "SIGNALING", "STOPPED",
]

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
    // --- disagg (multi-node): OUT OF SCOPE for this PR, left commented out.
    //     Disagg gen-worker bring-up under BOLT instrumentation is still flaky
    //     (servers hang / a gen worker crashes at startup), and the fan-out is
    //     all-or-nothing today: pollSlurm hard-fails on any non-COMPLETED job
    //     and Merge+Package runs only after every parallel branch succeeds, so
    //     one occurrence of that hang throws away the GPU-hours of every
    //     workload that DID finish and produces no bundle. Re-enable together
    //     with per-workload best-effort collection (merge over whatever .fdata
    //     landed, failing only if the validated agg set is missing).
    // [name: "dsr1_disagg_128k8k_c1",  testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-r1-fp4_128k8k_con1_ctx1_pp8_gen1_tep8_eplb0_mtp3_ccb-NIXL]"],
    // [name: "k2_disagg_1k1k_c4",      testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_kimi-k25-thinking-fp4_1k1k_con4_ctx1_dep4_gen1_tep4_eplb0_mtp0_ccb-NIXL]"],
    // [name: "dsr1_disagg_1k1k_c1",    testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-r1-fp4_1k1k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL]"],
    // [name: "dsr1_disagg_128k8k_c128",testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-r1-fp4_128k8k_con128_ctx1_pp8_gen1_dep16_eplb0_mtp1_ccb-NIXL]"],
    // [name: "dsv32_disagg_32k4k_c1",  testId: "perf/test_perf_sanity.py::test_e2e[disagg-e2e-gb200_deepseek-v32-fp4_32k4k_con1_ctx1_dep4_gen1_tep8_eplb0_mtp3_ccb-NIXL]"],
]

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


// Run a bash script on a remote login node whose shell may be csh/tcsh.
// Utils.bashWrappedRemoteCmd escapes with POSIX `'\''`, which the REMOTE login
// shell must parse -- fine on bash logins (oci-hsg) but csh/tcsh (aws-dfw)
// chokes ("Unmatched '", "set: Variable name must begin with a letter", "fi:
// Command not found"). Base64 the script instead: the payload is [A-Za-z0-9+/=]
// only, so it survives ANY login shell verbatim, and `base64 -d | bash` runs it
// under bash. Returns a value shaped for sshUserCmd's remotecmd arg (double
// quoted, like the plain mkdir/tar calls).
def b64BashRemoteCmd(String script)
{
    String b64 = script.bytes.encodeBase64().toString()
    return "\"echo ${b64} | base64 -d | bash\""
}

// As above, but for scripts that read their OWN stdin. `base64 -d | bash` makes
// bash read the script FROM stdin, so any `cat`/`read` inside it would swallow
// the rest of its own source instead of the caller's pipe. Decode to a file and
// run that, leaving stdin connected to whatever ssh forwarded. scriptPath is
// Groovy-interpolated (never a shell variable) so nothing here needs `$`, which
// would otherwise be eaten by the local shell building the ssh command.
def b64BashRemoteCmdStdin(String script, String scriptPath)
{
    String b64 = script.bytes.encodeBase64().toString()
    return "\"echo ${b64} | base64 -d > ${scriptPath}; trap 'rm -f ${scriptPath}' EXIT; bash ${scriptPath}\""
}

// ---------------------------------------------------------------------------
// SLURM profile generation: fan out perf-harness runs (one per workload, BOLT
// hook enabled) then a single cross-workload merge (slurm_merge.sh).
// ---------------------------------------------------------------------------
def submitProfileGen(pipeline)
{
    // The tarball to profile:
    def llmTarfile = "${URM_ARTIFACTORY_BASE}/${ARTIFACT_PATH}/${BOLT_TARNAME}"
    pipeline.echo("Input tarball: ${llmTarfile}")

    SlurmPartition partition = SlurmConfig.resolvePlatform(SLURM_PLATFORM)
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]
    def scratch = cluster.scratchPath ?: "/lustre/fs1/portfolios/coreai/projects/coreai_tensorrt_ci"
    // Ephemeral workspaces live under the service-user dir on the lustre root
    // (NOT the project root directly, and NOT $HOME which is size-limited).
    // BUILD_TAG alone isn't unique across Jenkins instances, so key the path on
    // the instance name too (jobs on different instances can share BUILD_TAGs).
    def instanceName = (env.JENKINS_URL ?: Jenkins.instance.rootUrl).replaceAll(/\/+$/, '').tokenize('/').last()
    def ws = "${scratch}/users/svc_tensorrt/bolt-ci/${BRANCH}/${TRIPLE}/${instanceName}/${env.BUILD_TAG}"
    // Run-level shared fdata root: each workload writes $fdataRoot/<workload>/<host>,
    // the single merge job globs across all of them and packages into _bundle.
    def fdataRoot = "${ws}/runs/profile-fdata"
    def outDir = "${fdataRoot}/_bundle"
    def bundle = "${outDir}/bolt-profile-${BOLT_REF}-${TRIPLE}.tar.gz"

    // Bootstrap on the frontend entirely from that tarball (no agent->
    // cluster file copy): extract the full source tree + wheel (the perf harness
    // runs from the checkout, install_mode=wheel), and stage llvm-bolt once
    // (shared, reused by the per-node instrument hook + merge). The merge job
    // mounts scripts/bolt directly from the extracted tree (no separate copy).
    // Then fan out one perf-harness run per workload, and merge once.
    CloudManager.withSlurmSshCredentials(pipeline, partition.clusterName, cluster) { remote ->
        // 1) Bootstrap on the frontend as SEPARATE commands with NO timeout: the
        //    tarball download is a multi-GB cross-region transfer, and Utils.exec
        //    defaults to a 10-min timeout, so pass timeout:false on each.
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, "\"mkdir -p ${ws}/builds ${ws}/runs\""))

        // Download the tarball from Artifactory to the cluster frontend.
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
        // bashWrappedRemoteCmd: not all clusters default to bash, so wrap the
        // multi-line script instead of relying on the login shell.
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, b64BashRemoteCmd(tarStage)))

        // Extract the full source tree + wheel from the tarball (which packs the
        // build commit's TensorRT-LLM/src). The perf harness runs from this
        // checkout (jenkins/scripts/perf + tests/scripts/perf-sanity), and
        // install_mode=wheel uses the bundled TensorRT-LLM/tensorrt_llm-*.whl;
        // scripts/bolt (TOOLKIT_HOST for the merge job) lives under src/ too.
        Utils.exec(pipeline, timeout: false, numRetries: 2,
            script: Utils.sshUserCmd(remote, "\"tar -xf ${ws}/builds/${BOLT_TARNAME} -C ${ws} TensorRT-LLM\""))

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
            script: Utils.sshUserCmd(remote, b64BashRemoteCmd(llvmStage)))

        // 2) Fan-out: ONE perf-sanity run per workload (Jenkins parallel{}).
        //    Each drives the perf harness (run_disagg.sh) for its test id, with
        //    the BOLT POST_INSTALL_HOOK swapping in instrumented libs so the run
        //    emits .fdata under $FDATA_ROOT/<workload>/<host>. submit.py sizes the
        //    SLURM allocation from the config, so we don't pass node counts.
        def modelsRoot = env.boltModelsRoot ?: "${scratch}/llm-models"
        def trtllmSrc = "${ws}/TensorRT-LLM/src"
        def imageEnroot = (LLM_DOCKER_IMAGE ?: "").replace("urm.nvidia.com/", "urm.nvidia.com#")
        // Cluster values for the harness .conf, AUTO-DERIVED from the same resolved
        // SLURM partition the merge job uses -- so a plain `/bot run` works with
        // nothing to set by hand. `partition.name` is the SLURM partition name
        // (getPartitionArgs builds `--partition=<name>` from it); mounts default to
        // the workspace + models (covers hook, llvm, fdata -- all under ws).
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
        // Promote the packaged bundle. Opt-in (PROMOTE defaults false), so a
        // plain generate-and-consume run publishes nothing. Runs BEFORE the
        // retention delete below: promote first, then reclaim scratch.
        if (PROMOTE == "true") {
            stage("Promote") {
                promoteBundle(pipeline, remote, bundle)
            }
        } else {
            pipeline.echo("PROMOTE=false: skipping Artifactory promote of ${bundle}")
        }

        // Retention: best-effort purge of workspaces older than 7 days so scratch
        // doesn't grow unbounded across runs. Depth 4 under the bolt-ci root maps
        // to <BRANCH>/<TRIPLE>/<instance>/<BUILD_TAG>, i.e. one per-run workspace.
        def retentionRoot = "${scratch}/users/svc_tensorrt/bolt-ci"
        Utils.exec(pipeline, timeout: false, numRetries: 1,
            script: Utils.sshUserCmd(remote,
                b64BashRemoteCmd("find ${retentionRoot} -mindepth 4 -maxdepth 4 -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true")))
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
    // Values are Groovy-interpolated; bashWrappedRemoteCmd avoids depending on
    // the cluster login shell being bash (heredoc + multi-line script).
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
        # TLLM_BOLT_CLEAR_COUNTERS=1 makes the workload zero BOLT counters
        # after warmup (see tensorrt_llm/_torch/bolt_profiling.py) so the profile
        # excludes startup/JIT. Forwarded as an in-container export -> reaches the
        # ranks for single-node agg workloads; multi-node also needs a
        # --container-env passthrough (follow-up).
        # TRITON_CACHE_DIR / CUDA_CACHE_PATH: aws-dfw containers ship a dangling
        # ~/.triton symlink (oci-hsg pre-provisioned the node-local target; aws-dfw
        # does not), so Triton's os.makedirs('/root/.triton/cache') raises
        # FileNotFoundError and the worker dies during warmup. Point the JIT caches
        # at node-local /tmp (always exists, not a broken symlink) so makedirs
        # succeeds. Cold per job -> JIT still happens in warmup (good for BOLT).
        export EXTRA_CONTAINER_EXPORTS='BOLT_FDATA_DIR=${fdataRoot}/${wl.name};BOLT_LLVM_DIR=${ws}/builds/llvm;TLLM_BOLT_CLEAR_COUNTERS=1;TLLM_BOLT_CLEAR_STRICT=1;BOLT_CLEAR_OFFSETS_FILE=/tmp/bolt_clear_offsets.txt;TRITON_CACHE_DIR=/tmp/bolt-triton-cache;CUDA_CACHE_PATH=/tmp/bolt-cuda-cache;DG_JIT_CACHE_DIR=/tmp/bolt-dg-cache;BOLT_ITER_MULT=${BOLT_ITER_MULT}'
        bash ${runDisagg} -c ${conf}
        # run_disagg records '<jobid>|<test_id>' lines; emit the (single) job id.
        cut -d'|' -f1 ${workDir}/slurm_jobs.txt | head -1
    """.stripIndent()
    return Utils.exec(pipeline, timeout: false, returnStdout: true, numRetries: 1,
                      script: Utils.sshUserCmd(remote, b64BashRemoteCmd(script))).trim().readLines().last().trim()
}

def submitMerge(pipeline, remote, String ws, String fdataRoot, String outDir, String partArgs)
{
    // Match the enroot URI form used for the collect jobs (imageEnroot): pyxis
    // expects urm.nvidia.com#<path>, not urm.nvidia.com/<path>.
    def mergeImage = (LLM_DOCKER_IMAGE ?: "").replace("urm.nvidia.com/", "urm.nvidia.com#")
    // Point TOOLKIT_HOST at the extracted tree (no separate ${ws}/toolkit copy).
    def toolkitHost = "${ws}/TensorRT-LLM/src/scripts/bolt"
    def cmd = "cd ${toolkitHost} && " +
        "CONTAINER_IMAGE=${mergeImage} " +
        "WORKSPACE=${ws} TOOLKIT_HOST=${toolkitHost} BUILDS_HOST=${ws}/builds " +
        "BOLT_REF=${BOLT_REF} TRIPLE=${TRIPLE} TARBALL_NAME=${BOLT_TARNAME} " +
        "FDATA_ROOT=${fdataRoot} OUT_DIR=${outDir} BOLT_APPLY=${APPLY_PROFILES == 'true' ? '1' : '0'} " +
        "sbatch --parsable --nodes=1 ${partArgs} internal/slurm_merge.sh"
    return Utils.exec(pipeline, timeout: false, returnStdout: true, numRetries: 1,
                      script: Utils.sshUserCmd(remote, b64BashRemoteCmd(cmd))).trim().tokenize(';')[0].trim()
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
        // Empty: not registered in the accounting DB yet, or a transient blip.
        if (st.isEmpty()) { return false }
        // split() drops trailing empties, so an all-empty row yields a 0-length
        // array; guard it rather than risk an exception failing the stage.
        def fields = st.split(/\|/)
        // State can carry a suffix ("CANCELLED by 12345"), so compare the first
        // token only -- same parse as L0_Test.groovy's `cut -f1 | awk '{print $1}'`.
        def stateTokens = fields.size() > 0 ? fields[0].trim().toUpperCase().tokenize() : []
        if (stateTokens.isEmpty()) { return false }
        def state = stateTokens[0]
        def exitCode = fields.size() > 1 ? fields[1].trim() : ""
        if (SLURM_NON_TERMINAL_STATES.contains(state)) { return false }
        if (state != "COMPLETED" || exitCode != "0:0") {
            error("BoltProfileGen: SLURM job ${jobId} (${label}) did not complete cleanly: ${st}")
        }
        return true
    }
    // The waitUntil wall-clock (what Blue Ocean shows for this stage) lumps SLURM
    // queue wait + actual run time together -- Jenkins can't see inside the
    // scheduler. Split them from sacct so an oversubscribed cluster (long queue)
    // is distinguishable from a genuinely slow job. Diagnostic only.
    logSlurmJobTiming(pipeline, remote, jobId, label)
}

// Best-effort SLURM timing breakdown for a COMPLETED job: queue wait
// (Start-Submit) vs run time (End-Start). Emitted as a single [TIMING] line so
// runs are easy to grep/compare across regions and clusters. NEVER fails the
// caller -- timing is diagnostic, so any sacct/date hiccup is swallowed.
def logSlurmJobTiming(pipeline, remote, String jobId, String label)
{
    try {
        def script = """
            set -o pipefail
            row=\$(sacct -j ${jobId} --format=Submit,Start,End -Pn --allocations | head -1)
            sub=\$(echo "\$row" | cut -d'|' -f1)
            beg=\$(echo "\$row" | cut -d'|' -f2)
            end=\$(echo "\$row" | cut -d'|' -f3)
            ss=\$(date -d "\$sub" +%s 2>/dev/null || echo "")
            bs=\$(date -d "\$beg" +%s 2>/dev/null || echo "")
            es=\$(date -d "\$end" +%s 2>/dev/null || echo "")
            if [ -n "\$ss" ] && [ -n "\$bs" ] && [ -n "\$es" ]; then
                echo "queue=\$((bs-ss))s run=\$((es-bs))s total=\$((es-ss))s"
            else
                echo "unavailable (\$row)"
            fi
        """.stripIndent()
        def t = Utils.exec(pipeline, returnStdout: true, numRetries: 1, timeout: false,
            script: Utils.sshUserCmd(remote, b64BashRemoteCmd(script))).trim()
        pipeline.echo("[TIMING] ${label} job ${jobId}: ${t}")
    } catch (Throwable e) {
        pipeline.echo("[TIMING] ${label} job ${jobId}: timing unavailable (${e.message})")
    }
}

// ---------------------------------------------------------------------------
// Promote the produced bundle to the branch-keyed Artifactory path so premerge
// can pull `latest` (apply_latest.sh). Runs CLUSTER-SIDE: the frontend already
// reaches Artifactory (it pulled the input tarball), and there is no BSL
// remote->agent download primitive, so uploading in place avoids copying the
// bundle back to the agent just to re-upload it.
//
// Auth: the urm-artifactory-creds Jenkins credential (username/password). The
// credential MUST NOT be Groovy-interpolated into the script: Utils.exec passes
// the script to pipeline.sh (which echoes it) and, on any retryable failure,
// hands the same string to _postExecNVDFEvent, which uploads it to NVDF. Jenkins
// masks the literal secret in console output, but this script is base64'd for
// the remote shell, and a base64 blob is not the literal -- so masking would not
// apply and the value would be recoverable from the build log and the telemetry
// record.
//
// So the credential stays a SHELL variable: printf runs on the agent, where
// withCredentials has bound $ART_USER/$ART_PASS, and only the rendered netrc
// travels -- over ssh stdin, which the remote script reads with `cat`. The
// script string therefore contains the literal text "$ART_PASS", never its
// value. printf is a shell builtin and the values are %s arguments rather than
// part of the format string, so there is no `ps` exposure and no risk of a `%`
// in the password being read as a format specifier. noNVDFEvent is belt and
// braces for the telemetry path.
//
// Note this relies on sshUserCmd forwarding stdin (it does not pass ssh -n);
// the same "pipe into sshUserCmd" idiom is used inside the shared lib itself.
// ---------------------------------------------------------------------------
def promoteBundle(pipeline, remote, String bundle)
{
    def bundleName = bundle.tokenize('/').last()
    def base = "${URM_ARTIFACTORY_BASE}/${PROFILE_PROMOTE_DIR}"
    def host = URM_ARTIFACTORY_BASE.replaceFirst(/^https?:\/\//, "").tokenize('/').first()
    def netrc = "${bundle}.netrc"
    pipeline.echo("Promoting ${bundleName} -> ${PROFILE_PROMOTE_DIR}/ (versioned + latest)")
    // Remote side: no credential anywhere in here, just "read it from stdin".
    // Clean up via trap, not a trailing rm: under `set -e` a failed curl exits
    // before the rm and would leave the plaintext netrc on shared scratch until
    // the 7-day retention purge. umask covers the window before the chmod.
    def promote = """
        set -e
        umask 077
        trap 'rm -f "${netrc}"' EXIT
        cat > "${netrc}"
        chmod 600 "${netrc}"
        curl -fsS --netrc-file "${netrc}" --retry 5 --retry-all-errors -T "${bundle}" "${base}/${bundleName}"
        curl -fsS --netrc-file "${netrc}" --retry 5 --retry-all-errors -T "${bundle}" "${base}/latest.tar.gz"
    """.stripIndent()
    pipeline.withCredentials([pipeline.usernamePassword(credentialsId: 'urm-artifactory-creds',
            usernameVariable: 'ART_USER', passwordVariable: 'ART_PASS')]) {
        // \$ART_USER / \$ART_PASS are left for the agent's shell to expand.
        def feed = "printf 'machine ${host} login %s password %s\\n' \"\$ART_USER\" \"\$ART_PASS\" | "
        Utils.exec(pipeline, timeout: false, numRetries: 2, noNVDFEvent: true,
            script: feed + Utils.sshUserCmd(remote, b64BashRemoteCmdStdin(promote, "${bundle}.promote.sh")))
    }
    pipeline.echo("Promoted. latest = ${base}/latest.tar.gz")
}


pipeline {
    agent {
        // Lightweight x86 CPU dispatcher pod; the heavy work runs on SLURM nodes.
        kubernetes createKubernetesPodConfig(AGENT_IMAGE, "amd64")
    }
    // "" default wherever the real default is derived at run time (from targetArch,
    // the resolved partition, or parent env) and so can't be a static value: the
    // resolution chains above handle those, and "" is falsy, so empty == default.
    // gitlabSourceRepoHttpUrl/gitlabCommit are declared because this block REPLACES
    // the job's parameter list on every run, and an undeclared parameter is dropped
    // even when the parent sends it -- so the job config's ${gitlabSourceRepoHttpUrl}
    // SCM URL, which fetches this file, would never resolve. The remaining
    // pass-throughs (artifactCommit, gitlabTargetBranch, boltModelsRoot) are optional
    // and stay undeclared.
    parameters {
        string(
            name: "gitlabSourceRepoHttpUrl",
            defaultValue: "",
            description: "Repo the job config's SCM step clones to read this Jenkinsfile. Passed by the parent, already resolved via its default-llm-repo credential; set explicitly for a standalone run. Left empty here so no repo is hardcoded and the per-instance credential stays authoritative."
        )
        string(
            name: "gitlabCommit",
            defaultValue: "",
            description: "Commit the SCM step checks out, and the boltRef fallback recorded in the bundle manifest. Passed by the parent; empty lets boltRef fall through to \"unknown\"."
        )
        string(
            name: "targetArch",
            defaultValue: "",
            description: "Build triple to profile. Empty -> aarch64-linux-gnu; also selects the boltTarName and slurmPlatform defaults."
        )
        string(
            name: "artifactPath",
            defaultValue: "",
            description: "Artifactory directory holding the input tarball, e.g. sw-tensorrt-generic/llm-artifacts/LLM/main/L0_PostMerge/<build>. Passed by the parent pipeline; set explicitly for a standalone run."
        )
        string(
            name: "dockerImage",
            defaultValue: "",
            description: "Container image the SLURM collect and merge jobs run in; must match targetArch (LLM_SBSA_DOCKER_IMAGE for aarch64). Passed by the parent pipeline."
        )
        string(
            name: "boltRef",
            defaultValue: "",
            description: "Provenance ref recorded in the bundle manifest and versioned bundle name. Empty -> artifactCommit, then gitlabCommit, then \"unknown\"."
        )
        string(
            name: "branch",
            defaultValue: "",
            description: "Branch key of the promote path (.../bolt-profiles/<branch>/<triple>/). Empty -> gitlabTargetBranch, then main."
        )
        string(
            name: "boltTarName",
            defaultValue: "",
            description: "Name of the tarball to profile under artifactPath. Empty -> TensorRT-LLM-GH200.tar.gz for aarch64, TensorRT-LLM.tar.gz for x86_64."
        )
        choice(
            name: "promote",
            choices: ["false", "true"],
            description: "Publish the bundle to the branch-keyed Artifactory path, as a versioned copy and latest.tar.gz. false still builds a bundle, but repoints nothing."
        )
        choice(
            name: "applyProfiles",
            choices: ["true", "false"],
            description: "Re-BOLT the input tarball with the just-generated bundle in the merge job, as a same-commit check that the profiles apply. No extra GPU allocation."
        )
        string(
            name: "slurmPlatform",
            defaultValue: "",
            description: "SlurmConfig platform for the collect and merge jobs. Empty -> gb300-flex-aws-cmh for aarch64, unset for x86_64."
        )
        string(
            name: "numNodes",
            defaultValue: "2",
            description: "Legacy single-workload node count. UNUSED by the fan-out, which sizes each allocation from the workload's own config."
        )
        string(
            name: "boltIterMult",
            defaultValue: "",
            description: "Multiplier on each workload's client iterations (num_requests = concurrency * iterations * mult), for a steady-state profile. Empty -> 64."
        )
        string(
            name: "boltHarnessPartition",
            defaultValue: "",
            description: "SLURM partition for the perf-harness collect jobs. Empty -> the partition of the resolved slurmPlatform."
        )
        string(
            name: "boltHarnessMounts",
            defaultValue: "",
            description: "Container bind mounts for the harness. Empty -> the run workspace plus the models root."
        )
        string(
            name: "boltHarnessTimeLimit",
            defaultValue: "",
            description: "SLURM walltime per collect job. Empty -> 04:00:00 (instrumented runs are much slower than uninstrumented ones)."
        )
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
