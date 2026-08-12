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

// =============================================================================
// L0_ResourceJanitor -- Tier-3 backstop for the pipeline resource ledger.
// =============================================================================
//
// The in-process resourceLedger (trtllm-jenkins-shared-lib) reconciles resources
// when the owning *frame* dies but an ancestor within the same build survives.
// It CANNOT help when the whole build dies (orchestrator/JVM killed, controller
// restart): its @Field state dies with the build. This periodic job is the
// out-of-band backstop for exactly that case -- it enumerates externally-visible
// resources that carry an owner-build tag and reclaims any whose owning Jenkins
// build is no longer running.
//
// Scope of this first cut: orphaned SLURM jobs (the highest-value leak -- they
// outlive the dispatcher pod and keep holding a GPU allocation). K8s agent nodes
// / pods are largely self-reaped by the kubernetes plugin, so they are a later
// extension (see reapOrphanedNodes stub).
//
// SAFETY MODEL (fail safe, never reap a live build's resource):
//   * DRY_RUN defaults to true -- prints what it *would* reap, changes nothing.
//   * Only ever considers resources carrying OUR owner tag (OWNER_TAG_KEY); a job
//     with no CI owner tag is never touched.
//   * Reaps only when the owner build is *confirmed not running*. If the
//     owner-alive check errors or is inconclusive, the resource is LEFT (assume
//     alive).
//   * STALE_MINUTES grace period on top of that, so a job whose build just
//     started (or whose API status lags) is not reaped.
//   * Per-cluster best-effort: one cluster's failure never aborts the others.
//
// COMPANION (same change set): SLURM submission in L0_Test.groovy stamps the owner
// tag onto each job's `--comment` on both the sbatch path (#SBATCH --comment) and
// the agent path (addSlurmOwnerComment); its SLURM_OWNER_TAG_KEY must stay in sync
// with OWNER_TAG_KEY below. A job that predates the tag, or that the agent-path
// injector left untouched, simply carries no owner tag and is never a reap
// candidate (safe).
//
// INFRA-TODO (fill in with the team before enabling non-dry-run):
//   * Jenkins job config: schedule (cron), which Jenkins instance(s) it runs on.
//   * Agent pod must have SLURM login-node SSH access + the svc_tensorrt creds
//     that CloudManager.withSlurmFrontendFailover expects (same as L0_Test SLURM
//     dispatchers).
//   * Cross-instance owner-build lookup auth (multi-server); see isOwnerBuildRunning.
//   * Confirm squeue field/comment format on the target Slurm version.
// =============================================================================

@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.Utils

// Structured owner tag carried in each CI SLURM job's --comment. The companion
// submission change writes "trtllm-ci-owner=<BUILD_URL>" (co-existing with any
// other comment payload, ';'-separated); this job parses <BUILD_URL> back out and
// queries "<BUILD_URL>api/json?tree=building" to decide whether the owner is alive.
OWNER_TAG_KEY = "trtllm-ci-owner"

// SLURM account the CI jobs run under; only this user's jobs are ever considered.
SLURM_CI_USER = "svc_tensorrt"

// Janitor pod image: must have curl + an ssh client, since it queries the Jenkins
// API and reaches the SLURM login nodes over ssh (via CloudManager). Reuse a
// TRT-LLM CI image + its pull secret rather than a bare alpine (no curl/ssh).
// INFRA-TODO: confirm this image + secret match the SLURM dispatcher pool.
DOCKER_IMAGE = "artifactory.nvidia.com/sw-tensorrt-llm-docker-local/tensorrt-llm:pytorch-25.10-py3-x86_64-ubuntu24.04-trt10.13.3.9-skip-tritondevel-202510291120-8621"
ARTIFACTORY_IMAGE_PULL_SECRET = "trtllm-artifactory"

// Owner build URLs come from job comments (attacker-influenceable), so before use
// they are validated against this Jenkins host allowlist + a conservative charset,
// AND passed to curl via an environment variable rather than interpolated into the
// shell -- a crafted comment cannot inject shell code into the janitor pod.
// INFRA-TODO: confirm the full set of Jenkins hosts (multi-server).
JENKINS_HOST_ALLOWLIST = ["prod.blsm.nvidia.com"]

// True only for a plain https Jenkins build URL on an allowed host whose path uses
// a shell-safe charset (no metacharacters). Defense-in-depth over env-var passing.
@NonCPS
boolean isAllowedJenkinsBuildUrl(String url) {
    if (!url) {
        return false
    }
    def m = (url =~ /^https:\/\/([A-Za-z0-9.-]+)\/[A-Za-z0-9._~:\/%-]*$/)
    return m.matches() && JENKINS_HOST_ALLOWLIST.contains(m.group(1))
}

def createKubernetesPodConfig()
{
    // A small CPU pod that can reach the SLURM login nodes. Mirrors the SLURM
    // dispatcher pods so CloudManager.withSlurmFrontendFailover has what it needs.
    // INFRA-TODO: confirm image/creds/network match the SLURM dispatcher pool.
    def jnlpImage = "artifactory.pdx.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_amd_linux:jdk17"
    return [
        cloud: "kubernetes-cpu",
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                qosClass: Guaranteed
                imagePullSecrets:
                  - name: ${ARTIFACTORY_IMAGE_PULL_SECRET}
                nodeSelector:
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux
                containers:
                  - name: trt-llm
                    image: ${DOCKER_IMAGE}
                    command: ['cat']
                    tty: true
                    resources:
                      requests: { cpu: '1', memory: 2Gi, ephemeral-storage: 25Gi }
                      limits:   { cpu: '1', memory: 2Gi, ephemeral-storage: 25Gi }
                    imagePullPolicy: Always
                  - name: jnlp
                    image: ${jnlpImage}
                    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
                    resources:
                      requests: { cpu: '1', memory: 2Gi, ephemeral-storage: 5Gi }
                      limits:   { cpu: '1', memory: 2Gi, ephemeral-storage: 5Gi }
        """.stripIndent(),
    ]
}

// True if the owner Jenkins build is still running. FAIL SAFE: any error or
// ambiguity returns true (treat as alive) so we never reap a live build's job.
// INFRA-TODO: cross-instance auth -- derive the right API token credential from
// the build URL's instance (see trtllm_utils.appendBuildDescription for the
// instance->credential derivation pattern) instead of a single token.
boolean isOwnerBuildRunning(pipeline, String buildUrl) {
    if (!buildUrl) {
        return true
    }
    if (!isAllowedJenkinsBuildUrl(buildUrl)) {
        pipeline.echo "[JANITOR] owner tag is not an allowed Jenkins build URL; leaving job (not reaping)."
        return true   // fail safe -- never reap on an untrusted / garbled owner tag
    }
    try {
        // credentialsId placeholder -- see INFRA-TODO above.
        return pipeline.withCredentials([pipeline.usernamePassword(
                credentialsId: 'TOP_1_TOKEN', usernameVariable: 'J_USER', passwordVariable: 'J_TOKEN')]) {
            // Pass the (validated) URL via the environment and reference it quoted in
            // a single-quoted script -- the comment-sourced value is never interpolated
            // into the shell command text, so it cannot inject even if validation is
            // ever loosened. Bounded timeouts keep a stalled controller from blocking
            // the whole serial scan; || true keeps a timeout fail-safe (inconclusive).
            return pipeline.withEnv(["OWNER_BUILD_URL=${buildUrl}"]) {
                def json = pipeline.sh(
                    script: 'curl -sf --connect-timeout 10 --max-time 30 ' +
                            '-u "${J_USER}:${J_TOKEN}" "${OWNER_BUILD_URL}api/json?tree=building" || true',
                    returnStdout: true).trim()
                if (!json) {
                    pipeline.echo "[JANITOR] owner-alive check inconclusive; assuming RUNNING (leave)."
                    return true
                }
                // Reap only on an explicit building:false.
                return !(json =~ /"building"\s*:\s*false/)
            }
        }
    } catch (Exception e) {
        pipeline.echo "[JANITOR] owner-alive check failed (${e}); assuming RUNNING (leave)."
        return true
    }
}

// Parse the owner BUILD_URL out of a SLURM job comment, or null if the job carries
// no CI owner tag (in which case it is never a reap candidate).
@NonCPS
String parseOwnerBuildUrl(String comment) {
    if (!comment) {
        return null
    }
    def m = (comment =~ /${OWNER_TAG_KEY}=([^;\s]+)/)
    return m ? m.group(1) : null
}

// Reap orphaned SLURM jobs on one cluster. Best-effort; logs and returns a small
// summary map. Never throws out to the caller.
def reapOrphanedSlurmJobsOnCluster(pipeline, String clusterName, def cluster, boolean dryRun, int staleMinutes) {
    def summary = [cluster: clusterName, scanned: 0, orphans: 0, reaped: 0]
    try {
        CloudManager.withSlurmFrontendFailover(pipeline, clusterName, cluster) { remote ->
            // %i=jobid, %k=comment, %M=elapsed. -h: no header. Only our CI user.
            // INFRA-TODO: confirm %k carries the full comment on this Slurm version.
            // No `|| true`: a squeue / controller failure must surface to the
            // per-cluster handler below (logged as a failed scan) rather than look
            // like "no jobs" and silently skip reaping.
            def raw = Utils.exec(pipeline,
                script: Utils.sshUserCmd(remote, "\"squeue -u ${SLURM_CI_USER} -h -o '%i|%k|%M'\""),
                returnStdout: true).trim()

            raw.readLines().each { line ->
                def parts = line.split(/\|/, -1)
                if (parts.size() < 3) {
                    return
                }
                summary.scanned++
                def jobId = parts[0].trim()
                // jobId is SLURM-sourced (%i), not from the comment, but validate it
                // before interpolating into scancel.
                if (!(jobId ==~ /\d+(?:_\d+)?/)) {
                    pipeline.echo "[JANITOR] ${clusterName}: skipping unexpected job id '${jobId}'."
                    return
                }
                def ownerUrl = parseOwnerBuildUrl(parts[1])
                def ageMin = parseElapsedMinutes(parts[2])
                if (!ownerUrl) {
                    return   // not a CI-tagged job -- never touch
                }
                if (ageMin == null) {
                    // Unparsable squeue %M: we can't confirm the job is past the grace
                    // period, so fail safe and leave it (never reap on an unknown age).
                    pipeline.echo "[JANITOR] ${clusterName}: unparsable elapsed for job ${jobId}; leaving untouched."
                    return
                }
                if (ageMin < staleMinutes) {
                    return   // grace period -- too young to reap
                }
                if (isOwnerBuildRunning(pipeline, ownerUrl)) {
                    return   // owner alive (or check inconclusive) -- leave it
                }
                summary.orphans++
                if (dryRun) {
                    pipeline.echo "[JANITOR] DRY_RUN ${clusterName}: would scancel job ${jobId} " +
                                  "(owner not running, age=${ageMin}m)."
                    return
                }
                pipeline.echo "[JANITOR] ${clusterName}: reaping orphaned job ${jobId} (owner not running)."
                // Confirm the cancel actually succeeded before counting it reaped; a
                // failed scancel is logged and retried on the next run.
                def scancelOut = Utils.exec(pipeline, returnStdout: true, script: Utils.sshUserCmd(remote,
                    "\"scancel ${jobId} && echo JANITOR_SCANCEL_OK || echo JANITOR_SCANCEL_FAIL; " +
                    "sacct -j ${jobId} --format=JobID,State,ExitCode -Pn || true\""))
                if (scancelOut?.contains('JANITOR_SCANCEL_OK')) {
                    summary.reaped++
                } else {
                    pipeline.echo "[JANITOR] ${clusterName}: scancel of ${jobId} did not confirm success; will retry next run."
                }
            }
        }
    } catch (Exception e) {
        pipeline.echo "[JANITOR] ${clusterName}: scan failed (${e}); skipping this cluster."
    }
    return summary
}

// Approximate SLURM elapsed "[[D-]HH:]MM:SS" -> minutes, or null if unparsable.
@NonCPS
Integer parseElapsedMinutes(String elapsed) {
    if (!elapsed?.trim()) {
        return null
    }
    try {
        def s = elapsed.trim()
        int days = 0
        if (s.contains('-')) {
            def dp = s.split('-', 2)
            days = dp[0].toInteger()
            s = dp[1]
        }
        def f = s.split(':')
        int h = 0, m = 0
        if (f.size() == 3)      { h = f[0].toInteger(); m = f[1].toInteger() }
        else if (f.size() == 2) { m = f[0].toInteger() }
        else                    { return 0 }
        return days * 24 * 60 + h * 60 + m
    } catch (Exception ignored) {
        return null
    }
}

// K8s agent-node / pod reaping -- deferred. These are mostly self-reaped by the
// kubernetes plugin; add here only if leaks are observed in practice.
def reapOrphanedNodes(pipeline, boolean dryRun) {
    pipeline.echo "[JANITOR] node reaping not implemented yet (k8s plugin self-reaps); skipping."
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig()
    }
    options {
        timestamps()
        // A single long-running instance is pointless; keep runs short and bounded.
        timeout(time: 30, unit: 'MINUTES')
    }
    // INFRA-TODO: enable a schedule once validated in DRY_RUN, e.g.
    // triggers { cron('H/30 * * * *') }
    parameters {
        booleanParam(name: 'DRY_RUN', defaultValue: true,
            description: 'Log what would be reaped without cancelling anything.')
        string(name: 'STALE_MINUTES', defaultValue: '120',
            description: 'Only reap jobs older than this many minutes (grace period).')
        string(name: 'CLUSTER', defaultValue: 'all',
            description: "Cluster name to scan, or 'all'.")
    }
    stages {
        stage("Reap orphaned SLURM jobs") {
            steps {
                container("trt-llm") {
                    script {
                        boolean dryRun = params.DRY_RUN
                        int staleMinutes
                        try {
                            staleMinutes = (params.STALE_MINUTES ?: '120').toInteger()
                        } catch (NumberFormatException nfe) {
                            echo "[JANITOR] invalid STALE_MINUTES='${params.STALE_MINUTES}'; defaulting to 120."
                            staleMinutes = 120
                        }
                        if (staleMinutes < 0) {
                            echo "[JANITOR] STALE_MINUTES ${staleMinutes} is negative; clamping to 0 (no negative grace)."
                            staleMinutes = 0
                        }
                        def summaries = []

                        def clusters = SlurmConfig.clusterConfig.findAll { name, cfg ->
                            params.CLUSTER == 'all' || params.CLUSTER == name
                        }
                        echo "[JANITOR] scanning ${clusters.size()} cluster(s); DRY_RUN=${dryRun}, staleMinutes=${staleMinutes}."

                        clusters.each { clusterName, cluster ->
                            summaries << reapOrphanedSlurmJobsOnCluster(this, clusterName, cluster, dryRun, staleMinutes)
                        }

                        reapOrphanedNodes(this, dryRun)

                        def totalOrphans = summaries.sum { it.orphans } ?: 0
                        def totalReaped  = summaries.sum { it.reaped }  ?: 0
                        echo "[JANITOR] done. orphans found=${totalOrphans}, reaped=${totalReaped} " +
                             "(dryRun=${dryRun}). Per-cluster: ${summaries}"
                        currentBuild.description = "orphans=${totalOrphans} reaped=${totalReaped} dryRun=${dryRun}"
                    }
                }
            }
        }
    }
}
