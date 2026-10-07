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
// SAME-INSTANCE MODEL: CI SLURM jobs are launched from per-instance Jenkins
// servers, not from a central one, and cross-instance API auth is not set up. So
// this job runs per-instance (one cron per Jenkins instance) and checks each
// job's owner build via the in-process Jenkins Groovy API (isOwnerBuildRunning);
// a build URL belonging to a *different* instance is left for that instance's
// janitor. Using the in-process API (rather than an HTTP call) means there is no
// shell command and thus no way for an attacker-influenceable job comment to
// inject anything, and no credential to manage.
//
// SAFETY MODEL (fail safe, never reap a live build's resource):
//   * DRY_RUN defaults to true -- prints what it *would* reap, changes nothing.
//   * Only ever considers resources carrying OUR owner tag (OWNER_TAG_KEY); a job
//     with no CI owner tag is never touched.
//   * Reaps only when the owner build is *confirmed not running* on THIS instance.
//     Any parse error, unknown build, or other-instance URL leaves the job.
//   * STALE_MINUTES grace period; an unparsable elapsed value leaves the job.
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
//   * Jenkins job config: schedule (cron) on each instance; "Pipeline script from
//     SCM" so readTrusted() can read jenkins/current_image_tags.properties.
//   * Work-pod must have an ssh client + the svc_tensorrt creds that
//     CloudManager.withSlurmFrontendFailover expects (same as L0_Test SLURM
//     dispatchers).
//   * Confirm squeue field format (%k comment, %M elapsed) on the target Slurm
//     version.
// =============================================================================

@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.Utils
import jenkins.model.Jenkins

// Structured owner tag carried in each CI SLURM job's --comment. L0_Test writes
// "trtllm-ci-owner=<BUILD_URL>" (co-existing with any other comment payload,
// ';'-separated); this job parses <BUILD_URL> back out and checks the build.
OWNER_TAG_KEY = "trtllm-ci-owner"

// SLURM account the CI jobs run under; only this user's jobs are ever considered.
SLURM_CI_USER = "svc_tensorrt"

// K8s secret for pulling the work-pod image from artifactory.nvidia.com.
ARTIFACTORY_IMAGE_PULL_SECRET = "trtllm-artifactory"

// True if the owner Jenkins build is still running, via the in-process Jenkins
// API (no HTTP, no credentials, no shell). Only same-instance builds are checked:
// a build URL on another instance (or that can't be parsed) returns true so it is
// left for that instance's janitor. FAIL SAFE: any error / unknown item returns
// true (treat as alive). A build that no longer exists returns false (reapable).
@NonCPS
boolean isOwnerBuildRunning(String buildUrl) {
    if (!buildUrl) {
        return true
    }
    try {
        def jenkins = Jenkins.instance
        def rootUrl = jenkins?.rootUrl
        if (!rootUrl || !buildUrl.startsWith(rootUrl)) {
            return true   // another instance's build (or unknown) -- not ours
        }
        // "<root>/job/A/job/B/123/" -> jobFullName "A/B", build number 123.
        def rel = buildUrl.substring(rootUrl.length()).replaceAll('/+$', '')
        def flat = rel.replaceAll('(^|/)job/', '/').replaceAll('^/+', '')
        def segs = flat.split('/')
        if (segs.length < 2 || !segs[-1].isInteger()) {
            return true
        }
        def job = jenkins.getItemByFullName(segs[0..-2].join('/'))
        if (job == null) {
            return true   // unknown job -- fail safe
        }
        def build = job.getBuildByNumber(segs[-1].toInteger())
        if (build == null) {
            return false  // build no longer exists -> owner is gone -> reapable
        }
        return build.isBuilding()
    } catch (Exception ignored) {
        return true       // fail safe -- never reap on an ambiguous check
    }
}

// Extract a "KEY=value" image tag from current_image_tags.properties text. Pure /
// node-free so it can run on the controller alongside readTrusted.
@NonCPS
String parseImageTag(String propsText, String key) {
    if (!propsText || !key) {
        return null
    }
    def pattern = ~("(?m)^\\s*" + java.util.regex.Pattern.quote(key) + "\\s*=\\s*(\\S+)\\s*\$")
    def m = pattern.matcher(propsText)
    return m.find() ? m.group(1) : null
}

// Work pod: reaches the SLURM login nodes over ssh (CloudManager). Image is read
// from current_image_tags.properties at runtime rather than hardcoded.
// INFRA-TODO: confirm image/creds/network match the SLURM dispatcher pool.
def createKubernetesPodConfig(image)
{
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
                    image: ${image}
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
                      requests: { cpu: '1', memory: 2Gi, ephemeral-storage: 25Gi }
                      limits:   { cpu: '1', memory: 2Gi, ephemeral-storage: 25Gi }
        """.stripIndent(),
    ]
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
            // -r expands array jobs to one identifier per line (so %i is a concrete
            // id, not a compressed "123_[1-4]"). %i=jobid, %k=comment, %M=elapsed.
            // No `|| true`: a squeue / controller failure must surface to the
            // per-cluster handler below (logged as a failed scan) rather than look
            // like "no jobs" and silently skip reaping.
            def raw = Utils.exec(pipeline,
                script: Utils.sshUserCmd(remote, "\"squeue -u ${SLURM_CI_USER} -h -r -o '%i|%k|%M'\""),
                returnStdout: true).trim()

            raw.readLines().each { line ->
                def parts = line.split(/\|/, -1)
                if (parts.size() < 3) {
                    return
                }
                summary.scanned++
                def jobId = parts[0].trim()
                // jobId is SLURM-sourced (%i), not from the comment, but validate it
                // before interpolating into scancel. Accept an array element ("_")
                // or a heterogeneous-job component ("+").
                if (!(jobId ==~ /\d+(?:[_+]\d+)?/)) {
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
                if (isOwnerBuildRunning(ownerUrl)) {
                    return   // owner alive / other instance / inconclusive -- leave it
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
    agent none
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

                    // Read the work-pod image from the repo (single source of truth:
                    // jenkins/current_image_tags.properties) instead of hardcoding it.
                    // readTrusted fetches the file from the same SCM revision as this
                    // Jenkinsfile on the controller -- no node / checkout / bootstrap
                    // pod. Requires the job to be "Pipeline script from SCM".
                    def propsText
                    try {
                        propsText = readTrusted('jenkins/current_image_tags.properties')
                    } catch (Exception e) {
                        error "[JANITOR] could not read current_image_tags.properties via readTrusted (${e}); " +
                              "configure this job as 'Pipeline script from SCM'."
                    }
                    def podImage = parseImageTag(propsText, 'LLM_DOCKER_IMAGE')
                    if (!podImage) {
                        error "[JANITOR] could not resolve LLM_DOCKER_IMAGE from current_image_tags.properties."
                    }
                    echo "[JANITOR] work-pod image: ${podImage}"

                    def summaries = []
                    trtllm_utils.launchKubernetesPod(this, createKubernetesPodConfig(podImage), "trt-llm") {
                        def clusters = SlurmConfig.clusterConfig.findAll { name, cfg ->
                            params.CLUSTER == 'all' || params.CLUSTER == name
                        }
                        echo "[JANITOR] scanning ${clusters.size()} cluster(s); DRY_RUN=${dryRun}, staleMinutes=${staleMinutes}."

                        clusters.each { clusterName, cluster ->
                            summaries << reapOrphanedSlurmJobsOnCluster(this, clusterName, cluster, dryRun, staleMinutes)
                        }
                        reapOrphanedNodes(this, dryRun)
                    }

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
