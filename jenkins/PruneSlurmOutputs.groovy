// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Prunes SLURM job output that L0_Test keeps on the clusters for debugging:
// sbatch job workspaces (bloom/scripts/<host>-multi_node_test-*) in which no
// file has changed within the retention period, and agent job logs
// (slurm-logs/slurm-<jobid>-<host>-test-*.out) older than it. Also prunes enroot
// layer cache files neither read nor modified within the retention period.

@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.Utils

UBUNTU_24_04_IMAGE = "artifactory.pdx.nvidia.com/docker-remote/ubuntu:24.04"
// Clusters left out of the default list: computelab does not accept
// svc_tensorrt logins from this job. They can still be named in CLUSTERS.
DEFAULT_EXCLUDED_CLUSTERS = ["computelab"]

def createKubernetesPodConfig(image)
{
    def jnlpImage = "artifactory.pdx.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_amd_linux:jdk17"

    def podConfig = [
        cloud: "kubernetes-cpu",
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                nodeSelector:
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux
                containers:
                  - name: trt-llm
                    image: ${image}
                    command: ['cat']
                    tty: true
                    resources:
                      requests:
                        cpu: 1
                        memory: 2Gi
                        ephemeral-storage: 5Gi
                      limits:
                        cpu: 1
                        memory: 2Gi
                        ephemeral-storage: 5Gi
                    imagePullPolicy: Always
                  - name: jnlp
                    image: ${jnlpImage}
                    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
                    resources:
                      requests:
                        cpu: '1'
                        memory: 2Gi
                        ephemeral-storage: 5Gi
                      limits:
                        cpu: '1'
                        memory: 2Gi
                        ephemeral-storage: 5Gi
                qosClass: Guaranteed
        """.stripIndent(),
    ]

    return podConfig
}

// Clusters CI submits to with the default svc_tensorrt credentials; partitions
// with their own credentialId (e.g. QA) run under other accounts.
@NonCPS
def defaultClusterNames() {
    return SlurmConfig.partitionConfig.values()
        .findAll { !it.credentialId }
        .collect { it.clusterName }
        .unique()
        .findAll { !(it in DEFAULT_EXCLUDED_CLUSTERS) }
        .sort()
}

// Runs under bash on the login node. -H follows the top-level dirs, which some
// clusters symlink to scratch storage. Avoids `$` so no shell along the ssh path
// expands anything.
def pruneCommands(String homeDir, String days, boolean dryRun) {
    def removeDir = dryRun ? "echo would remove {}" : "(echo removing {} && rm -rf {})"
    def deleteLog = dryRun ? "" : "-delete"
    // Agent logs can number in the hundreds of thousands, so print only the count and
    // the oldest few / newest one (sorted by mtime) instead of every name.
    def summarizeLogs = "awk 'NR <= 3 { printf \"  oldest: \"; print } END { if (NR) { printf \"  newest: \"; print } print \"agent logs ${dryRun ? 'to prune' : 'pruned'}: \" NR }'"
    return [
        "find -H ${homeDir}/bloom/scripts -mindepth 1 -maxdepth 1 -type d -name '*-multi_node_test-*' -exec bash -c 'find {} -mtime -${days} -print -quit | grep -q . || ${removeDir}' \\; 2>/dev/null || true",
        "find -H ${homeDir}/slurm-logs -maxdepth 1 -type f -name 'slurm-*-test-*.out' ! -mtime -${days} -printf '%TY-%Tm-%Td %p\\n' ${deleteLog} 2>/dev/null | sort | ${summarizeLogs}",
    ].join(" ; ")
}

// Prunes the enroot layer cache L0_Test imports images through (ENROOT_CACHE_PATH):
// layer blobs and partial downloads, which sit directly in the cache dir, neither
// read nor modified within `days`. An import that reuses a cached layer reads it
// but keeps its mtime at the first download, so the access time is what shows a
// layer is still in use; on filesystems mounted noatime this falls back to mtime.
// Reports the cache before pruning and what is pruned.
def enrootCachePruneCommands(String homeDir, String days, boolean dryRun) {
    def cacheDir = "${homeDir}/.cache/enroot"
    def findFiles = "find -H ${cacheDir} -maxdepth 1 -type f"
    def findStale = "${findFiles} ! -atime -${days} ! -mtime -${days}"
    def listByAtime = "-printf '%AY-%Am-%Ad %TY-%Tm-%Td %p\\n' 2>/dev/null | sort"
    def summarize = { int oldestCount, String label ->
        "awk 'NR <= ${oldestCount} { printf \"  oldest: \"; print } END { if (NR) { printf \"  newest: \"; print } print \"  ${label}: \" NR }'"
    }
    def pruneLabel = "not read or modified within ${days} days ${dryRun ? 'to prune' : 'pruned'}"
    def commands = [
        "echo 'enroot cache ${cacheDir} (last read, last modified, path):'",
        "du -shH ${cacheDir} 2>/dev/null | awk '{ printf \"  size: \"; print }'",
        "${findFiles} ${listByAtime} | ${summarize(1, 'files')}",
        "${findStale} ${listByAtime} | ${summarize(3, pruneLabel)}",
        "${findStale} -print0 2>/dev/null | du -ch --files0-from=- 2>/dev/null | tail -n 1 | awk '{ printf \"  size ${dryRun ? 'to prune' : 'pruned'}: \"; print }'",
    ]
    if (!dryRun) {
        commands << "${findStale} -delete 2>/dev/null || true"
    }
    return commands.join(" ; ")
}

def pruneCluster(String clusterName, String days, boolean dryRun) {
    def cluster = SlurmConfig.clusterConfig[clusterName]
    if (!cluster) {
        error("Unknown SLURM cluster '${clusterName}'.")
    }
    def homeDir = cluster.homeDir ?: "/home/svc_tensorrt"
    CloudManager.withSlurmFrontendFailover(this, clusterName, cluster) { remote ->
        Utils.exec(
            this,
            // Utils.exec defaults to a 10-minute timeout; the per-cluster stage
            // timeout bounds this instead.
            timeout: 0,
            script: Utils.sshUserCmd(
                remote,
                Utils.bashWrappedRemoteCmd(
                    pruneCommands(homeDir, days, dryRun) + " ; " +
                        enrootCachePruneCommands(homeDir, days, dryRun)
                )
            )
        )
    }
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig(UBUNTU_24_04_IMAGE)
    }
    options {
        timestamps()
        timeout(time: 2, unit: 'HOURS')
        disableConcurrentBuilds()
    }
    triggers {
        cron('H 3 * * 0')
    }
    parameters {
        string(
            name: 'RETENTION_DAYS',
            defaultValue: '10',
            description: 'Prune SLURM job output not modified, and enroot layer cache files neither read nor modified, within this many days.')
        string(
            name: 'CLUSTERS',
            defaultValue: '',
            description: "Comma-separated SLURM cluster names to prune. Empty means every cluster CI submits to, except ${DEFAULT_EXCLUDED_CLUSTERS.join(', ')}.")
        booleanParam(
            name: 'DRY_RUN',
            defaultValue: false,
            description: 'When true, only list what would be pruned.')
    }
    stages {
        stage('Prune SLURM Outputs') {
            steps {
                container('trt-llm') {
                    script {
                        if (!(params.RETENTION_DAYS ==~ /^[1-9][0-9]*$/)) {
                            error("RETENTION_DAYS must be a positive integer (got: '${params.RETENTION_DAYS}').")
                        }
                        def clusters = params.CLUSTERS?.trim()
                            ? params.CLUSTERS.split(',').collect { it.trim() }.findAll { it }
                            : defaultClusterNames()
                        echo "Pruning output older than ${params.RETENTION_DAYS} days on: ${clusters.join(', ')}"

                        // Sequential: the shared lib installs ssh tools with apt
                        // on every connection, which cannot run concurrently.
                        for (clusterName in clusters) {
                            stage(clusterName) {
                                // One unreachable cluster must not block the others.
                                catchError(buildResult: 'UNSTABLE', stageResult: 'FAILURE') {
                                    timeout(time: 1, unit: 'HOURS') {
                                        pruneCluster(clusterName, params.RETENTION_DAYS, params.DRY_RUN)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
