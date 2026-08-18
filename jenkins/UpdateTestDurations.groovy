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

@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

LLM_ROOT = "llm"

UBUNTU_24_04_IMAGE = "urm.nvidia.com/docker/ubuntu:24.04"
DURATION_FILE_PATH = "tests/integration/defs/.test_durations"
// Target repository the updated duration file is committed straight back into.
TARGET_REPO = "NVIDIA/TensorRT-LLM"

def sanityCheckItemCount(String oldPath, String newPath) {
    def countItems = { path ->
        sh(script: "python3 -c \"import json; print(len(json.load(open('${path}'))))\"",
           returnStdout: true).trim() as Integer
    }
    def oldCount = countItems(oldPath)
    def newCount = countItems(newPath)
    echo "Duration-file item counts -> old: ${oldCount}, new: ${newCount}"
    if (oldCount == 0) {
        error("Existing duration file is empty or missing; aborting.")
    }
    def diffPct = Math.abs(newCount - oldCount) * 100.0 / oldCount
    echo "Item-count difference: ${String.format('%.1f', diffPct)}%"
    if (diffPct >= 50.0) {
        error("Item-count difference ${String.format('%.1f', diffPct)}% " +
              ">= 50%; refusing to auto-commit. Download the archived " +
              "new_test_durations.json, review, and upload manually.")
    }
}

def createKubernetesPodConfig(image, arch = "amd64")
{
    def archSuffix = arch == "arm64" ? "arm" : "amd"
    def jnlpImage = "artifactory.pdx.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_${archSuffix}_linux:jdk17"

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
                    volumeMounts:
                    - name: sw-tensorrt-pvc
                      mountPath: "/mnt/sw-tensorrt-pvc"
                      readOnly: false
                    tty: true
                    resources:
                      requests:
                        cpu: 2
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: 2
                        memory: 5Gi
                        ephemeral-storage: 25Gi
                    imagePullPolicy: Always
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
                qosClass: Guaranteed
                volumes:
                - name: sw-tensorrt-pvc
                  persistentVolumeClaim:
                    claimName: sw-tensorrt-pvc
        """.stripIndent(),
    ]

    return podConfig
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig(UBUNTU_24_04_IMAGE)
    }
    options {
        timestamps()
        timeout(time: 1, unit: 'HOURS')
        disableConcurrentBuilds(abortPrevious: true)
    }
    triggers {
        cron('H 2 * * 1')
    }
    parameters {
        string(
            name: 'DAYS',
            defaultValue: '7',
            description: 'Number of days to look back in OpenSearch for test durations (e.g. 3, 7, 14). ')
        string(
            name: 'SOURCE_REPO',
            defaultValue: 'NVIDIA/TensorRT-LLM',
            description: 'GitHub repo to checkout scripts from (e.g. EmmaQiaoCh/TensorRT-LLM for testing).')
        string(
            name: 'TARGET_BRANCH',
            defaultValue: 'main',
            description: 'Branch of the target repo to commit the updated duration file to.')
        booleanParam(
            name: 'DRY_RUN',
            defaultValue: false,
            description: 'When true, generate the duration file but skip the commit/push.')
    }
    environment {
        OPEN_SEARCH_DB_BASE_URL = credentials('open_search_db_base_url')
    }
    stages {
        stage('Setup') {
            steps {
                container('trt-llm') {
                    script {
                        if (!(params.DAYS ==~ /^[1-9][0-9]*$/)) {
                            error("DAYS parameter must be a positive integer (got: '${params.DAYS}').")
                        }
                        if (!(params.SOURCE_REPO ==~ /^[a-zA-Z0-9\/_.-]+$/)) {
                            error("Invalid SOURCE_REPO: '${params.SOURCE_REPO}'. Must match [a-zA-Z0-9/_.-]+.")
                        }
                        if (!(params.TARGET_BRANCH ==~ /^[a-zA-Z0-9\/_.-]+$/)) {
                            error("Invalid TARGET_BRANCH: '${params.TARGET_BRANCH}'. Must match [a-zA-Z0-9/_.-]+.")
                        }
                    }
                    sh """
                        apt-get update -qq && \
                        apt-get install -y -qq git python3-pip curl && \
                        pip3 install --quiet --break-system-packages requests pyyaml
                    """
                }
            }
        } // stage Setup

        stage('Checkout') {
            steps {
                container('trt-llm') {
                    script {
                        def sourceRepo = "https://github.com/${params.SOURCE_REPO}.git"
                        // Initialize submodules for older arbitrary refs; this is a no-op after their removal.
                        trtllm_utils.checkoutSource(sourceRepo, params.TARGET_BRANCH, LLM_ROOT, true, false)
                    }
                }
            }
        } // stage Checkout

        stage('Generate Duration File') {
            steps {
                container('trt-llm') {
                    sh """
                        cd ${LLM_ROOT}
                        python3 jenkins/scripts/generate_duration.py \
                            --days ${params.DAYS} \
                            --duration-file new_test_durations.json
                        echo "Generated file size: \$(wc -l < new_test_durations.json) lines"
                        echo "Sample output (first 5 lines):"
                        head -5 new_test_durations.json

                    """

                    // Always archive the freshly generated file so the user can download
                    // it and upload manually if the job later refuses to auto-commit.
                    archiveArtifacts(
                        artifacts: "${LLM_ROOT}/new_test_durations.json",
                        fingerprint: true)

                    // Sanity gate for DRY_RUN only: fetch TARGET_REPO baseline via HTTP
                    // (no credentials available here) and fail early if the generated file
                    // diverges too much. For non-dry-runs the gate runs after the git reset
                    // in 'Commit and Push', where the baseline comes from the authoritative
                    // TARGET_REPO workspace rather than a separate HTTP fetch.
                    script {
                        if (params.DRY_RUN) {
                            sh """
                                curl -sSf "https://raw.githubusercontent.com/${TARGET_REPO}/${params.TARGET_BRANCH}/${DURATION_FILE_PATH}" \
                                    -o ${LLM_ROOT}/target_test_durations.json
                            """
                            sanityCheckItemCount(
                                "${LLM_ROOT}/target_test_durations.json",
                                "${LLM_ROOT}/new_test_durations.json")
                        }
                    }
                }
            }
        } // stage Generate Duration File

        stage('Commit and Push') {
            when {
                expression { !params.DRY_RUN }
            }
            steps {
                container('trt-llm') {
                    script {
                        sh """
                            cd ${LLM_ROOT}
                            git config --global --add safe.directory \$(pwd)
                            git config user.email "90828364+tensorrt-cicd@users.noreply.github.com"
                            git config user.name "TensorRT LLM"
                        """

                        withCredentials([usernamePassword(
                            credentialsId: 'github-cred-trtllm-ci',
                            usernameVariable: 'NOT_IN_USE',
                            passwordVariable: 'GITHUB_API_TOKEN')]) {
                            // Use the token directly in the URL (shell variable, not Groovy interpolation)
                            // to avoid persisting credentials to .git/config via git remote set-url.
                            int maxRetries = 3
                            for (int attempt = 1; attempt <= maxRetries; attempt++) {
                                if (attempt > 1) {
                                    echo "Push rejected; retrying (attempt ${attempt}/${maxRetries})..."
                                    sleep(15)
                                }

                                // Fetch and reset to the latest remote HEAD before applying the
                                // generated file, so concurrent pushes don't cause conflicts.
                                sh """
                                    cd ${LLM_ROOT}
                                    git fetch "https://svc_tensorrt:\${GITHUB_API_TOKEN}@github.com/${TARGET_REPO}.git" ${params.TARGET_BRANCH}
                                    git reset --hard FETCH_HEAD
                                """

                                // Sanity gate against the authoritative TARGET_REPO baseline
                                // (post-reset workspace, not SOURCE_REPO or a separate HTTP fetch).
                                sanityCheckItemCount(
                                    "${LLM_ROOT}/${DURATION_FILE_PATH}",
                                    "${LLM_ROOT}/new_test_durations.json")

                                sh "cp ${LLM_ROOT}/new_test_durations.json ${LLM_ROOT}/${DURATION_FILE_PATH}"

                                def changeCount = sh(
                                    script: "cd ${LLM_ROOT} && git diff --name-only ${DURATION_FILE_PATH} | wc -l",
                                    returnStdout: true).trim()
                                echo "Changed duration-file count: ${changeCount}"
                                if (changeCount == "0") {
                                    echo "Duration file already up to date on ${params.TARGET_BRANCH}; nothing to push."
                                    return
                                }

                                sh """
                                    cd ${LLM_ROOT}
                                    git add ${DURATION_FILE_PATH}
                                    git commit -s -m "[None][infra] Auto-update test durations from OpenSearch (last ${params.DAYS} days)"
                                """

                                try {
                                    sh """
                                        cd ${LLM_ROOT}
                                        git push "https://svc_tensorrt:\${GITHUB_API_TOKEN}@github.com/${TARGET_REPO}.git" HEAD:${params.TARGET_BRANCH}
                                    """
                                    break  // push succeeded
                                } catch (Exception pushErr) {
                                    echo "Push attempt ${attempt} failed: ${pushErr.getMessage()}"
                                    if (attempt == maxRetries) { throw pushErr }
                                    // Roll back the local commit so the next retry can re-apply cleanly.
                                    sh "cd ${LLM_ROOT} && git reset HEAD~1"
                                }
                            }
                        }
                    }
                }
            }
        } // stage Commit and Push
    } // stages
} // pipeline
