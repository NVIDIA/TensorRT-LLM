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
LLM_REPO = "https://github.com/${TARGET_REPO}.git"

def createKubernetesPodConfig(image, arch = "amd64")
{
    def archSuffix = arch == "arm64" ? "arm" : "amd"
    def jnlpImage = "urm.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_${archSuffix}_linux:jdk17"

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
                        trtllm_utils.checkoutSource(sourceRepo, params.TARGET_BRANCH, LLM_ROOT, false, false)
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

                    // Sanity gate: if the new file diverges too much from the one in use,
                    // fail the job instead of committing a possibly-broken duration file.
                    script {
                        def countItems = { path ->
                            sh(script: "python3 -c \"import json; print(len(json.load(open('${path}'))))\"",
                               returnStdout: true).trim() as Integer
                        }
                        def oldCount = countItems("${LLM_ROOT}/${DURATION_FILE_PATH}")
                        def newCount = countItems("${LLM_ROOT}/new_test_durations.json")
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
                        // Overwrite the checked-in duration file with the freshly generated one.
                        sh """
                            cd ${LLM_ROOT}
                            git config --global --add safe.directory \$(pwd)
                            git config user.email "90828364+tensorrt-cicd@users.noreply.github.com"
                            git config user.name "TensorRT LLM"
                            cp new_test_durations.json ${DURATION_FILE_PATH}
                        """

                        def changeCount = sh(
                            script: "cd ${LLM_ROOT} && git status --porcelain ${DURATION_FILE_PATH} | wc -l",
                            returnStdout: true).trim()
                        echo "Changed duration-file count: ${changeCount}"
                        if (changeCount == "0") {
                            echo "No update to the duration file; nothing to commit."
                            return
                        }

                        sh """
                            cd ${LLM_ROOT}
                            git add ${DURATION_FILE_PATH}
                            git commit -s -m "[None][infra] Auto-update test durations from OpenSearch (last ${params.DAYS} days)"
                        """

                        withCredentials([usernamePassword(
                            credentialsId: 'github-cred-trtllm-ci',
                            usernameVariable: 'NOT_IN_USE',
                            passwordVariable: 'GITHUB_API_TOKEN')]) {
                            def authedUrl = LLM_REPO.replaceFirst(
                                'https://', "https://svc_tensorrt:${GITHUB_API_TOKEN}@")
                            // Rebase onto the latest target branch before pushing to avoid
                            // clobbering commits landed since checkout.
                            sh """
                                cd ${LLM_ROOT}
                                git remote set-url origin ${authedUrl}
                                git fetch origin ${params.TARGET_BRANCH}
                                git rebase origin/${params.TARGET_BRANCH}
                                git push origin HEAD:${params.TARGET_BRANCH}
                            """
                        }
                    }
                }
            }
        } // stage Commit and Push
    } // stages
} // pipeline
