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

@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@create_PR']) _

// LLM repository URL
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ?: DEFAULT_LLM_REPO
}
LLM_ROOT = "llm"

UBUNTU_24_04_IMAGE = "urm.nvidia.com/docker/ubuntu:24.04"
DURATION_FILE_PATH = "tests/integration/defs/.test_durations"
AUTO_UPDATE_BRANCH = "auto/update_test_durations"
TARGET_REPO = "NVIDIA/TensorRT-LLM"

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
            defaultValue: '3',
            description: 'Number of days to look back in OpenSearch for test durations (e.g. 3, 5, 7).')
        string(
            name: 'TARGET_BRANCH',
            defaultValue: 'main',
            description: 'Base branch for the auto-update PR.')
        booleanParam(
            name: 'DRY_RUN',
            defaultValue: false,
            description: 'When true, generate the duration file but skip PR creation.')
    }
    environment {
        OPEN_SEARCH_DB_BASE_URL = credentials('open_search_db_base_url')
        OPEN_SEARCH_DB_CREDENTIALS = credentials('open_search_db_credentials')
    }
    stages {
        stage('Setup') {
            steps {
                container('trt-llm') {
                    sh '''
                        apt-get update -qq
                        apt-get install -y -qq git python3-pip curl
                        pip3 install --quiet requests
                    '''
                }
            }
        } // stage Setup

        stage('Checkout') {
            steps {
                container('trt-llm') {
                    script {
                        trtllm_utils.checkoutSource(LLM_REPO, params.TARGET_BRANCH, LLM_ROOT, false, false)
                    }
                }
            }
        } // stage Checkout

        stage('Generate Duration File') {
            steps {
                container('trt-llm') {
                    sh """
                        cd ${LLM_ROOT}
                        python3 scripts/generate_duration.py \
                            --from-opensearch \
                            --days ${params.DAYS} \
                            --opensearch-url "${OPEN_SEARCH_DB_BASE_URL}" \
                            --opensearch-credentials "${OPEN_SEARCH_DB_CREDENTIALS}" \
                            --duration-file new_test_durations.json
                        echo "Generated file size: \$(wc -l < new_test_durations.json) lines"
                        echo "Sample output (first 5 lines):"
                        head -5 new_test_durations.json
                    """
                }
            }
        } // stage Generate Duration File

        stage('Create PR') {
            when {
                expression { !params.DRY_RUN }
            }
            steps {
                container('trt-llm') {
                    script {
                        def prUrl = trtllm_utils.createAutoUpdatePR(this, [
                            repoDir:       LLM_ROOT,
                            srcFile:       "${LLM_ROOT}/new_test_durations.json",
                            dstFile:       DURATION_FILE_PATH,
                            branchName:    AUTO_UPDATE_BRANCH,
                            baseBranch:    params.TARGET_BRANCH,
                            forcePush:     true,
                            commitMessage: "[None][infra] Auto-update test durations from OpenSearch (last ${params.DAYS} days)",
                            prTitle:       "[None][infra] Auto-update test durations from OpenSearch",
                            prBody:        "Automatically generated by the UpdateTestDurations Jenkins job.\n\nQueries OpenSearch for the last ${params.DAYS} days of PASSED test results and computes average duration per test. Updates the pytest-split duration file used for time-based test sharding.",
                            prComment:     "/bot run",
                            autoMerge:     true,
                        ])
                        if (prUrl) {
                            echo "PR: ${prUrl}"
                        }
                    }
                }
            }
        } // stage Create PR
    } // stages
} // pipeline
