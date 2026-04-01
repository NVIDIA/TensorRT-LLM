@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import java.lang.InterruptedException

import com.nvidia.bloom.SlurmConfig
import com.nvidia.bloom.SlurmCluster
import com.nvidia.bloom.SlurmPartition
import com.nvidia.bloom.CloudManager
import com.nvidia.bloom.Utils

DOCKER_IMAGE = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.10-py3-x86_64-ubuntu24.04-trt10.13.3.9-skip-tritondevel-202510291120-8621"

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}
LLM_ROOT = "llm"

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

def runPerfTriageBot() {
    // Resolve cluster and get login node
    SlurmPartition partition = SlurmConfig.resolvePlatform(params.CLUSTER)
    SlurmCluster cluster = SlurmConfig.clusterConfig[partition.clusterName]

    CloudManager.withSlurmSshCredentials(this, partition.clusterName, cluster) { remote ->

        // Install SSH tools on the K8s pod
        Utils.exec(this, script: "apt-get update && apt-get install -y sshpass openssh-client")

        // Create workspace on the login node
        def workspace = "/home/svc_tensorrt/bloom/agent-run/perf-triage-bot-${env.BUILD_TAG}"
        Utils.exec(this, script: Utils.sshUserCmd(remote, "\"mkdir -p ${workspace}\""), numRetries: 3)
        Utils.exec(this, script: Utils.sshUserCmd(remote,
            "\"cd ${workspace} && (rm -rf trtllm-perf-triage-bot; git clone https://gitlab-master.nvidia.com/chenfeiz/trtllm-perf-triage-bot.git)\""),
            numRetries: 3)
        withCredentials([usernamePassword(credentialsId: 'svc_tensorrt_gitlab_api_token', usernameVariable: 'GIT_USER', passwordVariable: 'GIT_TOKEN')]) {
            Utils.exec(this, script: Utils.sshUserCmd(remote,
                "\"cd ${workspace} && (rm -rf trtllm-agent-toolkit; git clone https://${GIT_USER}:${GIT_TOKEN}@gitlab-master.nvidia.com/ftp/trtllm-agent-toolkit.git)\""),
                numRetries: 3)
        }

        // Install anthropic Python SDK on the login node in a venv
        def venvDir = "${workspace}/.venv"
        def installScriptContent = """#!/bin/bash
python3 -m venv ${venvDir}
source ${venvDir}/bin/activate
pip install anthropic
"""
        def installScriptBase64 = installScriptContent.bytes.encodeBase64().toString()
        Utils.exec(this, script: Utils.sshUserCmd(remote,
            "\"echo '${installScriptBase64}' | base64 -d | bash\""),
            numRetries: 3)

        // Handle context restore from a previous job
        def contextFile = "${workspace}/conversation-context.json"
        def restoreContextArg = ""
        if (params.RESTORE_JOB_ID?.trim()) {
            def restoreJobId = params.RESTORE_JOB_ID.trim()
            if (!(restoreJobId ==~ /^\d+$/)) {
                error("RESTORE_JOB_ID must be a numeric build number, got: '${restoreJobId}'")
            }
            def jobNameNormalized = env.JOB_NAME.replaceAll('/', '-')
            def restoreBuildTag = "jenkins-${jobNameNormalized}-${restoreJobId}"
            def restoreWorkspace = "/home/svc_tensorrt/bloom/agent-run/perf-triage-bot-${restoreBuildTag}"
            def restoreContextFile = "${restoreWorkspace}/conversation-context.json"

            // Copy context file from previous job's workspace
            Utils.exec(this, script: Utils.sshUserCmd(remote,
                "\"cp ${restoreContextFile} ${contextFile}\""),
                numRetries: 3)
            restoreContextArg = "--restore-context ${contextFile}"
        }

        // Construct structured agent prompt from typed parameters
        def promptLines = [
            "Bug Type: ${params.BUG_TYPE}",
            "GPU & Cluster: ${params.CLUSTER}",
        ]
        if (params.PERF_METRIC?.trim()) {
            promptLines << "Metric: ${params.PERF_METRIC}"
        }
        promptLines << ""
        promptLines << "Bad commit: ${params.BAD_COMMIT}"
        if (params.BAD_BRANCH?.trim()) {
            promptLines << "  Branch: ${params.BAD_BRANCH}"
        }
        if (params.BAD_PERF_VALUE?.trim()) {
            promptLines << "  Reported perf: ${params.BAD_PERF_VALUE}"
        }
        if (params.GOOD_COMMIT?.trim()) {
            promptLines << "Good commit: ${params.GOOD_COMMIT}"
            if (params.GOOD_BRANCH?.trim()) {
                promptLines << "  Branch: ${params.GOOD_BRANCH}"
            }
            if (params.GOOD_PERF_VALUE?.trim()) {
                promptLines << "  Reported perf: ${params.GOOD_PERF_VALUE}"
            }
        }
        promptLines << ""
        promptLines << "Test name: ${params.TEST_NAME}"
        if (params.ADDITIONAL_CONTEXT?.trim()) {
            promptLines << ""
            promptLines << "Additional context: ${params.ADDITIONAL_CONTEXT}"
        }

        def fullPrompt = promptLines.join("\n")
        def escapedPrompt = fullPrompt.replace('"', '\\"').replace('$', '\\$').replace('`', '\\`')
        def authToken = env.ANTHROPIC_AUTH_TOKEN
        def botDir = "${workspace}/trtllm-perf-triage-bot"
        def skillsDir = "${botDir}/skills"
        def systemPromptFile = "${workspace}/system-prompt.txt"
        def runScriptContent = """#!/bin/bash
set -e -o pipefail
source ${venvDir}/bin/activate
cd ${botDir}

# Generate system prompt with dynamic skills listing
SKILLS_LIST=\$(ls -1 ${skillsDir}/*.md 2>/dev/null | xargs -I{} basename {} | sed 's/^/  - /' || echo "  (none)")
cat > ${systemPromptFile} <<'SYSPROMPT_HEADER'
You are an autonomous AI agent for triaging TRTLLM Perf or Functional Issue.
You have access to the following tools: bash, read_file, write_file, edit_file, glob_files, grep_search.

## Tool Usage Guidelines

- Use bash for shell commands (git, python, sbatch, squeue, etc.)
- Use read_file to read files instead of cat/head/tail
- Use write_file to create new files
- Use edit_file for find-and-replace modifications to existing files
- Use glob_files to find files by pattern instead of find/ls
- Use grep_search to search file contents instead of grep/rg
- You can call multiple tools in a single response
- Bash commands have a default timeout of 600 seconds

## Safety Rules

- You are sandboxed to the workspace directory. NEVER read, write, or execute files outside of it.
- NEVER run rm -rf on / or ~
- NEVER use sudo
- NEVER use absolute paths outside the workspace (e.g., /home/svc_tensorrt/.bashrc, /etc/*)
- When modifying files, read them first to understand the context
- For long-running SLURM jobs, use the bash tool to submit and poll status

## Important Notes

- The utils.py file in the triage bot repo provides helper functions for SLURM job management, git operations, performance evaluation, bisect algorithm, and artifactory acceleration
- Import and use functions from utils.py when appropriate
- Always check existing output directories before re-running expensive operations
- The trtllm-agent-toolkit repo is cloned in the same parent directory as trtllm-perf-triage-bot (i.e. ../trtllm-agent-toolkit relative to this repo)

## Skills Reference

SYSPROMPT_HEADER

echo "Working directory: \$(pwd)" >> ${systemPromptFile}
echo "" >> ${systemPromptFile}
echo "Workflow skills are located at: ${skillsDir}" >> ${systemPromptFile}
echo "Read the relevant skill files with read_file as needed. Available skills:" >> ${systemPromptFile}
echo "\${SKILLS_LIST}" >> ${systemPromptFile}
echo "" >> ${systemPromptFile}
echo "Start by reading PIPELINE.md to understand the overall workflow." >> ${systemPromptFile}

# Run claude_cli.py — do not let its exit code fail the pipeline.
# The agent may encounter errors (e.g. SLURM failures) that it should
# handle and retry autonomously. Even if it ultimately fails, the
# pipeline should still succeed so that context is preserved for
# follow-up runs.
set +e +o pipefail
ANTHROPIC_BASE_URL=https://inference-api.nvidia.com ANTHROPIC_AUTH_TOKEN=${authToken} python3 claude_cli.py -p "${escapedPrompt}" --system-prompt-file ${systemPromptFile} --allowed-paths ${workspace} --model aws/anthropic/bedrock-claude-opus-4-6 --max-turns 500 --save-context ${contextFile} ${restoreContextArg} 2>${workspace}/claude-stderr.log | tee ${workspace}/claude-output.log
CLAUDE_EXIT=\$?
set -e -o pipefail

if [ \$CLAUDE_EXIT -ne 0 ]; then
    echo "WARNING: claude_cli.py exited with code \$CLAUDE_EXIT (see claude-stderr.log for details)"
fi
"""
        def runScriptBase64 = runScriptContent.bytes.encodeBase64().toString()
        Utils.exec(this, timeout: false, script: Utils.sshUserCmd(remote,
            "\"echo '${runScriptBase64}' | base64 -d | bash\""),
            numRetries: 1)

        // Cleanup workspace
        // Utils.exec(this, script: Utils.sshUserCmd(remote, "\"rm -rf ${workspace}\""), numRetries: 3)
    }
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig(DOCKER_IMAGE)
    }
    options {
        timestamps()
    }
    environment {
        OPEN_SEARCH_DB_BASE_URL=credentials("open_search_db_base_url")
        OPEN_SEARCH_DB_CREDENTIALS=credentials("open_search_db_credentials")
        ANTHROPIC_AUTH_TOKEN=credentials("ANTHROPIC_AUTH_TOKEN")
    }
    parameters {
        string(name: "OPERATION", defaultValue: "TRTLLM PERF TRIAGE BOT", description: "Operation to perform.")
        string(name: "BRANCH", defaultValue: "main", description: "Branch to checkout. (Only used when OPERATION is SLACK BOT SENDS MESSAGE)")
        string(name: "OPEN_SEARCH_PROJECT_NAME", defaultValue: "swdl-trtllm-infra-ci-prod-perf_sanity_info", description: "OpenSearch project name. (Only used when OPERATION is SLACK BOT SENDS MESSAGE)")
        string(name: "QUERY_JOB_NUMBER", defaultValue: "1", description: "Number of latest jobs to query. (Only used when OPERATION is SLACK BOT SENDS MESSAGE)")
        string(name: "SLACK_CHANNEL_ID", defaultValue: "C0A7D0LCA1F", description: "Slack channel IDs to send messages to. (Only used when OPERATION is SLACK BOT SENDS MESSAGE)")
        string(name: "SLACK_BOT_TOKEN", defaultValue: "", description: "Slack bot token for authentication. (Only used when OPERATION is SLACK BOT SENDS MESSAGE)")
        choice(name: "CLUSTER", choices: ["auto:gb200-flex", "auto:dgx-b200-flex"], description: "Cluster to run perf triage bot on. (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        choice(name: "BUG_TYPE", choices: ["Perf Regression", "Perf Improvement", "Perf Instability", "Functional Failure"], description: "Type of bug. (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "PERF_METRIC", defaultValue: "", description: "Optional: the metric that regressed or fluctuates (e.g., output_token_throughput, total_token_throughput, e2e_latency, e2e_runtime). (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "TEST_NAME", defaultValue: "", description: "Required: CI perf sanity test name (e.g., k2_thinking_fp4_tep8_32k8k-con2_iter10_32k8k) or full pytest ID. (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "BAD_COMMIT", defaultValue: "", description: "Required: bad commit hash (e.g., def5678). (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "BAD_BRANCH", defaultValue: "", description: "Optional: branch for bad commit (e.g., main). (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "GOOD_COMMIT", defaultValue: "", description: "Optional: good commit hash (e.g., abc1234). (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "GOOD_BRANCH", defaultValue: "", description: "Optional: branch for good commit (e.g., main). (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "GOOD_PERF_VALUE", defaultValue: "", description: "Optional: good perf value (e.g., 1186.43). (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "BAD_PERF_VALUE", defaultValue: "", description: "Optional: bad perf value (e.g., 1085.13). (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        text(name: "ADDITIONAL_CONTEXT", defaultValue: "", description: "Optional: any extra context or notes for the agent. (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
        string(name: "RESTORE_JOB_ID", defaultValue: "", description: "Build number of a previous PerfSanityTriage job to restore conversation context from. Leave empty for a fresh session. (Only used when OPERATION is TRTLLM PERF TRIAGE BOT)")
    }
    stages {
        stage("Run Perf Triage Bot") {
            when { expression { params.OPERATION == "TRTLLM PERF TRIAGE BOT" } }
            steps {
                container("trt-llm") {
                    script {
                        sh "pwd && ls -alh"
                        timeout(time: 24, unit: 'HOURS') {
                            runPerfTriageBot()
                        }
                    }
                }
            }
        } // stage Run Perf Triage Bot
        stage("Run Perf Sanity Script") {
            when { expression { params.OPERATION == "SLACK BOT SENDS MESSAGE" } }
            steps {
                container("trt-llm") {
                    script {
                        sh "pwd && ls -alh"
                        sh "env | sort"
                        trtllm_utils.checkoutSource(LLM_REPO, params.BRANCH, LLM_ROOT, false, false)
                        sh "pip install slack_sdk"
                        sh """
                            cd ${LLM_ROOT}/jenkins/scripts/perf && ls -alh && python3 perf_sanity_triage.py \
                            --project_name "${params.OPEN_SEARCH_PROJECT_NAME}" \
                            --operation "${params.OPERATION}" \
                            --channel_id "${params.SLACK_CHANNEL_ID}" \
                            --bot_token "${params.SLACK_BOT_TOKEN}" \
                            --query_job_number "${params.QUERY_JOB_NUMBER}"
                        """
                    }
                }
            }
        } // stage Run Perf Sanity Script
    } // stages
} // pipeline
