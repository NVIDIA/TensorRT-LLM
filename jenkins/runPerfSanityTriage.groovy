@Library(['trtllm-jenkins-shared-lib@main']) _

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
    }
    parameters {
        choice(name: "OPERATION", choices: ["Update Perf Data"], description: "Operation to perform.")
        string(name: "BRANCH", defaultValue: "main", description: "Branch to checkout.")
        choice(name: "OPEN_SEARCH_PROJECT_NAME", choices: ["swdl-trtllm-infra-ci-prod-perf_sanity_info"], description: "OpenSearch project name.")
        text(name: "COMMANDS", defaultValue: "", description: "UPDATE commands, one per line. Example: UPDATE SET field=value WHERE condition=value.")
    }
    stages {
        stage("Update Perf Data") {
            when { expression { params.OPERATION == "Update Perf Data" } }
            steps {
                container("trt-llm") {
                    script {
                        sh "pwd && ls -alh"
                        trtllm_utils.checkoutSource(LLM_REPO, params.BRANCH, LLM_ROOT, false, false)
                        def commandsBase64 = params.COMMANDS.bytes.encodeBase64().toString()
                        sh """
                            cd ${LLM_ROOT}/jenkins/scripts/perf && python3 perf_sanity_triage.py \
                            --project_name "${params.OPEN_SEARCH_PROJECT_NAME}" \
                            --commands "\$(echo '${commandsBase64}' | base64 -d)"
                        """
                    }
                }
            }
        } // stage Update Perf Data
    } // stages
} // pipeline
