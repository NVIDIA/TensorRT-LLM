
import java.lang.InterruptedException

DOCKER_IMAGE = "urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm:pytorch-25.03-py3-x86_64-ubuntu24.04-trt10.9.0.34-skip-devel-202504101610-3421"

def createKubernetesPodConfig(image)
{
    def podConfig = [
        cloud: "kubernetes-cpu",
        namespace: "sw-tensorrt",
        yaml: """
            apiVersion: v1
            kind: Pod
            spec:
                qosClass: Guaranteed
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
                    - name: sw-tensorrt-blossom
                      mountPath: "/mnt/sw-tensorrt-blossom"
                      readOnly: false
                    tty: true
                    resources:
                      requests:
                        cpu: 2
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: 2
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                    imagePullPolicy: Always
                  - name: jnlp
                    image: urm.nvidia.com/docker/jenkins/inbound-agent:4.11-1-jdk11
                    args: ['\$(JENKINS_SECRET)', '\$(JENKINS_NAME)']
                    resources:
                      requests:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                qosClass: Guaranteed
                volumes:
                - name: sw-tensorrt-pvc
                  persistentVolumeClaim:
                    claimName: sw-tensorrt-pvc
                - name: sw-tensorrt-blossom
                  nfs:
                    server: 10.117.145.13
                    path: /vol/scratch1/scratch.svc_tensorrt_blossom

        """.stripIndent(),
    ]

    return podConfig
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig(DOCKER_IMAGE)
    }
    options {
        // to better analyze the time for each step/test
        timestamps()
    }
    parameters {
        choice(name: "OPERATION", choices: ["Stats", "NormalClean", "NuclearClean", "Reset", "Config"], description: "Cache operation.")
        choice(name: "BASE", choices: ["NFS", "PVC"], description: "Base directory.")
    }
    stages {
        stage("Check or reset CCACHE") {
            steps {
                container("trt-llm") {
                    script {
                        // Random sleep to avoid resource contention
                        sleep(10 * Math.random())

                        switch(params.BASE) {
                          case "NFS":
                            BASE_DIR="/mnt/sw-tensorrt-blossom"
                            break
                          case "PVC":
                            BASE_DIR="/mnt/sw-tensorrt-pvc"
                            break
                          default:
                            assert false, "Unknown operation"
                            break
                        }

                        withEnv(["CCACHE_DIR=${BASE_DIR}/scratch.trt_ccache/llm_ccache"]) {
                            sh "nproc && free -g && hostname"
                            sh "env | sort"
                            sh "pwd && ls -alh"
                            sh "timeout 30 mount"
                            sh "ccache -sv"
                            sh "df -h"

                            switch(params.OPERATION) {
                              case "NormalClean":
                                sh "ccache -cz"
                                break
                              case "NuclearClean":
                                sh "ccache -Cz"
                                break
                              case "Reset":
                                sh "rm -rf ${CCACHE_DIR}"
                                sh "mkdir -p ${CCACHE_DIR}"
                                sh "printf \"max_size=300G\ntemporary_dir=/tmp/ccache\ncompression = true\n\" > ${CCACHE_DIR}/ccache.conf"
                                break
                              case "Config":
                                sh "printf \"max_size=300G\ntemporary_dir=/tmp/ccache\ncompression = true\n\" > ${CCACHE_DIR}/ccache.conf"
                                break
                              case "Stats":
                                sh "ccache -sv"
                                sh "cat ${CCACHE_DIR}/ccache.conf"
                                sh "ls -alh ${BASE_DIR}"
                                sh "ls -alh ${BASE_DIR}/scratch.trt_ccache"
                                sh "ls -alh ${BASE_DIR}/scratch.trt_ccache/llm_ccache"
                                break
                              default:
                                assert false, "Unknown operation"
                                break
                            }

                            sh "ccache -sv"
                            sh "df -h"
                        }
                    }
                }
            }
        } // stage Check or reset CCACHE
    } // stages
} // pipeline
