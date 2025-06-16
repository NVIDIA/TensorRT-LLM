@Library(['bloom-jenkins-shared-lib@main', 'trtllm-jenkins-shared-lib@main']) _

import groovy.transform.Field

// LLM repository configuration
withCredentials([string(credentialsId: 'default-llm-repo', variable: 'DEFAULT_LLM_REPO')]) {
    LLM_REPO = env.gitlabSourceRepoHttpUrl ? env.gitlabSourceRepoHttpUrl : "${DEFAULT_LLM_REPO}"
}
LLM_ROOT = "llm"

ARTIFACT_PATH = env.artifactPath ? env.artifactPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"
UPLOAD_PATH = env.uploadPath ? env.uploadPath : "sw-tensorrt-generic/llm-artifacts/${JOB_NAME}/${BUILD_NUMBER}"

X86_64_TRIPLE = "x86_64-linux-gnu"
AARCH64_TRIPLE = "aarch64-linux-gnu"

LLM_DOCKER_IMAGE = env.dockerImage

AGENT_IMAGE = env.dockerImage

POD_TIMEOUT_SECONDS = env.podTimeoutSeconds ? env.podTimeoutSeconds : "21600"

// Literals for easier access.
@Field
def WHEEL_EXTRA_ARGS = "extraArgs"

@Field
def TARNAME = "tarName"

@Field
def WHEEL_ARCHS = "wheelArchs"

@Field
def CONFIG_LINUX_X86_64_VANILLA = "linux_x86_64_Vanilla"

@Field
def CONFIG_LINUX_X86_64_SINGLE_DEVICE = "linux_x86_64_SingleDevice"

@Field
def CONFIG_LINUX_X86_64_LLVM = "linux_x86_64_LLVM"

@Field
CONFIG_LINUX_AARCH64 = "linux_aarch64"

@Field
def CONFIG_LINUX_AARCH64_LLVM = "linux_aarch64_LLVM"

@Field
def BUILD_CONFIGS = [
  // Vanilla TARNAME is used for packaging in runLLMPackage
  // cmake-vars cannot be empty, so passing (default) multi-device configuration.
  (CONFIG_LINUX_X86_64_VANILLA) : [
    (WHEEL_EXTRA_ARGS) : "--extra-cmake-vars ENABLE_MULTI_DEVICE=1 --extra-cmake-vars WARNING_IS_ERROR=ON --extra-cmake-vars NIXL_ROOT=/opt/nvidia/nvda_nixl --micro_benchmarks",
    (TARNAME) : "TensorRT-LLM.tar.gz",
    (WHEEL_ARCHS): "80-real;86-real;89-real;90-real;100-real;120-real",
  ],
  (CONFIG_LINUX_X86_64_SINGLE_DEVICE) : [
    (WHEEL_EXTRA_ARGS) : "--extra-cmake-vars ENABLE_MULTI_DEVICE=0 --extra-cmake-vars WARNING_IS_ERROR=ON --extra-cmake-vars ENABLE_UCX=0 --micro_benchmarks",
    (TARNAME) : "single-device-TensorRT-LLM.tar.gz",
    (WHEEL_ARCHS): "80-real;86-real;89-real;90-real;100-real;120-real",
  ],
  (CONFIG_LINUX_X86_64_LLVM) : [
    (WHEEL_EXTRA_ARGS) : "--extra-cmake-vars ENABLE_MULTI_DEVICE=1 --extra-cmake-vars WARNING_IS_ERROR=ON --micro_benchmarks -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=clang -DCMAKE_LINKER_TYPE=LLD",
    (TARNAME) : "llvm-TensorRT-LLM.tar.gz",
    (WHEEL_ARCHS): "80-real;86-real;89-real;90-real;100-real;120-real",
  ],
  (CONFIG_LINUX_AARCH64): [
    (WHEEL_EXTRA_ARGS) : "--extra-cmake-vars WARNING_IS_ERROR=ON",
    (TARNAME) : "TensorRT-LLM-GH200.tar.gz",
    (WHEEL_ARCHS): "90-real;100-real;120-real",
  ],
  (CONFIG_LINUX_AARCH64_LLVM) : [
    (WHEEL_EXTRA_ARGS) : "--extra-cmake-vars WARNING_IS_ERROR=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CUDA_HOST_COMPILER=clang -DCMAKE_LINKER_TYPE=LLD",
    (TARNAME) : "llvm-TensorRT-LLM-GH200.tar.gz",
    (WHEEL_ARCHS): "90-real;100-real;120-real",
  ],
]

@Field
def GITHUB_PR_API_URL = "github_pr_api_url"
@Field
def CACHED_CHANGED_FILE_LIST = "cached_changed_file_list"
@Field
def ACTION_INFO = "action_info"
def globalVars = [
    (GITHUB_PR_API_URL): null,
    (CACHED_CHANGED_FILE_LIST): null,
    (ACTION_INFO): null,
]

// TODO: Move common variables to an unified location
BUILD_CORES_REQUEST = "8"
BUILD_CORES_LIMIT = "8"
BUILD_MEMORY_REQUEST = "48Gi"
BUILD_MEMORY_LIMIT = "64Gi"
BUILD_JOBS = "8"

TESTER_CORES = "12"
TESTER_MEMORY = "96Gi"

CCACHE_DIR="/mnt/sw-tensorrt-pvc/scratch.trt_ccache/llm_ccache"

String getShortenedJobName(String path)
{
    static final nameMapping = [
        "L0_MergeRequest": "l0-mr",
        "L0_Custom": "l0-cus",
        "L0_PostMerge": "l0-pm",
        "L0_PostMergeDocker": "l0-pmd",
        "L1_Custom": "l1-cus",
        "L1_Nightly": "l1-nt",
        "L1_Stable": "l1-stb",
    ]
    def parts = path.split('/')
    // Apply nameMapping to the last part (jobName)
    def jobName = parts[-1]
    boolean replaced = false
    nameMapping.each { key, value ->
        if (jobName.contains(key)) {
            jobName = jobName.replace(key, value)
            replaced = true
        }
    }
    if (!replaced) {
        jobName = jobName.length() > 7 ? jobName.substring(0, 7) : jobName
    }
    // Replace the last part with the transformed jobName
    parts[-1] = jobName
    // Rejoin the parts with '-', convert to lowercase
    return parts.join('-').toLowerCase()
}

def createKubernetesPodConfig(image, type, arch = "amd64")
{
    def targetCould = "kubernetes-cpu"
    def selectors = """
                  nvidia.com/node_type: builder
                  kubernetes.io/os: linux
                  kubernetes.io/arch: ${arch}"""
    def containerConfig = ""
    def nodeLabelPrefix = ""
    def jobName = getShortenedJobName(env.JOB_NAME)
    def buildID = env.BUILD_ID

    def archSuffix = arch == "arm64" ? "arm" : "amd"
    def jnlpImage = "urm.nvidia.com/sw-ipp-blossom-sre-docker-local/lambda/custom_jnlp_images_${archSuffix}_linux:jdk17"

    switch(type)
    {
    case "build":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['sleep', ${POD_TIMEOUT_SECONDS}]
                    volumeMounts:
                    - name: sw-tensorrt-pvc
                      mountPath: "/mnt/sw-tensorrt-pvc"
                      readOnly: false
                    tty: true
                    resources:
                      requests:
                        cpu: ${BUILD_CORES_REQUEST}
                        memory: ${BUILD_MEMORY_REQUEST}
                        ephemeral-storage: 200Gi
                      limits:
                        cpu: ${BUILD_CORES_LIMIT}
                        memory: ${BUILD_MEMORY_LIMIT}
                        ephemeral-storage: 200Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    case "package":
        containerConfig = """
                  - name: trt-llm
                    image: ${image}
                    command: ['cat']
                    tty: true
                    resources:
                      requests:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                    imagePullPolicy: Always"""
        nodeLabelPrefix = "cpu"
        break
    }
    def nodeLabel = trtllm_utils.appendRandomPostfix("${nodeLabelPrefix}---tensorrt-${jobName}-${buildID}")
    def pvcVolume = """
                - name: sw-tensorrt-pvc
                  persistentVolumeClaim:
                    claimName: sw-tensorrt-pvc
    """
    if (arch == "arm64") {
        // PVC mount isn't supported on aarch64 platform. Use NFS as a WAR.
        pvcVolume = """
                - name: sw-tensorrt-pvc
                  nfs:
                    server: 10.117.145.13
                    path: /vol/scratch1/scratch.svc_tensorrt_blossom
        """
    }
    def podConfig = [
        cloud: targetCould,
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
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                      limits:
                        cpu: '2'
                        memory: 10Gi
                        ephemeral-storage: 25Gi
                qosClass: Guaranteed
                volumes:
                ${pvcVolume}
        """.stripIndent(),
    ]

    return podConfig
}

def echoNodeAndGpuInfo(pipeline, stageName)
{
    String hostNodeName = sh(script: 'echo $HOST_NODE_NAME', returnStdout: true)
    String gpuUuids = pipeline.sh(script: "nvidia-smi -q | grep \"GPU UUID\" | awk '{print \$4}' | tr '\n' ',' || true", returnStdout: true)
    pipeline.echo "HOST_NODE_NAME = ${hostNodeName} ; GPU_UUIDS = ${gpuUuids} ; STAGE_NAME = ${stageName}"
}

def downloadArtifacts(stageName, reuseArtifactPath, artifacts, serverId = 'Artifactory')
{
    def reused = true
    stage(stageName) {
        for (downit in artifacts) {
            def uploadpath = downit.key
            try {
                rtDownload(
                    failNoOp: true,
                    serverId: serverId,
                    spec: """{
                        "files": [
                            {
                            "pattern": "${reuseArtifactPath}/${uploadpath}"
                            }
                        ]
                    }""",
                )
            } catch (Exception e) {
                echo "failed downloading ${reuseArtifactPath}/${uploadpath}, need rebuild."
                reused = false
                catchError(buildResult: 'SUCCESS', stageResult: 'UNSTABLE') { throw e }
            }
        }

        if (!reused) {
            return null
        }

        reuseArtifactPath = reuseArtifactPath.substring(reuseArtifactPath.indexOf('/')+1)
        def newArtifacts = [:]
        for (reuseit in artifacts) {
            def uploadpath = reuseit.key
            newArtifacts[reuseit.key] = "${reuseArtifactPath}/${uploadpath}"
        }

        return newArtifacts
    }
}

def uploadArtifacts(artifacts, prefix = UPLOAD_PATH, retryTimes = 2, serverId = 'Artifactory')
{
    for (it in artifacts) {
        def uploadpath = it.key
        def filepath = it.value
        echo "uploading ${filepath} as ${uploadpath}"
        trtllm_utils.llmRetry(retryTimes, "uploadArtifacts", {
            rtUpload (
                serverId: serverId,
                spec: """{
                    "files": [
                        {
                        "pattern": "${filepath}",
                        "target": "${prefix}/${uploadpath}"
                        }
                    ]
                }""",
            )
        })
    }
}

def buildOrCache(pipeline, key, reuseArtifactPath, artifacts, image, k8s_cpu, runner)
{
    if (reuseArtifactPath) {
        stage(key) {
            def newArtifacts = downloadArtifacts("[${key}] Reuse", reuseArtifactPath, artifacts)
            if (newArtifacts != null) {
                uploadArtifacts(newArtifacts)
            } else {
                reuseArtifactPath = null
            }
        }
    }
    if (reuseArtifactPath) {
        return
    }

    trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(image, "build", k8s_cpu), "trt-llm", {
        stage(key) {
            stage("[${key}] Run") {
                echoNodeAndGpuInfo(pipeline, key)
                runner()
            }
            stage("Upload") {
                rtServer (
                    id: 'Artifactory',
                    url: 'https://urm.nvidia.com/artifactory',
                    credentialsId: 'urm-artifactory-creds',
                    // If Jenkins is configured to use an http proxy, you can bypass the proxy when using this Artifactory server:
                    bypassProxy: true,
                    // Configure the connection timeout (in seconds).
                    // The default value (if not configured) is 300 seconds:
                    timeout: 300
                )
                uploadArtifacts(artifacts)
            }
        }
    })
}

def prepareLLMBuild(pipeline, config)
{
    def buildFlags = BUILD_CONFIGS[config]
    def tarName = buildFlags[TARNAME]

    def is_linux_x86_64 = config.contains("linux_x86_64")
    def artifacts = ["${tarName}": tarName]
    def runner = {
        runLLMBuild(pipeline, buildFlags, tarName, is_linux_x86_64)
    }

    return [artifacts, runner]

}

def runLLMBuild(pipeline, buildFlags, tarName, is_linux_x86_64)
{
    // Step 1: cloning tekit source code
    sh "pwd && ls -alh"
    sh "env | sort"
    sh "ccache -sv"
    sh "rm -rf **/*.xml *.tar.gz"

    trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
    if (env.alternativeTRT) {
        sh "cd ${LLM_ROOT} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
    }

    sh "mkdir TensorRT-LLM"
    sh "cp -r ${LLM_ROOT}/ TensorRT-LLM/src/"

    // Step 2: building wheels in container
    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"
    sh "cat ${CCACHE_DIR}/ccache.conf"

    sh "env | sort"
    sh "ldconfig --print-cache || true"
    sh "ls -lh /"
    sh "id || true"
    sh "whoami || true"
    echo "Building TensorRT-LLM Python package ..."
    sh "git config --global --add safe.directory \"*\""
    def pipArgs = "--no-cache-dir"
    if (is_linux_x86_64) {
        pipArgs = ""
    }

    // install python package
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${LLM_ROOT} && pip3 install -r requirements-dev.txt ${pipArgs}")

    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, "cp312")
    }

    withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'CONAN_LOGIN_USERNAME', passwordVariable: 'CONAN_PASSWORD')]) {
        sh "cd ${LLM_ROOT} && python3 scripts/build_wheel.py --use_ccache -j ${BUILD_JOBS} -a '${buildFlags[WHEEL_ARCHS]}' ${buildFlags[WHEEL_EXTRA_ARGS]} --benchmarks"
    }
    if (is_linux_x86_64) {
        sh "cd ${LLM_ROOT} && python3 scripts/build_cpp_examples.py"
    }

    // Build tritonserver artifacts
    def llmPath = sh (script: "realpath ${LLM_ROOT}",returnStdout: true).trim()
    sh "cd ${LLM_ROOT}/triton_backend/inflight_batcher_llm && mkdir build && cd build && cmake .. -DTRTLLM_DIR=${llmPath} -DUSE_CXX11_ABI=ON && make -j${BUILD_JOBS} install"

    // Step 3: packaging wheels into tarfile
    sh "cp ${LLM_ROOT}/build/tensorrt_llm-*.whl TensorRT-LLM/"

    // Step 4: packaging tritonserver artifacts into tarfile
    sh "mkdir -p TensorRT-LLM/triton_backend/inflight_batcher_llm/"
    sh "cp ${LLM_ROOT}/triton_backend/inflight_batcher_llm/build/libtriton_tensorrtllm.so TensorRT-LLM/triton_backend/inflight_batcher_llm/"
    sh "cp ${LLM_ROOT}/triton_backend/inflight_batcher_llm/build/trtllmExecutorWorker TensorRT-LLM/triton_backend/inflight_batcher_llm/"

    // Step 5: packaging benchmark and required cpp dependencies into tarfile
    sh "mkdir -p TensorRT-LLM/benchmarks/cpp"
    sh "cp ${LLM_ROOT}/cpp/build/benchmarks/bertBenchmark TensorRT-LLM/benchmarks/cpp"
    sh "cp ${LLM_ROOT}/cpp/build/benchmarks/gptManagerBenchmark TensorRT-LLM/benchmarks/cpp"
    sh "cp ${LLM_ROOT}/cpp/build/tensorrt_llm/libtensorrt_llm.so TensorRT-LLM/benchmarks/cpp"
    sh "cp ${LLM_ROOT}/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so TensorRT-LLM/benchmarks/cpp"

    if (is_linux_x86_64) {
        sh "rm -rf ${tarName}"
        sh "pigz --version || true"
        sh "bash -c 'tar --use-compress-program=\"pigz -k\" -cf ${tarName} TensorRT-LLM/'"
    } else {
        sh "tar -czvf ${tarName} TensorRT-LLM/"
    }
}

def buildWheelInContainer(pipeline, libraries=[], triple=X86_64_TRIPLE, clean=false, pre_cxx11abi=false, cpver="312", extra_args="")
{
    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"
    sh "ccache -sv"
    sh "cat ${CCACHE_DIR}/ccache.conf"

    // Step 1: cloning tekit source code
    trtllm_utils.checkoutSource(LLM_REPO, env.gitlabCommit, LLM_ROOT, true, true)
    if (env.alternativeTRT) {
        trtllm_utils.replaceWithAlternativeTRT(env.alternativeTRT, cpver)
        sh "cd ${LLM_ROOT} && sed -i 's#tensorrt~=.*\$#tensorrt#g' requirements.txt && cat requirements.txt"
    }
    // Step 2: building libs in container
    sh "bash -c 'pip3 show tensorrt || true'"

    if (extra_args == "") {
        if (triple == AARCH64_TRIPLE) {
            extra_args = "-a '90-real;100-real;120-real'"
        } else {
            extra_args = "-a '80-real;86-real;89-real;90-real;100-real;120-real'"
        }
    }
    if (pre_cxx11abi) {
        extra_args = extra_args + " -l -D 'USE_CXX11_ABI=0'"
    } else {
        if (libraries.size() != 0) {
            extra_args = extra_args + " -l -D 'USE_CXX11_ABI=1'"
        }
    }
    if (clean) {
        extra_args = extra_args + " --clean"
    }
    sh "bash -c 'git config --global --add safe.directory \"*\"'"
    // Because different architectures involve different macros, a comprehensive test is conducted here.
    withCredentials([usernamePassword(credentialsId: "urm-artifactory-creds", usernameVariable: 'CONAN_LOGIN_USERNAME', passwordVariable: 'CONAN_PASSWORD')]) {
        trtllm_utils.llmExecStepWithRetry(pipeline, script: "bash -c \"cd ${LLM_ROOT} && python3 scripts/build_wheel.py --use_ccache -j ${BUILD_JOBS} -D 'WARNING_IS_ERROR=ON' ${extra_args}\"")
    }
}

def prepareBuildLib(pipeline, triple, pre_cxx11abi)
{
    def libraries = [
        "batch_manager",
        "executor",
        "internal_cutlass_kernels",
    ]
    if ((triple == X86_64_TRIPLE && pre_cxx11abi) || (triple == AARCH64_TRIPLE && !pre_cxx11abi)) {
        libraries += [
            "ucx_wrapper",
        ]
    }

    def artifacts = [:]
    for (library_name in libraries) {
        def libdir
        def is_static
        if (library_name == "batch_manager") {
            libdir = "tensorrt_llm/batch_manager"
            is_static = true
        } else if (library_name == "executor") {
            libdir = "tensorrt_llm/executor"
            is_static = true
        } else if (library_name == "internal_cutlass_kernels"){
            libdir = "tensorrt_llm/kernels/internal_cutlass_kernels"
            is_static = true
        } else if (library_name == "ucx_wrapper") {
            libdir = "tensorrt_llm/executor/cache_transmission/ucx_utils"
        }

        def libname = "libtensorrt_llm_" + library_name
        def ext = ".so"
        if (is_static) {
            libname += "_static"
            ext = ".a"
        }
        def filepath = "${LLM_ROOT}/cpp/build/" + libdir + "/" + libname + ext
        def uploadname = libname + ext
        if (is_static && pre_cxx11abi) {
            uploadname = libname + ".pre_cxx11" + ext
        }
        def uploadpath = "${triple}/${uploadname}"
        artifacts[uploadpath] = filepath
    }

    def cpver = "cp312"
    if (triple == X86_64_TRIPLE) {
        cpver = "cp310"
    }

    return [artifacts, {
        buildWheelInContainer(pipeline, libraries, triple, false, pre_cxx11abi, cpver)
    }]
}

def prepareLLMPackage(pipeline, archTriple=X86_64_TRIPLE)
{
    def tarFileName = "TensorRT-LLM.tar.gz"
    def linuxPkgName = "tensorrt-llm-release-src-${env.gitlabCommit}.tar.gz"
    if (archTriple == AARCH64_TRIPLE) {
        tarFileName = "TensorRT-LLM-GH200.tar.gz"
        linuxPkgName = "tensorrt-llm-sbsa-release-src-${env.gitlabCommit}.tar.gz"
    }
    def artifacts = ["${linuxPkgName}": "${LLM_ROOT}/${linuxPkgName}"]
    return [artifacts, { runLLMPackage(pipeline, archTriple, tarFileName, linuxPkgName) }]
}

def runLLMPackage(pipeline, archTriple, tarFileName, linuxPkgName)
{
    // Random sleep to avoid resource contention
    sleep(10 * Math.random())
    sh "curl ifconfig.me || true"
    sh "nproc && free -g && hostname"

    // Step 1: create LLM_ROOT dir and download code
    sh "pwd && ls -alh"
    sh "mkdir ${LLM_ROOT}"
    def llmPath = sh (script: "realpath ${LLM_ROOT}",returnStdout: true).trim()

    // Download tar generated from build jobs
    def llmTarfile = "https://urm.nvidia.com/artifactory/${ARTIFACT_PATH}/${tarFileName}"
    trtllm_utils.llmExecStepWithRetry(pipeline, script: "cd ${llmPath} && wget -nv ${llmTarfile}")
    // The path TensorRT-LLM/src is defined in the build job
    sh "cd ${llmPath} && tar -zxf ${tarFileName} TensorRT-LLM/src"
    // create a additional `pkg/tensorrt_llm` folder to make sure the generated tar.gz has only one tensorrt_llm folder
    def llmPackage = "${llmPath}/TensorRT-LLM/pkg/"
    sh "rm -rf ${llmPackage}"
    sh "mkdir -p ${llmPackage}"
    sh "mv ${llmPath}/TensorRT-LLM/src ${llmPackage}/tensorrt_llm"

    // download libs
    trtllm_utils.llmExecStepWithRetry(pipeline, script: """
        pip3 install gitignore_parser && \
        python3 ${llmPackage}/tensorrt_llm/scripts/package_trt_llm.py \
        --lib_list oss \
        --arch ${archTriple} \
        --download ${env.gitlabCommit} \
        --addr https://urm.nvidia.com/artifactory/${ARTIFACT_PATH} \
        -v \
        ${llmPackage}/tensorrt_llm
    """)

    // clean the internal files and create one tar package
    sh """cd ${llmPackage}/tensorrt_llm && \
        python3 ${llmPackage}/tensorrt_llm/scripts/package_trt_llm.py \
        --lib_list oss \
        --clean \
        --package ${llmPath}/${linuxPkgName} \
        ${llmPackage}/tensorrt_llm
    """

    sh "cd ${llmPath} && ls -alh"
}

def launchStages(pipeline, cpu_arch, enableFailFast, globalVars)
{
    stage("Show Environment") {
        sh "env | sort"
        echo "dockerImage: ${env.dockerImage}"
        echo "gitlabSourceRepoHttpUrl: ${env.gitlabSourceRepoHttpUrl}"
        echo "gitlabCommit: ${env.gitlabCommit}"
        echo "alternativeTRT: ${env.alternativeTRT}"
        echo "Using GitLab repo: ${LLM_REPO}. Commit: ${env.gitlabCommit}"

        echo "env.globalVars is: ${env.globalVars}"
        globalVars = trtllm_utils.updateMapWithJson(pipeline, globalVars, env.globalVars, "globalVars")
        globalVars[ACTION_INFO] = trtllm_utils.setupPipelineDescription(pipeline, globalVars[ACTION_INFO])
    }

    def wheelDockerImage = env.wheelDockerImagePy310
    if (!wheelDockerImage && cpu_arch == AARCH64_TRIPLE) {
        wheelDockerImage = env.dockerImage
    }

    buildConfigs = [
        "Build TRT-LLM": [LLM_DOCKER_IMAGE] + prepareLLMBuild(
            pipeline, cpu_arch == AARCH64_TRIPLE ? CONFIG_LINUX_AARCH64 : CONFIG_LINUX_X86_64_VANILLA),
        "Build TRT-LLM LLVM": [LLM_DOCKER_IMAGE] + prepareLLMBuild(
            pipeline, cpu_arch == AARCH64_TRIPLE ? CONFIG_LINUX_AARCH64_LLVM : CONFIG_LINUX_X86_64_LLVM),
    ] + [true, false].collectEntries{ cxx11 -> [
        "Build libs (cxx11=${cxx11})".toString(), [wheelDockerImage] + prepareBuildLib(
            pipeline, cpu_arch, !cxx11),
    ]}

    if (cpu_arch == X86_64_TRIPLE) {
        buildConfigs += [
        "Build TRT-LLM SingleDevice": [LLM_DOCKER_IMAGE] + prepareLLMBuild(
            pipeline, CONFIG_LINUX_X86_64_SINGLE_DEVICE),
        ]
    }

    def packageConf = prepareLLMPackage(pipeline, cpu_arch)
    def artifacts = packageConf[0]
    def runner = packageConf[1]

    rtServer (
        id: 'Artifactory',
        url: 'https://urm.nvidia.com/artifactory',
        credentialsId: 'urm-artifactory-creds',
        // If Jenkins is configured to use an http proxy, you can bypass the proxy when using this Artifactory server:
        bypassProxy: true,
        // Configure the connection timeout (in seconds).
        // The default value (if not configured) is 300 seconds:
        timeout: 300
    )
    def reuseArtifactPath = env.reuseArtifactPath
    if (reuseArtifactPath) {
        def stageName = "Reuse Check"
        newArtifacts = downloadArtifacts(stageName, reuseArtifactPath, artifacts)
        if (!newArtifacts) {
            echo "previous package does not exist, rebuild all the artifacts"
            reuseArtifactPath = null
        } else {
            artifacts = newArtifacts
            runner = null
        }
    }

    def k8s_cpu = "amd64"
    if (cpu_arch == AARCH64_TRIPLE) {
        k8s_cpu = "arm64"
    }

    parallelJobs = buildConfigs.collectEntries{key, values -> [key, {
        script {
            buildOrCache(pipeline, key, reuseArtifactPath, values[1], values[0], k8s_cpu, values[2])
        }
    }]}
    parallelJobs.failFast = enableFailFast

    if (cpu_arch == X86_64_TRIPLE && !reuseArtifactPath) {
        def key = "Build with build type Debug"
        parallelJobs += [
        (key): {
            script {
                trtllm_utils.launchKubernetesPod(pipeline, createKubernetesPodConfig(LLM_DOCKER_IMAGE, "build", k8s_cpu), "trt-llm", {
                    stage(key) {
                        stage("[${key}] Run") {
                            echoNodeAndGpuInfo(pipeline, key)
                            buildWheelInContainer(pipeline, [], X86_64_TRIPLE, false, false, "cp312", "-a '90-real' -b Debug --benchmarks --micro_benchmarks")
                        }
                    }
                })
            }
        }]
    }

    stage("Build") {
        pipeline.parallel parallelJobs
    } // Build stage
    stage("Package") {
        container("trt-llm") {
            if (!reuseArtifactPath) {
                runner()
            }
            uploadArtifacts(artifacts)
        }
    }
}

pipeline {
    agent {
        kubernetes createKubernetesPodConfig(AGENT_IMAGE, "package", "amd64")
    }
    options {
        // Check the valid options at: https://www.jenkins.io/doc/book/pipeline/syntax/
        // some step like results analysis stage, does not need to check out source code
        skipDefaultCheckout()
        // to better analyze the time for each step/test
        timestamps()
        timeout(time: 24, unit: 'HOURS')
    }
    environment {
        //Workspace normally is: /home/jenkins/agent/workspace/LLM/L0_MergeRequest@tmp/
        HF_HOME="${env.WORKSPACE_TMP}/.cache/huggingface"
        CCACHE_DIR="${CCACHE_DIR}"
        PIP_INDEX_URL="https://urm.nvidia.com/artifactory/api/pypi/pypi-remote/simple"
        // force datasets to be offline mode, to prevent CI jobs are downloading HF dataset causing test failures
        HF_DATASETS_OFFLINE=1
    }
    stages {
        stage("BuildJob") {
            steps {
                launchStages(this, params.targetArch, params.enableFailFast, globalVars)
            }
        }
    } // stage
} // pipeline
