# ARCH=$(uname -m)
# if [ $ARCH == "x86_64" ]; then

# wget https://urm.nvidia.com/artifactory/sw-gpu-cuda-installer-generic-local/packaging/r13.0/cuda_nvrtc/linux-x86_64/13.0.48/cuda-nvrtc-dev-13-0_13.0.48-1_amd64.deb && \
#     dpkg -i cuda-nvrtc-dev-13-0_13.0.48-1_amd64.deb && \
#     rm cuda-nvrtc-dev-13-0_13.0.48-1_amd64.deb

# wget https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-x86_64.sh && \
#     bash cmake-4.0.3-linux-x86_64.sh --skip-license --prefix=/usr/local/cmake --exclude-subdir

# apt update
# apt install -y libstdc++-14-dev

# elif [ $ARCH == "aarch64" ]; then

# # to be moved to docker/common/ scripts
# wget https://urm.nvidia.com/artifactory/sw-gpu-cuda-installer-generic-local/packaging/r13.0/cuda_nvrtc/linux-sbsa/13.0.48/cuda-nvrtc-dev-13-0_13.0.48-1_arm64.deb && \
#     dpkg -i cuda-nvrtc-dev-13-0_13.0.48-1_arm64.deb && \
#     rm cuda-nvrtc-dev-13-0_13.0.48-1_arm64.deb

# wget https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-linux-aarch64.sh && \
#     bash cmake-4.0.3-linux-aarch64.sh --skip-license --prefix=/usr/local/cmake --exclude-subdir

# apt update
# # fix LLVM build
# apt install -y libstdc++-14-dev

# # wait for https://github.com/NVIDIA/TensorRT-LLM/pull/6588
# pip install deep_gemm@git+https://github.com/VALLIS-NERIA/DeepGEMM.git@97d97a20c2ecd53a248ab64242219d780cf822b8 --no-build-isolation

# else
#     echo "Unsupported architecture: $ARCH"
#     exit 1
# fi

# # wait for new triton to be published
# cd /usr/local/lib/python3.12/dist-packages/ && \
#     ls -la | grep pytorch_triton && \
#     mv pytorch_triton-3.3.1+gitc8757738.dist-info triton-3.3.1+gitc8757738.dist-info && \
#     cd triton-3.3.1+gitc8757738.dist-info && \
#     echo "Current directory: $(pwd)" && \
#     echo "Files in directory:" && \
#     ls -la && \
#     sed -i 's/^Name: pytorch-triton/Name: triton/' METADATA && \
#     sed -i 's|pytorch_triton-3.3.1+gitc8757738.dist-info/|triton-3.3.1+gitc8757738.dist-info/|g' RECORD && \
#     echo "METADATA after update:" && \
#     grep "^Name:" METADATA

# # pip install git+https://github.com/triton-lang/triton.git@main
