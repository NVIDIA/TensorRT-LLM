#!/bin/bash

set -ex

# Use latest stable version from https://pypi.org/project/torch/#history
# and closest to the version specified in
# https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html#rel-25-05
TORCH_VERSION="2.7.1"
SYSTEM_ID=$(grep -oP '(?<=^ID=).+' /etc/os-release | tr -d '"')

prepare_environment() {
    if [[ $SYSTEM_ID == *"ubuntu"* ]]; then
      apt-get update && apt-get -y install ninja-build
      apt-get clean && rm -rf /var/lib/apt/lists/*
    elif [[ $SYSTEM_ID == *"rocky"* ]]; then
      dnf makecache --refresh && dnf install -y ninja-build && dnf clean all
    else
      echo "This system type cannot be supported..."
      exit 1
    fi
}

install_from_source() {
    if [[ $SYSTEM_ID == *"rocky"* ]]; then
      VERSION_ID=$(grep -oP '(?<=^VERSION_ID=).+' /etc/os-release | tr -d '"')
      if [[ $VERSION_ID == "7" ]]; then
        echo "Installation from PyTorch source codes cannot be supported..."
        exit 1
      fi
    fi
    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi

    prepare_environment $1

    export _GLIBCXX_USE_CXX11_ABI=$1

    export TORCH_CUDA_ARCH_LIST="8.0;8.6;9.0;10.0;12.0"
    export PYTORCH_BUILD_VERSION=${TORCH_VERSION}
    export PYTORCH_BUILD_NUMBER=0
    export MAX_JOBS=12
    pip3 uninstall -y torch
    cd /tmp
    git clone --depth 1 --branch v${TORCH_VERSION} https://github.com/pytorch/pytorch
    cd pytorch
    git submodule sync && git submodule update --init --recursive
    pip3 install -r requirements.txt
    python3 setup.py install
    cd /tmp && rm -rf /tmp/pytorch

    # Get torchvision version by dry run
    TORCHVISION_VERSION=$(pip3 install --dry-run torch==${TORCH_VERSION} torchvision | grep "Would install" | tr ' ' '\n' | grep torchvision | cut -d "-" -f 2)

    export PYTORCH_VERSION=${PYTORCH_BUILD_VERSION}
    export FORCE_CUDA=1
    export BUILD_VERSION=${TORCHVISION_VERSION}
    export MAX_JOBS=12
    pip3 uninstall -y torchvision
    cd /tmp
    git clone --depth 1 --branch v${TORCHVISION_VERSION} https://github.com/pytorch/vision
    cd vision
    python3 setup.py install
    cd /tmp && rm -rf /tmp/vision
}

install_from_pypi() {
    ARCH=$(uname -m)
    if [ "$ARCH" = "amd64" ];then ARCH="x86_64";fi
    if [ "$ARCH" = "aarch64" ];then ARCH="sbsa";fi

    pip3 uninstall -y torch torchvision torchaudio
    pip3 install torch==${TORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
}

# Apply upstream fix for AttributeError: 'NoneType' object has no attribute 'store_cubin'
# https://github.com/pytorch/pytorch/pull/150860/files
# Remove this after upgrade pytorch to DLFW 25.08 or later
apply_patch() {
    echo "Applying patch for PyTorch..."
    pushd `python -c 'import torch; print(torch.__path__[0])'`
    cat > triton_heuristics.patch << 'EOF'
diff --git a/_inductor/runtime/triton_heuristics.py b/_inductor/runtime/triton_heuristics.py
index be02d43..daf1afa 100644
--- a/_inductor/runtime/triton_heuristics.py
+++ b/_inductor/runtime/triton_heuristics.py
@@ -978,7 +978,15 @@ class CachingAutotuner(KernelInterface):
                 self.autotune_time_taken_ns + coordesc_time_taken_ns,
                 found_by_coordesc=True,
             )
-        return config2launcher.get(best_config)
+
+        if best_config not in config2launcher:
+            # On a Coordesc cache hit, we might not have loaded the launcher
+            # This can happen because PyCodeCache saves CachingAutotuners in memory,
+            # even for separate compile IDs (which can have different inputs without changing output code)
+            config2launcher[best_config] = self._precompile_config(
+                best_config
+            ).make_launcher()
+        return config2launcher[best_config]

     def run(
         self,
EOF
    git apply triton_heuristics.patch
    rm triton_heuristics.patch
    popd
}

case "$1" in
  "skip")
    apply_patch
    ;;
  "pypi")
    install_from_pypi
    apply_patch
    ;;
  "src_cxx11_abi")
    install_from_source 1
    ;;
  "src_non_cxx11_abi")
    install_from_source 0
    ;;
  *)
    echo "Incorrect input argument..."
    exit 1
    ;;
esac
