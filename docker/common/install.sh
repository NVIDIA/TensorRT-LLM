#!/bin/bash

# Function to assert that all provided variable names are non-empty
assert_non_empty() {
    for var_name in "$@"; do
        local var_value="${!var_name}"
        
        # Check if the variable is empty or not set
        if [[ -z "$var_value" ]]; then
            echo "Error: Variable '$var_name' is empty or not set" >&2
            exit 1
        fi
    done
}

echo "TRT_VER: $TRT_VER"
echo "CUDA_VER: $CUDA_VER"
echo "CUDNN_VER: $CUDNN_VER"
echo "NCCL_VER: $NCCL_VER"
echo "CUBLAS_VER: $CUBLAS_VER"



# Default values
base=0
cmake=0
ccache=0
cuda_toolkit=0
tensorrt=0
polygraphy=0
mpi4py=0
pytorch=0
opencv=0
protobuf=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --base)
            base=1
            shift 1
            ;;
        --cmake)
            cmake=1
            shift 1
            ;;
        --ccache)
            ccache=1
            shift 1
            ;;
        --cuda_toolkit)
            cuda_toolkit=1
            shift 1
            ;;
        --tensorrt)
            tensorrt=1
            shift 1
            ;;
        --polygraphy)
            polygraphy=1
            shift 1
            ;;
        --mpi4py)
            mpi4py=1
            shift 1
            ;;
        --pytorch)
            pytorch=1
            shift 1
            ;;
        --opencv)
            opencv=1
            shift 1
            ;;
        --protobuf)
            protobuf=1
            shift 1
            ;;
        --all)
            base=1
            cmake=1
            ccache=1
            cuda_toolkit=1
            tensorrt=1
            polygraphy=1
            mpi4py=1
            pytorch=1
            opencv=1
            protobuf=1
            shift 1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

GITHUB_MIRROR=""
TORCH_INSTALL_TYPE="skip"
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.99999"

if [ $base -eq 1 ]; then
    echo "Installing base dependencies..."
    PYTHON_VERSION="3.12.3"
    echo "Using Python version: $PYTHON_VERSION"
    GITHUB_MIRROR=$GITHUB_MIRROR bash ./install_base.sh $PYTHON_VERSION
fi

if [ $cmake -eq 1 ]; then
    echo "Installing CMake..."
    GITHUB_MIRROR=$GITHUB_MIRROR bash ./install_cmake.sh
fi

if [ $ccache -eq 1 ]; then
    echo "Installing ccache..."
    GITHUB_MIRROR=$GITHUB_MIRROR bash ./install_ccache.sh
fi

if [ $cuda_toolkit -eq 1 ]; then
    echo "Installing CUDA toolkit..."
    GITHUB_MIRROR=$GITHUB_MIRROR bash ./install_cuda_toolkit.sh
fi

if [ $tensorrt -eq 1 ]; then
    echo "Installing TensorRT..."
    bash ./install_tensorrt.sh \
        --TRT_VER=${TRT_VER} \
        --CUDA_VER=${CUDA_VER} \
        --CUDNN_VER=${CUDNN_VER} \
        --NCCL_VER=${NCCL_VER} \
        --CUBLAS_VER=${CUBLAS_VER}
fi

if [ $polygraphy -eq 1 ]; then
    echo "Installing Polygraphy..."
    bash ./install_polygraphy.sh
fi

if [ $mpi4py -eq 1 ]; then
    echo "Installing mpi4py..."
    GITHUB_MIRROR=$GITHUB_MIRROR bash ./install_mpi4py.sh
fi

if [ $pytorch -eq 1 ]; then
    echo "Installing PyTorch..."
    bash ./install_pytorch.sh $TORCH_INSTALL_TYPE
fi

if [ $opencv -eq 1 ]; then
    echo "Installing OpenCV..."
    pip3 uninstall -y opencv && \
    rm -rf /usr/local/lib/python3*/dist-packages/cv2/ && \
    pip3 install opencv-python-headless --force-reinstall --no-deps --no-cache-dir
fi

if [ $protobuf -eq 1 ]; then
    pip3 install --upgrade --no-cache-dir \
    "protobuf>=4.25.8"
fi

