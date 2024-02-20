#/bin/bash
parent_dir=$(dirname $(pwd))
echo "parent_dir is ${parent_dir}"

TensorRT_path=/home/nvcntse/qgai/jingwei_station/TensorRT-9.0.1.4
cmake_path=/home/nvcntse/qgai/cmake-3.27.7-linux-aarch64/bin
export PATH=${TensorRT_path}/bin:${cmake_path}:${PATH}
export LD_LIBRARY_PATH=${TensorRT_path}/lib:${LD_LIBRARY_PATH}