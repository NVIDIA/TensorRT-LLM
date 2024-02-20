#/bin/bash
parent_dir=$(dirname $(pwd))
pip3 uninstall tensorrt-llm
echo "==== ${parent_dir}"
TensorRT_path=/home/nvcntse/qgai/jingwei_station/TensorRT-9.0.1.4
python3 command_build_install.py --TensorRT_path=${TensorRT_path} --clean
# #--nccl_root ${parent_dir}/nccl/build --clean 
