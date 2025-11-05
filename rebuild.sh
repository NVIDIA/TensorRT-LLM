# python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a "89-real;90-real;100-real;120-real" --use_ccache --clean -s 2>&1 | tee -a /home/scratch.timothyg_gpu/TensorRT-LLM/build_logs.log
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a "89-real;90-real;100-real;120-real" --use_ccache -s 2>&1 | tee -a /home/scratch.timothyg_gpu/TensorRT-LLM/build_logs.log

# python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a native --use_ccache -s &> /home/scratch.timothyg_gpu/TensorRT-LLM/build_logs.log


# Current using:
# python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a native --use_ccache --clean -s &> /home/scratch.timothyg_gpu/TensorRT-LLM/build_logs.log
# python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a native --use_ccache -s &> /home/scratch.timothyg_gpu/TensorRT-LLM/build_logs.log

# python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j"$(nproc)" -a native --use_ccache -s \
#   2>&1 | tee -a /home/scratch.timothyg_gpu/TensorRT-LLM/build_logs.log
