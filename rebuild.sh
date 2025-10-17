# python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a "89-real;90-real;100-real;120-real" --use_ccache -s


# python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a native --use_ccache --clean -s
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt -j$(nproc) -a native --use_ccache -s
