# Makefile for Cortex cortex.tensorrt-llm engine - Build, Lint, Test, and Clean
.PHONY: all build package run-e2e-test

RUN_TESTS ?= false
CODE_SIGN ?= false
AZURE_KEY_VAULT_URI ?= xxxx
AZURE_CLIENT_ID ?= xxxx
AZURE_TENANT_ID ?= xxxx
AZURE_CLIENT_SECRET ?= xxxx
AZURE_CERT_NAME ?= xxxx

# Default target, does nothing
all:
	@echo "Specify a target to run"

# Build the Cortex engine
build-deps:
ifeq ($(OS),Windows_NT)
	@powershell -Command "cd tensorrt_llm/cortex.tensorrt-llm; cmake -S ./third-party -B ./build_deps/third-party -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -DCMAKE_BUILD_TYPE=Release -DCMAKE_OBJECT_PATH_MAX=500; cmake --build ./build_deps/third-party --config Release -j8;"
else
	@cd tensorrt_llm/cortex.tensorrt-llm && cmake -S ./third-party -B ./build_deps/third-party -DCMAKE_BUILD_TYPE=Release && make -C ./build_deps/third-party -j 10 && rm -rf ./build_deps/third-party;
endif

# Build the Cortex engine
build:
ifeq ($(OS),Windows_NT)
	@powershell -Command "cd ..; python .\scripts\build_wheel.py -a '80-real;86-real;89-real' --trt_root 'C:\workspace\TensorRT-9.3.0.1\' -D 'BUILD_CORTEX_TENSORRT-LLM=ON' --use_ccache"
	@powershell -Command "cd build; cmake .. -DCMAKE_CUDA_ARCHITECTURES='80-real;86-real;89-real' -DTRT_LIB_DIR='C:/workspace/TensorRT-9.3.0.1/lib' -DTRT_INCLUDE_DIR='C:/workspace/TensorRT-9.3.0.1/include' -DBUILD_CORTEX_TENSORRT-LLM=ON -DCMAKE_CUDA_COMPILER='C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2/bin/nvcc.exe' -DENABLE_MULTI_DEVICE=0 -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -G Ninja; cmake --build . --parallel 2 --config Release"
else
	@cd .. && python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --cuda_architectures "80-real;86-real;89-real" --extra-cmake-vars "BUILD_CORTEX_TENSORRT-LLM=ON" --use_ccache
	@cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES='80-real;86-real;89-real' -DTRT_LIB_DIR='/usr/local/tensorrt/lib' -DTRT_INCLUDE_DIR='/usr/local/tensorrt/include' -DBUILD_CORTEX_TENSORRT-LLM=ON -DCMAKE_CXX_COMPILER_LAUNCHER=ccache -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache -G Ninja && cmake --build . --config Release;
endif

# Prepackage the Cortex engine
pre-package:
ifeq ($(OS),Windows_NT)
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; mkdir -p cortex.tensorrt-llm; cp ..\..\build\tensorrt_llm\cortex.tensorrt-llm\engine.dll cortex.tensorrt-llm\;"
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; cp -Force C:\workspace\TensorRT-9.3.0.1\lib\nvinfer.dll cortex.tensorrt-llm\;"
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; cp -Force C:\workspace\cuDNN\bin\cudnn64_8.dll cortex.tensorrt-llm\;"
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; cp -Force C:\workspace\cuDNN\bin\cudnn_ops_infer64_8.dll cortex.tensorrt-llm\;"
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; cp -Force ..\..\build\tensorrt_llm\plugins\nvinfer_plugin_tensorrt_llm.dll cortex.tensorrt-llm\;"
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; cp -Force ..\..\build\tensorrt_llm\tensorrt_llm.dll cortex.tensorrt-llm\;"
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; cp -Force .\build_deps\_install\bin\zlib.dll cortex.tensorrt-llm\;"
else
	cd ./tensorrt_llm/cortex.tensorrt-llm && \
	mkdir -p cortex.tensorrt-llm && \
	cp ../../build/tensorrt_llm/cortex.tensorrt-llm/libengine.so cortex.tensorrt-llm && \
	cp /usr/local/tensorrt/targets/x86_64-linux-gnu/lib/libnvinfer.so.9 cortex.tensorrt-llm && \
	cp /usr/lib/x86_64-linux-gnu/libcudnn.so.8 cortex.tensorrt-llm && \
	cp /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 cortex.tensorrt-llm && \
	cp /home/runner/actions-runner/_work/cortex.tensorrt-llm/cortex.tensorrt-llm/cpp/build/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so.9 cortex.tensorrt-llm && \
	cp /home/runner/actions-runner/_work/cortex.tensorrt-llm/cortex.tensorrt-llm/cpp/build/tensorrt_llm/libtensorrt_llm.so cortex.tensorrt-llm && \
	cp /usr/lib/x86_64-linux-gnu/libnccl.so cortex.tensorrt-llm/libnccl.so.2
endif

codesign:
ifeq ($(CODE_SIGN),false)
	@echo "Skipping Code Sign"
	@exit 0
endif

ifeq ($(OS),Windows_NT)
	@powershell -Command "dotnet tool install --global AzureSignTool;"
endif

package:
ifeq ($(OS),Windows_NT)
	@powershell -Command "cd tensorrt_llm\cortex.tensorrt-llm\; 7z a -ttar temp.tar cortex.tensorrt-llm\*; 7z a -tgzip cortex.tensorrt-llm.tar.gz temp.tar;"
else
	@cd tensorrt_llm/cortex.tensorrt-llm && \
	tar -czvf cortex.tensorrt-llm.tar.gz cortex.tensorrt-llm
endif

run-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
	@exit 0
endif
ifeq ($(OS),Windows_NT)
	@powershell -Command "echo hello"
else
	echo "hello"
endif

run-python-e2e-test:
ifeq ($(RUN_TESTS),false)
	@echo "Skipping tests"
	@exit 0
endif
ifeq ($(OS),Windows_NT)
	echo hello
else
	echo hello
endif

clean:
ifeq ($(OS),Windows_NT)
	echo hello
else
	echo "hello"
endif