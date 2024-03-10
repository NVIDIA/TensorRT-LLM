mkdir build
cd build
cmake .. -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_NITRO=ON -DBUILD_BATCH_MANAGER_DEFAULT=OFF -DCMAKE_CUDA_ARCHITECTURES=89-real -DTRT_LIB_DIR=/usr/local/tensorrt/lib -DTRT_INCLUDE_DIR=/usr/local/tensorrt/include -DCMAKE_BUILD_TYPE=Release
make -j $(nproc)
