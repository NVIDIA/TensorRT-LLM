cmake -S ./third-party -B ./build_deps/third-party
make -C ./build_deps/third-party -j 10
rm -rf ./build_deps/third-party
