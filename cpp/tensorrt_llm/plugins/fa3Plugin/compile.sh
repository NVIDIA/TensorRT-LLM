g++ test.cpp -Wl,-rpath=. -L. -l:_C.abi3.so -I/usr/local/cuda/include -I./flash-attention/hopper -L/usr/local/cuda/lib64/  -lcudart
