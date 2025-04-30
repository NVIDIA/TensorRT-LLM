test_nvrtc:
	(cd .. && python3 gen_cpp_header.py) && g++ test_nvrtc.cpp -I/usr/local/cuda/include -I../generated -L/usr/local/cuda/lib64 -o test_nvrtc -lnvrtc -lcuda -lcudart

test: test_nvrtc
	./test_nvrtc

.PHONY: test_nvrtc test
