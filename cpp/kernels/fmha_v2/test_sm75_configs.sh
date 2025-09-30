# FP16: different b, fixed and var.seqlen

bin/fmha.exe -v 0 -runs 1 -s 256 -d 32 -min-s 256 -b 1
bin/fmha.exe -v 0 -runs 1 -s 256 -d 32 -min-s 256 -b 128
bin/fmha.exe -v 0 -runs 1 -s 256 -d 32 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 256 -d 32 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 128 -d 32 -min-s 128 -b 1
bin/fmha.exe -v 0 -runs 1 -s 128 -d 32 -min-s 128 -b 128
bin/fmha.exe -v 0 -runs 1 -s 128 -d 32 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 128 -d 32 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 384 -b 1
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 384 -b 128
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 256 -b 1
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 256 -b 128
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 128 -b 1
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 128 -b 128
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s   1 -b 128

# FP16: different b, fixed and var.seqlen (longer sequence length for flash attention)
# NOTE: HALF_ACCUMULATION_FOR_FLASH_ATTENTION has larger epsilon.

bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 512 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 512 -b 128 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s   1 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s   1 -b 128 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 512 -d 32 -min-s 512 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 32 -min-s 512 -b 128 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 32 -min-s   1 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 32 -min-s   1 -b 128 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 512 -d 16 -min-s 512 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 16 -min-s 512 -b 128 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 16 -min-s   1 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 512 -d 16 -min-s   1 -b 128 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 1024 -d 64 -min-s 1024 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 64 -min-s 1024 -b 128 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 64 -min-s   1  -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 64 -min-s   1  -b 128 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 1024 -d 32 -min-s 1024 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 32 -min-s 1024 -b 128 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 32 -min-s   1  -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 32 -min-s   1  -b 128 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 1024 -d 16 -min-s 1024 -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 16 -min-s 1024 -b 128 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 16 -min-s   1  -b 1   -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 1024 -d 16 -min-s   1  -b 128 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 2048 -d 64 -min-s 2048 -b 1  -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 64 -min-s 2048 -b 32 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 64 -min-s   1  -b 1  -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 64 -min-s   1  -b 32 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 2048 -d 32 -min-s 2048 -b 1  -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 32 -min-s 2048 -b 32 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 32 -min-s   1  -b 1  -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 32 -min-s   1  -b 32 -epsilon 0.02

bin/fmha.exe -v 0 -runs 1 -s 2048 -d 16 -min-s 2048 -b 1  -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 16 -min-s 2048 -b 32 -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 16 -min-s   1  -b 1  -epsilon 0.02
bin/fmha.exe -v 0 -runs 1 -s 2048 -d 16 -min-s   1  -b 32 -epsilon 0.02

# INT8: different b, fixed and var.seqlen

bin/fmha.exe -v 0 -runs 1 -int8 -s 512 -d 32 -min-s 512 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 512 -d 32 -min-s 512 -b 128
bin/fmha.exe -v 0 -runs 1 -int8 -s 512 -d 32 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 512 -d 32 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 32 -min-s 256 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 32 -min-s 256 -b 128
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 32 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 32 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 32 -min-s 128 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 32 -min-s 128 -b 128
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 32 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 32 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 384 -d 64 -min-s 384 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 384 -d 64 -min-s 384 -b 128
bin/fmha.exe -v 0 -runs 1 -int8 -s 384 -d 64 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 384 -d 64 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 64 -min-s 256 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 64 -min-s 256 -b 128
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 64 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 64 -min-s   1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 64 -min-s 128 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 64 -min-s 128 -b 128
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 64 -min-s   1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 64 -min-s   1 -b 128

# FP16: different b, fixed and var.seqlen

bin/fmha.exe -v 0 -runs 1 -s 512 -d 32 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 512 -d 32 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 256 -d 32 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 256 -d 32 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 128 -d 32 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 128 -d 32 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -v1 -b 128

# INT8: different b, fixed and var.seqlen

bin/fmha.exe -v 0 -runs 1 -int8 -s 512 -d 32 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 512 -d 32 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 32 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 32 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 32 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 32 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 384 -d 64 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 384 -d 64 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 64 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 256 -d 64 -v1 -b 128

bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 64 -v1 -b 1
bin/fmha.exe -v 0 -runs 1 -int8 -s 128 -d 64 -v1 -b 128
