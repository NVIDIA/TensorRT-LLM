#!/bin/sh

set -e
set -x

#bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 64 -b 3 -v 2 -h 2
#bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 64 -b 3 -v 2 -h 2 -scale-bmm1 0.25
#bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 64 -b 3 -v 2 -h 2 -use-tma
#bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 64 -b 3 -v 2 -h 2 -scale-bmm1 0.25 -use-tma

####################################################################################################
# H G M M A  F P 1 6 . F P 1 6
bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1

bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1 -ignore-b1opt

####################################################################################################
# Q G M M A  E 4 M 3 . F P 3 2
bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1 -e4m3
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1 -e4m3
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1 -e4m3
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1 -e4m3

bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1 -e4m3 -ignore-b1opt

####################################################################################################
# I G M M A  I N T 8 . I N T 3 2
bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1 -int8
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1 -int8
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1 -int8
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1 -int8

bin/fmha.exe -v 0 -runs 1 -s 512 -d 64 -min-s 1 -b 1 -h 1 -int8 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 384 -d 64 -min-s 1 -b 1 -h 1 -int8 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 256 -d 64 -min-s 1 -b 1 -h 1 -int8 -ignore-b1opt
bin/fmha.exe -v 0 -runs 1 -s 128 -d 64 -min-s 1 -b 1 -h 1 -int8 -ignore-b1opt
