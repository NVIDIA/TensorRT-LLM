# ##################################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2011-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ##################################################################################################

# #################################################################################################
# Compilers and build options.
# #################################################################################################

# The CUDA toolkit.
CUDA  ?= /usr/local/cuda
# The path to cudnn.
CUDNN ?= /usr/local/cudnn

IS_CUDA11 ?= 1

# The C++ compiler.
CXX ?= g++
# The CUDA compiler.
NVCC ?= $(CUDA)/bin/nvcc

# Flags to compile C++ files.
CXX_FLAGS = $(CXXFLAGS) -O3 -std=c++14 -g -DSAMPLES
# Flags to compile CUDA files.
#NVCC_FLAGS = $(CUDAFLAGS) -O3 -std=c++14 -g -lineinfo -ccbin $(CXX) -Xptxas=-v -use_fast_math
NVCC_FLAGS = $(CUDAFLAGS) -O3 -std=c++14 -g -lineinfo -ccbin $(CXX)  -use_fast_math

# The different preprocessor definitions.
PREPROCESSOR_FLAGS = -DMYGEN

# Do we want to enable the ordering for the softmax-summation to produce bit exact results.
PREPROCESSOR_FLAGS += -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE

# Do we want to enable the fast trick to skip F2I and I2F.
PREPROCESSOR_FLAGS += -DUSE_I2F_EMULATION_TRICK
PREPROCESSOR_FLAGS += -DUSE_F2I_EMULATION_TRICK

# Output the P matrix and/or S = softmax(P) for debugging.
# PREPROCESSOR_FLAGS += -DSTORE_P
# PREPROCESSOR_FLAGS += -DSTORE_S

# Append the preprocessor flags to the compilation flags.
CXX_FLAGS  += $(PREPROCESSOR_FLAGS)
NVCC_FLAGS += $(PREPROCESSOR_FLAGS)

# The include directories.
INCLUDE_DIRS += -I../src

GENCODE_SM75 = -gencode=arch=compute_75,code=\"sm_75\"
GENCODE_SM80 = -gencode=arch=compute_80,code=\"sm_80\"

# #################################################################################################
# The object files.
# #################################################################################################
OBJECTS  = obj/fmha_fprop_v2_fp16_128_64_kernel.sm80.cu.o
OBJECTS += obj/fmha_fprop_v2_fp16_256_64_kernel.sm80.cu.o
OBJECTS += obj/fmha_fprop_v2_fp16_384_64_kernel.sm80.cu.o
OBJECTS += obj/fmha_fprop_v2_fp16_512_64_kernel.sm80.cu.o

OBJECTS += obj/fmha_dgrad_v2_fp16_128_64_kernel.sm80.cu.o
OBJECTS += obj/fmha_dgrad_v2_fp16_256_64_kernel.sm80.cu.o
OBJECTS += obj/fmha_dgrad_v2_fp16_384_64_kernel.sm80.cu.o
OBJECTS += obj/fmha_dgrad_v2_fp16_512_64_kernel.sm80.cu.o

GENCODES = $(GENCODE_SM75) $(GENCODE_SM80)

# #################################################################################################
# R U L E S
# #################################################################################################

.PHONY: all
all:
	$(MAKE) dirs
	$(MAKE) $(OBJECTS)

dirs:
	if [ ! -d obj ]; then mkdir -p obj; fi

clean:
	rm -rf obj

###################################################################################################

obj/%.sm75.cu.o: ./%.sm75.cu ./*.h ../src/*.h ../src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM75) $(INCLUDE_DIRS) -c -o $@ $<
obj/%.sm80.cu.o: ./%.sm80.cu ./*.h ../src/*.h ../src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM80) $(INCLUDE_DIRS) -c -o $@ $<
###################################################################################################
