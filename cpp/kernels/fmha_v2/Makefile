# ##################################################################################################
#  Copyright (c) 2011-2023, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are not permit-
#  ted.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
#  IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFIT;
#  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
#  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# ##################################################################################################

# #################################################################################################
# Compilers and build options.
# #################################################################################################
SHELL := /bin/bash

# The CUDA toolkit.
CUDA  ?= /usr/local/cuda
# The path to cudnn.
CUDNN ?= /usr/local/cudnn

CMAKE_FLAGS = -DTORCH_CUDA_ARCH_LIST="8.0;9.0"
ifdef USE_CCACHE
  CCACHE = ccache
  CMAKE_FLAGS += -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
  CMAKE_FLAGS += -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache
endif

IS_CUDA11 ?= 1

TMP_DIR   ?= ./temp

# The C++ compiler.
CXX ?= $(CCACHE) g++
# The CUDA compiler.
NVCC ?= $(CCACHE) $(CUDA)/bin/nvcc

# Flags to compile C++ files.
CXX_FLAGS = $(CXXFLAGS) -O3 -std=c++17 -g
# Flags to compile CUDA files.
NVCC_FLAGS = $(CUDAFLAGS) -O3 -std=c++17 -ccbin $(CXX) -use_fast_math -Xptxas=-v --expt-relaxed-constexpr

# Remove -g -lineinfo when generating cubin files
ifndef GENERATE_CUBIN
	NVCC_FLAGS += -g -lineinfo
endif

# Google Test
GTEST_DIR ?= /usr
GTEST_LIB = -lgtest -lgtest_main
GTEST_INC = -I$(GTEST_DIR)/include

# Special SM89 support for QMMA.
ifdef ENABLE_SM89_QMMA
  NVCC_FLAGS += -DFMHA_ENABLE_SM89_QMMA
endif

# The different preprocessor definitions.

# Do we want to enable the ordering for the softmax-summation to produce bit exact results.
PREPROCESSOR_FLAGS += -DUSE_SAME_SUM_ORDER_IN_SOFTMAX_AS_REF_CODE

# Do we want to use half accumulation for flash attention
PREPROCESSOR_FLAGS += -DHALF_ACCUMULATION_FOR_FLASH_ATTENTION

# Add FLAGS when generating cubins.
ifdef GENERATE_CUBIN
	PREPROCESSOR_FLAGS += -DGENERATE_CUBIN
endif

# Output the P matrix and/or S = softmax(P) for debugging.
# PREPROCESSOR_FLAGS += -DSTORE_P
# PREPROCESSOR_FLAGS += -DSTORE_S
# PREPROCESSOR_FLAGS += -DDEBUG_HAS_PRINT_BUFFER

# Do we want to enable the fast trick to skip F2I and I2F.
I2F_F2I_FLAGS += -DUSE_I2F_EMULATION_TRICK
I2F_F2I_FLAGS += -DUSE_F2I_EMULATION_TRICK

# Append the preprocessor flags to the compilation flags.
CXX_FLAGS  += $(PREPROCESSOR_FLAGS)
NVCC_FLAGS += $(PREPROCESSOR_FLAGS)

# The include directories.
INCLUDE_DIRS += -I./src -I./generated -I$(CUDA)/include

GENCODE_SM80 = -gencode=arch=compute_80,code=\"sm_80\"
GENCODE_SM86 = -gencode=arch=compute_86,code=\"sm_86\"
GENCODE_SM87 = -gencode=arch=compute_87,code=\"sm_87\"
GENCODE_SM89 = -gencode=arch=compute_89,code=\"sm_89\"
GENCODE_SM90 = -gencode=arch=compute_90a,code=\"sm_90a\"

ifndef ENABLE_SM100
GENCODE_SM100 =
else
GENCODE_SM100 = -gencode=arch=compute_100,code=\"sm_100\"
endif

ifndef ENABLE_SM120
GENCODE_SM120 =
else
GENCODE_SM120 = -gencode=arch=compute_120,code=\"sm_120\"
endif


NVCC_FLAGS += --keep --keep-dir $(TMP_DIR)

# #################################################################################################
# The object files.
# #################################################################################################

ifneq ($(wildcard generated),)
include generated/makefile
endif

CUBIN_CPP = $(patsubst %.cu.cubin, %.cubin.cpp, $(CUBINS))
CUBIN_OBJ = $(patsubst %.cubin.cpp, %.cubin.o, $(CUBIN_CPP))

GENCODES =

GENCODES += $(GENCODE_SM80)
GENCODES += $(GENCODE_SM86)
GENCODES += $(GENCODE_SM89)
GENCODES += $(GENCODE_SM90)
GENCODES += $(GENCODE_SM100)
GENCODES += $(GENCODE_SM120)

SOFTMAX_GENCODES = $(GENCODE_SM80) $(GENCODE_SM89) $(GENCODE_SM90)
ifdef SOFTMAX_ALL_GENCODES
SOFTMAX_GENCODES = $(GENCODES)
endif

# #################################################################################################
# C++ unit tests
# #################################################################################################
UNIT_TEST_CPP_DIR = test/unit
UNIT_TEST_OBJ_DIR = obj/test/unit

# arch-independent
UNIT_TEST_CPP = $(wildcard $(UNIT_TEST_CPP_DIR)/*.cu)
UNIT_TEST_OBJ = $(patsubst %.cu, obj/%.o, $(UNIT_TEST_CPP))
UNIT_TEST_EXE = $(patsubst %.cu, bin/%.exe, $(UNIT_TEST_CPP))

# arch-dependent boilerplates
UNIT_TEST_CPP_SM80 = $(wildcard $(UNIT_TEST_CPP_DIR)/arch/*_sm80.cu)
UNIT_TEST_OBJ_SM80 = $(patsubst %_sm80.cu, obj/%_sm80.o, $(UNIT_TEST_CPP_SM80))
UNIT_TEST_EXE_SM80 = $(patsubst %_sm80.cu, bin/%_sm80.exe, $(UNIT_TEST_CPP_SM80))

# aggregate exes as prerequisite of build target "test"
UNIT_TEST_EXE_ARCH =
UNIT_TEST_EXE_ARCH += $(UNIT_TEST_EXE_SM80)

# #################################################################################################
# R U L E S
# #################################################################################################

.PHONY: all
all:
	$(MAKE) dirs
	$(MAKE) $(OBJECTS_MHA)
	$(MAKE) $(OBJECTS_MHCA)
	$(MAKE) bin/fmha.exe
	$(MAKE) bin/fmhca.exe

dirs:
	if [ ! -d bin ]; then mkdir -p bin; fi
	if [ ! -d obj ]; then mkdir -p obj; fi
	if [ ! -d cubin ]; then mkdir -p cubin; fi
	if [ ! -d temp ]; then mkdir -p temp; fi
	if [ ! -d obj/test/unit ]; then mkdir -p obj/test/unit; fi
	if [ ! -d obj/test/unit/arch ]; then mkdir -p obj/test/unit/arch; fi
	if [ ! -d bin/test/unit ]; then mkdir -p bin/test/unit; fi
	if [ ! -d bin/test/unit/arch ]; then mkdir -p bin/test/unit/arch; fi

.PHONY: cubin
cubin: $(CUBIN_CPP)
	$(MAKE) dirs
	$(MAKE) bin/libfmha_cubin.a

cubin_demobert:
	$(MAKE) dirs
	$(MAKE) PREPROCESSOR_FLAGS="$(PREPROCESSOR_FLAGS) -DUSE_DEMO_BERT_PARAMS=1" $(CUBIN_CPP)

clean:
	rm -rf bin obj cubin generated train_ops/build

###################################################################################################

.PHONY: deps
deps:
	python3 -m pip install -r requirements.txt
	sudo apt update && sudo apt install libgtest-dev -y

.PHONY: test
test: dirs all $(UNIT_TEST_EXE) $(UNIT_TEST_EXE_ARCH) train_ops

# default pytest options defined in pytest.ini
.PHONY: test_run
test_run: test
	python3 -m pytest

# dry run for prompt based tests
# -s to switch off stdout/stderr interception by pytest
# -n0 to switch off multi-process runs
.PHONY: test_dryrun
test_dryrun: test
	python3 -m pytest test/fmha -n0 -s -vv --dry-run

###################################################################################################

# convenience target to get train_ops built
.PHONY: train_ops
train_ops:
	pushd train_ops; python3 train_setup.py; popd
	cmake train_ops -Btrain_ops/build $(CMAKE_FLAGS)
	cmake --build train_ops/build -j$(shell nproc)

###################################################################################################

bin/fmha.exe: $(OBJECTS_MHA)
	$(CXX) $(CXX_FLAGS) -o $@ $^ -L$(CUDA)/lib64 -Wl,-rpath=$(CUDA)/lib64 -lcudart -lcudart -lcublas -lcublasLt

bin/fmhca.exe: $(OBJECTS_MHCA)
	$(CXX) $(CXX_FLAGS) -o $@ $^ -L$(CUDA)/lib64 -Wl,-rpath=$(CUDA)/lib64 -lcudart -lcudart -lcublas -lcublasLt

###################################################################################################

bin/libfmha_cubin.a: $(CUBIN_OBJ)
	ar rcs $@ $^

###################################################################################################

obj/%_sm80.cu.o: ./generated/%_sm80.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM80) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm86.cu.o: ./generated/%_sm86.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM86) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm87.cu.o: ./generated/%_sm87.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM87) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm89.cu.o: ./generated/%_sm89.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM89) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm90.cu.o: ./generated/%_sm90.cu ./src/*.h ./src/fmha/*.h ./src/fmha/hopper/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM90) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm100.cu.o: ./generated/%_sm100.cu ./src/*.h ./src/fmha/*.h ./src/fmha/hopper/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM100) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm120.cu.o: ./generated/%_sm120.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM120) $(INCLUDE_DIRS) -c -o $@ $<

obj/%_sm80.no_i2f_f2i.cu.o: ./generated/%_sm80.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM80) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm86.no_i2f_f2i.cu.o: ./generated/%_sm86.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM86) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm87.no_i2f_f2i.cu.o: ./generated/%_sm87.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM87) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm89.no_i2f_f2i.cu.o: ./generated/%_sm89.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM89) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm90.no_i2f_f2i.cu.o: ./generated/%_sm90.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM90) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm100.no_i2f_f2i.cu.o: ./generated/%_sm100.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM100) $(INCLUDE_DIRS) -c -o $@ $<
obj/%_sm120.no_i2f_f2i.cu.o: ./generated/%_sm120.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM120) $(INCLUDE_DIRS) -c -o $@ $<

obj/softmax_%.cu.o: ./src/softmax_%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(SOFTMAX_GENCODES) -c -o $@ $<
obj/convert.cu.o: ./src/convert.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDE_DIRS) $(GENCODES) -c -o $@ $<
obj/%.cpp.o: ./src/%.cpp ./src/*.h ./generated/%_api.h
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIRS) -I$(CUDA)/include -c -o $@ $<

###################################################################################################
# C++ unit test build rule
###################################################################################################

# arch-independent test exes
$(UNIT_TEST_EXE): bin/test/unit/test_%.exe : $(UNIT_TEST_OBJ_DIR)/test_%.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(GTEST_LIB)

# arch-dependent test exes
$(UNIT_TEST_EXE_ARCH): bin/test/unit/arch/%.exe : $(UNIT_TEST_OBJ_DIR)/arch/%.o
	$(NVCC) $(NVCC_FLAGS) -o $@ $^ $(GTEST_LIB)

# arch-independent objs
$(UNIT_TEST_OBJ): $(UNIT_TEST_OBJ_DIR)/%.o : ${UNIT_TEST_CPP_DIR}/%.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODES) -c -o $@ $< -I./src $(GTEST_INC)

# arch-dependent objs
$(UNIT_TEST_OBJ_SM80): %.o : $(UNIT_TEST_CPP_SM80) ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM80) -c -o $@ $< -I./src $(GTEST_INC)

###################################################################################################

cubin/%_sm80.cu.cubin: ./generated/%_sm80.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM80) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm86.cu.cubin: ./generated/%_sm86.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM86) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm87.cu.cubin: ./generated/%_sm87.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM87) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm89.cu.cubin: ./generated/%_sm89.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM89) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm90.cu.cubin: ./generated/%_sm90.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM90) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm100.cu.cubin: ./generated/%_sm100.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM100) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm120.cu.cubin: ./generated/%_sm120.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(I2F_F2I_FLAGS) $(GENCODE_SM120) $(INCLUDE_DIRS) -cubin -o $@ $<

cubin/%_sm80.no_i2f_f2i.cu.cubin: ./generated/%_sm80.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM80) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm86.no_i2f_f2i.cu.cubin: ./generated/%_sm86.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM86) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm87.no_i2f_f2i.cu.cubin: ./generated/%_sm87.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM87) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm89.no_i2f_f2i.cu.cubin: ./generated/%_sm89.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM89) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm90.no_i2f_f2i.cu.cubin: ./generated/%_sm90.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM90) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm100.no_i2f_f2i.cu.cubin: ./generated/%_sm100.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM100) $(INCLUDE_DIRS) -cubin -o $@ $<
cubin/%_sm120.no_i2f_f2i.cu.cubin: ./generated/%_sm120.no_i2f_f2i.cu ./src/*.h ./src/fmha/*.h
	$(NVCC) $(NVCC_FLAGS) $(GENCODE_SM120) $(INCLUDE_DIRS) -cubin -o $@ $<

###################################################################################################

cubin/%.cubin.cpp: cubin/%.cu.cubin
	xxd -i $< >& $@
