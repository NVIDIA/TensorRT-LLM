# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION &
# AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

add_subdirectory(kernels)

add_gtest(cacheTransceiverTest cacheTransceiverTest.cpp)
target_link_libraries(cacheTransceiverTest PRIVATE ${Python3_LIBRARIES})

add_gtest(mpiUtilsTest mpiUtilsTest.cpp)
add_gtest(ncclUniqueIdRendezvousTest ncclUniqueIdRendezvousTest.cpp)
if(ENABLE_MULTI_DEVICE)
  set(NCCL_RENDEZVOUS_RAW_RECOVERY_TEST
      NcclUniqueIdRendezvousTest.RawCommunicatorRebuildRunsPostRecoveryCollective
  )
  set(NCCL_RENDEZVOUS_PP_RECOVERY_TEST
      NcclUniqueIdRendezvousTest.PipelineCommunicatorRebuildRunsPostRecoverySendRecv
  )
  set(NCCL_RENDEZVOUS_PROCESS_DEATH_TEST
      NcclUniqueIdRendezvousTest.ProcessDeathRendezvousRequiresOptInFaultTolerantMpiLauncher
  )
  set(NCCL_RENDEZVOUS_ISOLATED_TESTS
      "${NCCL_RENDEZVOUS_RAW_RECOVERY_TEST}:${NCCL_RENDEZVOUS_PP_RECOVERY_TEST}:${NCCL_RENDEZVOUS_PROCESS_DEATH_TEST}"
  )
  add_test(
    NAME ncclUniqueIdRendezvousTest.mpi3
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ncclUniqueIdRendezvousTest>
      --gtest_filter=NcclUniqueIdRendezvousTest.*-${NCCL_RENDEZVOUS_ISOLATED_TESTS}
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    ncclUniqueIdRendezvousTest.mpi3
    PROPERTIES ENVIRONMENT
               "CUDA_MODULE_LOADING=LAZY;TLLM_FAULT_TOLERANCE_MODE=1"
               PROCESSORS 3 TIMEOUT 180)

  # Raw communicator replacement mutates process-global NCCL state and needs one
  # distinct physical GPU per MPI rank. Keep it in its own CTest process.
  add_test(
    NAME ncclUniqueIdRendezvousTest.rawRecoveryMpi3
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ncclUniqueIdRendezvousTest>
      --gtest_filter=${NCCL_RENDEZVOUS_RAW_RECOVERY_TEST} ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    ncclUniqueIdRendezvousTest.rawRecoveryMpi3
    PROPERTIES ENVIRONMENT
               "CUDA_MODULE_LOADING=LAZY;TLLM_FAULT_TOLERANCE_MODE=1"
               PROCESSORS 3 TIMEOUT 180)

  # The runtime PP wrapper owns a distinct communicator implementation and
  # rendezvous tag set. Exercise its real post-recovery send/receive path in a
  # separate process so it cannot share native singleton state with raw ops.
  add_test(
    NAME ncclUniqueIdRendezvousTest.ppRecoveryMpi3
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ncclUniqueIdRendezvousTest>
      --gtest_filter=${NCCL_RENDEZVOUS_PP_RECOVERY_TEST} ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    ncclUniqueIdRendezvousTest.ppRecoveryMpi3
    PROPERTIES ENVIRONMENT
               "CUDA_MODULE_LOADING=LAZY;TLLM_FAULT_TOLERANCE_MODE=1"
               PROCESSORS 3 TIMEOUT 180)

  option(TRTLLM_ENABLE_NCCL_PROCESS_DEATH_TEST
         "Enable the destructive NCCL rendezvous test that exits one MPI rank"
         OFF)
  set(TRTLLM_NCCL_PROCESS_DEATH_MPIEXEC_PREFLAGS
      ""
      CACHE
        STRING
        "Semicolon-separated launcher recovery flags for the NCCL process-death test"
  )
  if(TRTLLM_ENABLE_NCCL_PROCESS_DEATH_TEST)
    if("${TRTLLM_NCCL_PROCESS_DEATH_MPIEXEC_PREFLAGS}" STREQUAL "")
      message(
        FATAL_ERROR
          "TRTLLM_ENABLE_NCCL_PROCESS_DEATH_TEST requires launcher-specific "
          "TRTLLM_NCCL_PROCESS_DEATH_MPIEXEC_PREFLAGS (for example OpenMPI recovery flags)"
      )
    endif()
    # Keep a launcher-level signal from bypassing the sentinel properties. The
    # CMake wrapper translates it to a normal nonzero process result.
    set(NCCL_RENDEZVOUS_PROCESS_DEATH_ENV
        CUDA_MODULE_LOADING=LAZY
        TLLM_FAULT_TOLERANCE_MODE=1
        TRTLLM_TEST_NCCL_PROCESS_DEATH=1
        TRTLLM_NCCL_NONBLOCKING_TIMEOUT_MS=20000
        TRTLLM_NCCL_RECOVERY_TIMEOUT_MS=20000)
    add_test(
      NAME ncclUniqueIdRendezvousTest.processDeathMpi3
      COMMAND
        ${CMAKE_COMMAND} -E env ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3
        ${MPIEXEC_PREFLAGS} ${TRTLLM_NCCL_PROCESS_DEATH_MPIEXEC_PREFLAGS}
        $<TARGET_FILE:ncclUniqueIdRendezvousTest>
        --gtest_filter=${NCCL_RENDEZVOUS_PROCESS_DEATH_TEST}
        ${MPIEXEC_POSTFLAGS})
    set_tests_properties(
      ncclUniqueIdRendezvousTest.processDeathMpi3
      PROPERTIES ENVIRONMENT
                 "${NCCL_RENDEZVOUS_PROCESS_DEATH_ENV}"
                 PROCESSORS
                 3
                 TIMEOUT
                 60
                 RUN_SERIAL
                 TRUE
                 # A launcher may report failure solely because the injected
                 # victim intentionally skips MPI_Finalize. The root emits this
                 # sentinel only after survivor-only communicator rebuild and a
                 # successful allreduce, so it is the authoritative pass
                 # condition.
                 PASS_REGULAR_EXPRESSION
                 "TRTLLM_PROCESS_DEATH_RENDEZVOUS_OK"
                 FAIL_REGULAR_EXPRESSION
                 "TRTLLM_PROCESS_DEATH_RENDEZVOUS_FAIL")
  endif()

  add_test(
    NAME ncclUniqueIdRendezvousTest.defaultOff
    COMMAND
      ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS}
      $<TARGET_FILE:ncclUniqueIdRendezvousTest>
      --gtest_filter=NcclUniqueIdRendezvousTest.FaultToleranceDisabledPreservesLegacyOperationPath
      ${MPIEXEC_POSTFLAGS})
  set_tests_properties(
    ncclUniqueIdRendezvousTest.defaultOff
    PROPERTIES
      ENVIRONMENT
      "CUDA_MODULE_LOADING=LAZY;TLLM_FAULT_TOLERANCE_MODE=0;TRTLLM_TEST_FT_DEFAULT_OFF=1"
      PROCESSORS
      1
      TIMEOUT
      60)
endif()
add_gtest(userBufferTest userBufferTest.cpp)
add_gtest(ncclUtilsTest ncclUtilsTest.cpp)
target_link_libraries(ncclUtilsTest PRIVATE ${Python3_LIBRARIES})
if(BUILD_PYT)
  target_compile_definitions(cacheTransceiverTest PUBLIC BUILD_PYT)
  target_link_libraries(cacheTransceiverTest PUBLIC ${TORCH_LIBRARIES})
endif()
if(BUILD_PYT)
  target_compile_definitions(ncclUtilsTest PUBLIC BUILD_PYT)
  target_link_libraries(ncclUtilsTest PUBLIC ${TORCH_LIBRARIES})
endif()
