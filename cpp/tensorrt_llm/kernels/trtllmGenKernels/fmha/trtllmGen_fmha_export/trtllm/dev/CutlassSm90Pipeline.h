/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cute/layout.hpp"
#include "cute/layout_composed.hpp" // cute::composition
#include "cute/swizzle.hpp"         // cute::Swizzle
#include "cute/swizzle_layout.hpp"  // cute::composition
#include "cute/util/type_traits.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/container/array.hpp"
#include "cute/numeric/integral_constant.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/dependent_false.hpp"
#include <cutlass/pipeline/sm90_pipeline.hpp>


////////////////////////////////////////////////////////////////////////////////////////////////////

namespace trtllm::dev {

////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
//
// TMA load (producer) Async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////
// Assumptions : Constructor is visible Cluster-wide (as it needs a Cluster-Sync)
// We have exactly one thread elected in the Producer as the "leader"
// Currently, it is optional to elect a leader for the Consumers
template <int Stages_> class PipelineTmaAsync {
public:
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = cutlass::PipelineState<Stages>;

  enum class ThreadCategory { NonParticipant, Producer, Consumer, ProducerConsumer };

  struct Params {
    uint32_t transaction_bytes = 0;
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0; // Number of consumer threads
    uint32_t num_producers = 1; // Number of producer threads
    int initializing_warp = 0;
  };

  template <class ClusterShape>
  static CUTLASS_DEVICE void init_barriers(FullBarrier* full_barrier_ptr,
                                           EmptyBarrier* empty_barrier_ptr,
                                           Params params,
                                           ClusterShape cluster_shape) {
    int warp_idx = cutlass::canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == params.initializing_warp);
    if (is_initializing_warp) {
      uint32_t const producer_arv_cnt = params.num_producers;
      uint32_t const num_consumer_warpgroups_per_cluster =
        cute::ceil_div(params.num_consumers,
                       static_cast<uint32_t>(cutlass::NumThreadsPerWarpGroup));
      uint32_t multicast_consumer_arrival_count = params.num_consumers; // If cluster_size is 1
      if (cute::size(cluster_shape) > 1) {
        multicast_consumer_arrival_count =
          (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
          num_consumer_warpgroups_per_cluster;
      }
      CUTLASS_ASSERT(multicast_consumer_arrival_count > 0 &&
                     "Multicast consumer arrival count must be non-zero");
      CUTLASS_ASSERT(producer_arv_cnt > 0 && "Producer arrival count must be non-zero");
      cutlass::arch::detail::
        initialize_barrier_array_pair_aligned<FullBarrier*, EmptyBarrier*, Stages>(
          full_barrier_ptr,
          empty_barrier_ptr,
          producer_arv_cnt,
          multicast_consumer_arrival_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  template <class ClusterShape, class InitMasks = cute::true_type>
  CUTLASS_DEVICE PipelineTmaAsync(FullBarrier* full_barrier_ptr,
                                  EmptyBarrier* empty_barrier_ptr,
                                  Params params,
                                  ClusterShape cluster_shape,
                                  InitMasks = {})
    : params_(params)
    , full_barrier_ptr_(full_barrier_ptr)
    , empty_barrier_ptr_(empty_barrier_ptr) {

    static_assert(cute::is_same_v<InitMasks, cute::true_type> ||
                  cute::is_same_v<InitMasks, cute::false_type>);

    if constexpr (cute::is_same_v<InitMasks, cute::true_type>) {
      int warp_idx = cutlass::canonical_warp_idx_sync();
      int thread_idx = threadIdx.x;
      // Logic to optimally schedule Empty Arrives
      // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
      dim3 block_id = cute::block_id_in_cluster();
      auto cluster_size = cute::size(cluster_shape);

      if (cluster_size == 1) {
        is_signaling_thread_ = true;
        dst_blockid_ = 0;
      } else {
        // STEP 1 : Use Cute Layout function to generate an optimal dst block-id (0-15)
        if (params_.num_consumers % cutlass::NumThreadsPerWarpGroup == 0) {
          auto [is_signaling_thread, dst_blockid] = cutlass::detail::spread_arrivals_to_warpgroup(
            thread_idx % cutlass::NumThreadsPerWarpGroup,
            warp_idx);
          is_signaling_thread_ = is_signaling_thread;
          dst_blockid_ = dst_blockid;
        } else if (params_.num_consumers == 32) {
          auto [is_signaling_thread, dst_blockid] =
            cutlass::detail::spread_arrivals_to_warp(thread_idx % 32);
          is_signaling_thread_ = is_signaling_thread;
          dst_blockid_ = dst_blockid;
        } else {
          is_signaling_thread_ = 0;
#ifndef NDEBUG
          asm volatile("brkpt;\n" ::);
#endif
        }

        // STEP 2: Find if this dst block-id needs an arrival for this problem
        is_signaling_thread_ &= dst_blockid_ < cluster_size;
        is_signaling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);
      }
    }
  }

  template <class ClusterShape>
  CUTLASS_DEVICE bool is_same_row_or_col(int dst_block_id,
                                         dim3 block_id,
                                         ClusterShape cluster_shape) {
    return (((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x) ||
            (((dst_block_id / cute::size<0>(cluster_shape)) == block_id.y)));
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.

  CUTLASS_DEVICE
  cutlass::ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state) { producer_acquire(state.index(), state.phase()); }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, cutlass::ProducerToken barrier_token) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  template <class UserDefinedArriveOp>
  CUTLASS_DEVICE void producer_commit(PipelineState state,
                                      UserDefinedArriveOp&& user_defined_arrive_op) {
    cute::forward<UserDefinedArriveOp>(user_defined_arrive_op)(producer_get_barrier(state.index()));
    ;
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    cutlass::detail::pipeline_check_is_producer(params_.role);
    for (int count = 0; count < Stages; ++count) {
      empty_barrier_ptr_[state.index()].wait(state.phase());
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  CUTLASS_DEVICE
  void producer_expect_transaction(PipelineState state, uint32_t transaction_bytes) {
    producer_expect_transaction(state.index(), transaction_bytes);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state) { consumer_wait(state.index(), state.phase()); }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, cutlass::ConsumerToken barrier_token) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) { consumer_release(state.index()); }

private:
  uint32_t dst_blockid_ = 0;
  uint32_t is_signaling_thread_ = 0;
  FullBarrier* full_barrier_ptr_ = nullptr;
  EmptyBarrier* empty_barrier_ptr_ = nullptr;
  Params params_;

  CUTLASS_DEVICE
  cutlass::ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    cutlass::detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {cutlass::BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<cutlass::BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase) {
    empty_barrier_ptr_[stage].wait(phase);

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
#ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer ||
        params_.role == ThreadCategory::NonParticipant) {
      asm volatile("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile("brkpt;\n" ::);
    }
#endif
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, cutlass::ProducerToken barrier_token) {
    cutlass::detail::pipeline_check_is_producer(params_.role);
    if (barrier_token != cutlass::BarrierStatus::WaitDone) {
      empty_barrier_ptr_[stage].wait(phase);
    }

    if (params_.is_leader) {
      full_barrier_ptr_[stage].arrive_and_expect_tx(params_.transaction_bytes);
    }
#ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer ||
        params_.role == ThreadCategory::NonParticipant) {
      asm volatile("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile("brkpt;\n" ::);
    }
#endif
  }

  CUTLASS_DEVICE
  void producer_expect_transaction(uint32_t stage, uint32_t transaction_bytes) {
    cutlass::detail::pipeline_check_is_producer(params_.role);
    if (params_.is_leader) {
      full_barrier_ptr_[stage].expect_transaction(transaction_bytes);
    }
  }

  // NOP for TMA based mainloop
  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
// Below code is used only for unit-testing (in the absence of TMA commit)
#if CUTLASS_UNIT_TEST_PIPELINE
    if (params_.is_leader) {
      // STEP 1 : Commit to self
      full_barrier_ptr_[stage].complete_transaction(bytes);

      // STEP 2 : Commit to other blocks in our cluster
      auto cluster_shape = cute::cluster_shape();
      auto block_layout_in_cluster = cute::make_layout(cluster_shape);
      dim3 local_block_id = cute::block_id_in_cluster();

      CUTLASS_PRAGMA_UNROLL
      for (int n = 0; n < cute::size<1>(block_layout_in_cluster); ++n) {
        uint32_t dst_block_id = block_layout_in_cluster(local_block_id.x, n, cute::Int<0>{});
        full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, n != local_block_id.y);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < cute::size<0>(block_layout_in_cluster); ++m) {
        uint32_t dst_block_id = block_layout_in_cluster(m, local_block_id.y, cute::Int<0>{});
        full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, m != local_block_id.x);
      }
    }
#endif
  }

  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {cutlass::BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<cutlass::BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {cutlass::BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<cutlass::BarrierStatus>(barrier_status)};
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    full_barrier_ptr_[stage].wait(phase);
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, cutlass::ConsumerToken barrier_token) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t skip = false) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(dst_blockid_, is_signaling_thread_ & (!skip));
#ifndef NDEBUG
    if (params_.role == ThreadCategory::Producer ||
        params_.role == ThreadCategory::NonParticipant) {
      asm volatile("brkpt;\n" ::);
    }
#endif
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Simple producer-consumer async Pipeline class
//
///////////////////////////////////////////////////////////////////////////////////////////////////

template <int Stages_> class PipelineAsync {
public:
  static constexpr uint32_t Stages = Stages_;
  using FullBarrier = cutlass::arch::ClusterBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = typename FullBarrier::ValueType;
  using ConsumerBarrierType = typename EmptyBarrier::ValueType;
  using PipelineState = cutlass::PipelineState<Stages>;

  enum class ThreadCategory { NonParticipant, Producer, Consumer, ProducerConsumer };

  struct Params {
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t producer_arv_count = 1;
    uint32_t consumer_arv_count = 1;
    uint32_t dst_blockid = cute::block_rank_in_cluster();
    int initializing_warp = 0;
  };

  static CUTLASS_DEVICE void init_barriers(FullBarrier* full_barrier_ptr,
                                           EmptyBarrier* empty_barrier_ptr,
                                           Params params) {
    int warp_idx = cutlass::canonical_warp_idx_sync();
    bool is_initializing_warp = (warp_idx == params.initializing_warp);
    if (is_initializing_warp) {
      CUTLASS_ASSERT(params.producer_arv_count > 0 && "Producer arrival count must be non-zero");
      CUTLASS_ASSERT(params.consumer_arv_count > 0 && "Consumer arrival count must be non-zero");
      cutlass::arch::detail::
        initialize_barrier_array_pair_aligned<FullBarrier*, EmptyBarrier*, Stages>(
          full_barrier_ptr,
          empty_barrier_ptr,
          params.producer_arv_count,
          params.consumer_arv_count);
    }
    cutlass::arch::fence_barrier_init();
  }

  CUTLASS_DEVICE
  PipelineAsync(FullBarrier* full_barrier_ptr,
                EmptyBarrier* empty_barrier_ptr,
                Params const& params)
    : params_(params)
    , full_barrier_ptr_(full_barrier_ptr)
    , empty_barrier_ptr_(empty_barrier_ptr) {}

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait.
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.
  CUTLASS_DEVICE
  cutlass::ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
    return producer_try_acquire(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state,
                        cutlass::ProducerToken barrier_token = {
                          cutlass::BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state) { producer_commit(state.index()); }

  template <class UserDefinedArriveOp>
  CUTLASS_DEVICE void producer_commit(PipelineState state,
                                      UserDefinedArriveOp&& user_defined_arrive_op) {
    cute::forward<UserDefinedArriveOp>(user_defined_arrive_op)(producer_get_barrier(state.index()));
    producer_commit(state);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state);
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state) {
    return producer_get_barrier(state.index());
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_try_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_test_wait(PipelineState state, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state,
                     cutlass::ConsumerToken barrier_token = {cutlass::BarrierStatus::WaitAgain}) {
    consumer_wait(state.index(), state.phase(), barrier_token);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state) { consumer_release(state.index()); }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[stage]);
  }

private:
  Params params_;
  FullBarrier* full_barrier_ptr_;
  EmptyBarrier* empty_barrier_ptr_;

  CUTLASS_DEVICE
  cutlass::ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    cutlass::detail::pipeline_check_is_producer(params_.role);
    if (skip_wait) {
      return {cutlass::BarrierStatus::WaitDone};
    }
    bool barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<cutlass::BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, cutlass::ProducerToken barrier_token) {
    cutlass::detail::pipeline_check_is_producer(params_.role);
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      empty_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void producer_commit(uint32_t stage) {
    cutlass::detail::pipeline_check_is_producer(params_.role);
    full_barrier_ptr_[stage].arrive();
  }

  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {cutlass::BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].try_wait(phase);
    return {static_cast<cutlass::BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  cutlass::ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    if (skip_wait) {
      return {cutlass::BarrierStatus::WaitDone};
    }
    bool barrier_status = full_barrier_ptr_[stage].test_wait(phase);
    return {static_cast<cutlass::BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    bool done = full_barrier_ptr_[stage].test_wait(phase);
    if (!done) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, cutlass::ConsumerToken barrier_token) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    if (barrier_token == cutlass::BarrierStatus::WaitAgain) {
      full_barrier_ptr_[stage].wait(phase);
    }
  }

  CUTLASS_DEVICE
  void consumer_release(uint32_t stage) {
    cutlass::detail::pipeline_check_is_consumer(params_.role);
    empty_barrier_ptr_[stage].arrive(params_.dst_blockid);
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// Barrier to ensure an Ordered Sequence between
// SequenceLength number of groups (each with group_size participants) executing SequenceDepth
// Stages i.e., for all i < j - only after id "i" arrives at a particular stage "m" will the wait()
// for id "j" succeed for the same stage
//
///////////////////////////////////////////////////////////////////////////////////////////////////

namespace PipelineDetail {

template <int SequenceDepth, int SequenceLength> struct OrderedSequenceBarrierSharedStorage {
  using Barrier = cutlass::arch::ClusterBarrier;
  Barrier barrier_[SequenceDepth][SequenceLength];
};

} // namespace PipelineDetail

template <int SequenceDepth_, int SequenceLength_> class OrderedSequenceBarrier {
public:
  static constexpr int SequenceDepth = SequenceDepth_;
  static constexpr int SequenceLength = SequenceLength_;
  using SharedStorage =
    PipelineDetail::OrderedSequenceBarrierSharedStorage<SequenceDepth, SequenceLength>;
  using Barrier = typename SharedStorage::Barrier;

  struct Params {
    uint32_t group_id;
    uint32_t group_size;
    int initializing_warp = 0;
  };

private:
  // In future this Params object can be replaced easily with a CG object
  Params params_;
  Barrier* barrier_ptr_;
  cutlass::PipelineState<SequenceDepth> stage_;

  static constexpr int Depth = SequenceDepth;
  static constexpr int Length = SequenceLength;

public:
  OrderedSequenceBarrier() = delete;
  OrderedSequenceBarrier(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier(OrderedSequenceBarrier&&) = delete;
  OrderedSequenceBarrier& operator=(const OrderedSequenceBarrier&) = delete;
  OrderedSequenceBarrier& operator=(OrderedSequenceBarrier&&) = delete;
  ~OrderedSequenceBarrier() = default;

  static CUTLASS_DEVICE void init_barriers(Barrier* barrier_ptr, Params const& params) {
    CUTLASS_ASSERT(params.group_size > 0 && "Group size must be non-zero");
#if (__CUDA_ARCH__ >= 1000)
    int arv_cnt = params.group_size;
    constexpr int Stages = Depth * Length;
    cutlass::arch::detail::initialize_barrier_array_aligned<decltype(barrier_ptr), Stages>(
      barrier_ptr,
      arv_cnt);
#else
    for (int d = 0; d < Depth; ++d) {
      for (int l = 0; l < Length; ++l) {
        barrier_ptr[d * Length + l].init(params.group_size);
      }
    }
#endif
  }

  static CUTLASS_DEVICE void init_barriers(SharedStorage& storage, Params const& params) {
    init_barriers(&storage.barrier_[0][0], params);
  }

  template <typename InitBarriers = cute::true_type>
  CUTLASS_DEVICE OrderedSequenceBarrier(Barrier* barrier_ptr,
                                        Params const& params,
                                        InitBarriers = {})
    : params_(params)
    , barrier_ptr_(barrier_ptr)
    , stage_({0, params.group_id == 0, 0}) {

    static_assert(cute::is_same_v<InitBarriers, cute::true_type> ||
                  cute::is_same_v<InitBarriers, cute::false_type>);

    if constexpr (cute::is_same_v<InitBarriers, cute::true_type>) {
      int warp_idx = cutlass::canonical_warp_idx_sync();
#if (__CUDA_ARCH__ >= 1000)
      if (warp_idx == params.initializing_warp) {
        init_barriers(barrier_ptr_, params);
      }
#else
      int lane_predicate = cute::elect_one_sync();
      if (warp_idx == 0 && lane_predicate) {
        init_barriers(barrier_ptr_, params);
      }
#endif
      cutlass::arch::fence_barrier_init();
    }
  }

  CUTLASS_DEVICE
  OrderedSequenceBarrier(SharedStorage& storage, Params const& params)
    : OrderedSequenceBarrier(&storage.barrier_[0][0], params) {}

  // Wait on a stage to be unlocked
  CUTLASS_DEVICE
  void wait() { get_barrier_for_current_stage(params_.group_id).wait(stage_.phase()); }

  // Signal completion of Stage and move to the next stage
  // (group_id) signals to (group_id+1)
  CUTLASS_DEVICE
  void arrive() {
    int signalling_id = (params_.group_id + 1) % Length;
    get_barrier_for_current_stage(signalling_id).arrive();
    ++stage_;
  }

  CUTLASS_DEVICE
  void advance() { ++stage_; }

private:
  CUTLASS_DEVICE
  Barrier& get_barrier_for_current_stage(int group_id) {
    return barrier_ptr_[stage_.index() * Length + group_id];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Synchronization call. Blocks until barriers are initialized in shared memory.
CUTLASS_DEVICE
void pipeline_init_wait(int cluster_size) {
  if (cluster_size > 1) {
    cute::cluster_wait();
  } else {
    __syncthreads();
  }
}

// Used to guarantee that the Pipeline init is visible
// to all producers and consumer threadblocks in the cluster
CUTLASS_DEVICE
void pipeline_init_arrive_relaxed(int cluster_size) {
  if (cluster_size > 1) {
    cute::cluster_arrive_relaxed();
  } else {
    __syncthreads();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace trtllm::dev
