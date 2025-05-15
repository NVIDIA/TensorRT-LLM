/*
 * SPDX-FileCopyrightText: Copyright (c) 2011-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <fmha/hopper/arrive_wait.h>
#include <fmha/utils.h>
#include <stdint.h>

#pragma once

namespace fmha
{
namespace ws
{

////////////////////////////////////////////////////////////////////////////////////////////////////

/* Shared storage for barriers needed by both producer and consumer */
template <int DEPTH>
struct CircularBufferBarriers
{
    __align__(8) uint64_t entryProducedBarriers[DEPTH];
    __align__(8) uint64_t entryConsumedBarriers[DEPTH];

    CircularBufferBarriers() = default;
    // CircularBufferBarriers must live in __shared__ -- cannot copy
    CircularBufferBarriers(CircularBufferBarriers const& other) = delete;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* Producer class */
template <int DEPTH, int CGA_SIZE>
class CircularBufferWriter
{
protected:
    uint32_t _wptr;
    uint32_t _phase;
    fmha::Arrive_wait _entryConsumedBarriers;
    fmha::Arrive_wait _entryProducedBarriers;

public:
    inline __device__ CircularBufferWriter(CircularBufferBarriers<DEPTH>* barriers)
        : _entryProducedBarriers(barriers->entryProducedBarriers)
        , _entryConsumedBarriers(barriers->entryConsumedBarriers)
        , _wptr(0)
        , _phase(0xffffffff)
    {
    }

    inline __device__ int ptr()
    {
        return _wptr;
    }

    // Return the equivalent read phase.
    inline __device__ int phase()
    {
        return _phase ^ 0xffffffff;
    }

    /* Reserve space in the buffer for TMA */
    inline __device__ int tmaReserve(int tid0, int transactioncnt)
    {
        int ptr = threadReserve();
        _entryProducedBarriers.bar_arrive_set_transactioncnt(ptr, transactioncnt, tid0);
        return ptr;
    }

    /* Reserve space in the buffer for producer threads */
    inline __device__ int threadReserve()
    {
        wait();
        return advance();
    }

    inline __device__ int advance()
    {
        int rval = _wptr;
        _phase ^= (1 << _wptr);
        _wptr += 1;
        if (_wptr >= DEPTH)
        {
            _wptr = 0;
        }
        return rval;
    }

    /* Wait for space to become available in the buffer */
    inline __device__ int wait()
    {
        int ready = _entryConsumedBarriers.bar_peek(_wptr, (_phase >> _wptr) & 1);
        if (!ready)
            _entryConsumedBarriers.bar_wait(_wptr, (_phase >> _wptr) & 1);
        return _wptr;
    }

    /* Signal that data is ready */
    inline __device__ void threadCommit(int tid0, int id)
    {
        if (tid0)
        {
            _entryProducedBarriers.bar_arrive_normal(id);
        }
    }

    /* Get the barrier address, needed by TMA */
    inline __device__ uint64_t* barrier_ptr(int id)
    {
        return _entryProducedBarriers.get_bar_addr(id);
    }

    inline __device__ void setPtr(int ptr)
    {
        _wptr = ptr;
    }

    inline __device__ void setPhase(int phase)
    {
        _phase = phase;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* Consumer class */
template <int DEPTH, int CGA_SIZE>
class CircularBufferReader
{
private:
    uint32_t _rptr;
    uint32_t _phase;

public:
    fmha::Arrive_wait _entryProducedBarriers;
    fmha::Arrive_wait _entryConsumedBarriers;

    inline __device__ CircularBufferReader(CircularBufferBarriers<DEPTH>* barriers)
        : _entryProducedBarriers(barriers->entryProducedBarriers)
        , _entryConsumedBarriers(barriers->entryConsumedBarriers)
        , _rptr(0)
        , _phase(0)
    {
    }

    inline __device__ void setProducerCta(int cta_id)
    {
        _entryConsumedBarriers.set_bar_base_dsmem(cta_id);
    }

    /* Peek at the head */
    inline __device__ int peek()
    {
        return _entryProducedBarriers.bar_peek(_rptr, (_phase >> _rptr) & 1);
    }

    /* Wait for the head to be ready */
    inline __device__ int wait()
    {
        _entryProducedBarriers.bar_wait(_rptr, (_phase >> _rptr) & 1);
        return _rptr;
    }

    /* Advance the head pointer */
    inline __device__ void advance()
    {
        _phase ^= (1 << _rptr);
        _rptr += 1;
        if (_rptr >= DEPTH)
        {
            _rptr = 0;
        }
    }

    inline __device__ int ptr()
    {
        return _rptr;
    }

    inline __device__ uint32_t phase()
    {
        return _phase;
    }

    /* Indicate consumption of data at specified pointer.
    The producer is now free to overwrite it
    */
    inline __device__ void complete(int tid0, int ptr)
    {
        if (tid0)
        {
            if (CGA_SIZE > 1)
            {
                _entryConsumedBarriers.bar_arrive_dsmem(ptr);
            }
            else
            {
                _entryConsumedBarriers.bar_arrive_normal(ptr);
            }
        }
    }

    /* Simplification of complete and advance for cases
       where they don't need to be reordered/separated for performance
    */
    inline __device__ void pop(int tid0)
    {
        complete(tid0, _rptr);
        advance();
    }

    /* Overrides for pointer and phase.  Used for shared buffers */
    inline __device__ void setPtr(int ptr)
    {
        _rptr = ptr;
    }

    inline __device__ void setPhase(uint32_t phase)
    {
        _phase = phase;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DEPTH, int CGA_SIZE = 1>
class CircularBuffer
{
protected:
    CircularBufferBarriers<DEPTH> _barriers;

public:
    inline __device__ void init(int tid0, int producer_thread_count, int consumer_thread_count)
    {
        if (tid0)
        {
            for (int i = 0; i < DEPTH; i++)
            {
                fmha::bar_create(&_barriers.entryProducedBarriers[i], producer_thread_count);
                fmha::bar_create(&_barriers.entryConsumedBarriers[i], consumer_thread_count);
            }
        }
    }

    using Reader = CircularBufferReader<DEPTH, CGA_SIZE>;
    using Writer = CircularBufferWriter<DEPTH, CGA_SIZE>;

    inline __device__ Reader createReader()
    {
        return Reader(&_barriers);
    }

    inline __device__ Writer createWriter()
    {
        return Writer(&_barriers);
    }

    inline __device__ int depth()
    {
        return DEPTH;
    }

    CircularBuffer() = default;
    // CircularBuffer must live in __shared__ -- cannot copy
    CircularBuffer(CircularBuffer const& other) = delete;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DEPTH, typename T, int CGA_SIZE>
class CircularBufferWithDataReader : public CircularBufferReader<DEPTH, CGA_SIZE>
{
private:
    T* _data;

public:
    inline __device__ CircularBufferWithDataReader(CircularBufferBarriers<DEPTH>* barriers, T* data)
        : CircularBufferReader<DEPTH, CGA_SIZE>(barriers)
        , _data(data)
    {
    }

    inline __device__ T read()
    {
        return _data[this->ptr()];
    }

    inline __device__ T pop(int tid0, bool read_data = true)
    {
        T rval;
        int ready = this->peek();
        if (!ready)
            this->wait();
        if (read_data)
        {
            rval = read();
            fmha::fence_view_async_shared();
        }
        this->complete(tid0, this->ptr());
        this->advance();
        return rval;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DEPTH, typename T, int CGA_SIZE>
class CircularBufferWithDataWriter : public CircularBufferWriter<DEPTH, CGA_SIZE>
{
private:
    T* _data;

public:
    inline __device__ CircularBufferWithDataWriter(CircularBufferBarriers<DEPTH>* barriers, T* data)
        : CircularBufferWriter<DEPTH, CGA_SIZE>(barriers)
        , _data(data)
    {
    }

    inline __device__ void write(int ptr, T const& wrdat)
    {
        _data[ptr] = wrdat;
    }

    inline __device__ int push(int tid0, T const& wrdat, bool writeData = true, uint32_t transactioncnt = 0)
    {
        int ptr = this->threadReserve();
        if (tid0 && writeData)
        {
            write(ptr, wrdat);
            __threadfence_block();
        }
        if (transactioncnt == 0)
            this->threadCommit(tid0, ptr);
        else
            this->_entryProducedBarriers.bar_arrive_set_transactioncnt(ptr, transactioncnt, tid0);
        return ptr;
    }

    template <int SYNC_BAR, int SYNC_THREADS>
    inline __device__ int push_with_sync(int tid0, T const& wrdat, bool writeData = true, uint32_t transactioncnt = 0)
    {
        int ptr = this->threadReserve();
        named_barrier_wait(SYNC_BAR, SYNC_THREADS);
        if (tid0 && writeData)
        {
            write(ptr, wrdat);
            __threadfence_block();
        }
        if (transactioncnt == 0)
            this->threadCommit(tid0, ptr);
        else
            this->_entryProducedBarriers.bar_arrive_set_transactioncnt(ptr, transactioncnt, tid0);
        return ptr;
    }

    inline __device__ void broadcast(T const& wrdat)
    {
        int offset = this->threadReserve();
        for (int i = 0; i < CGA_SIZE; i++)
        {
            push_to_cta(wrdat, i, offset);
        }
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int DEPTH, typename T, int CTAS_PER_CGA = 1>
class CircularBufferWithData : public CircularBuffer<DEPTH, CTAS_PER_CGA>
{
private:
    T _data[DEPTH];

public:
    inline __device__ T* data()
    {
        return _data;
    }

    using Reader = CircularBufferWithDataReader<DEPTH, T, CTAS_PER_CGA>;
    using Writer = CircularBufferWithDataWriter<DEPTH, T, CTAS_PER_CGA>;

    inline __device__ Reader createReader()
    {
        return Reader(&this->_barriers, _data);
    }

    inline __device__ Writer createWriter()
    {
        return Writer(&this->_barriers, _data);
    }

    CircularBufferWithData() = default;
    // Must live in __shared__ -- cannot copy
    CircularBufferWithData(CircularBufferWithData const& other) = delete;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct OrderedMutex
{
    uint64_t barriers[2];

    inline __device__ void init(int tid0, int threads0, int threads1)
    {
        if (tid0)
        {
            fmha::bar_create(&barriers[0], threads0);
            fmha::bar_create(&barriers[1], threads1);
        }
    }
};

class OrderedMutexAccessor
{
private:
    int _phase;
    int _id;
    int _barrier_id;

    fmha::Arrive_wait _barriers;

public:
    inline __device__ OrderedMutexAccessor(OrderedMutex& m, int id, int barrier_id)
        : _phase(0)
        , _id(id)
        , _barriers(m.barriers)
        , _barrier_id(barrier_id)
    {
    }

    inline __device__ void arrive()
    {
        _barriers.bar_arrive_normal(_id);
    }

    inline __device__ void wait()
    {
        int ready = _barriers.bar_peek(_id ^ 1, _phase);
        if (!ready)
        {
            _barriers.bar_wait(_id ^ 1, _phase);
        }
        _phase ^= 1;
    }

    inline __device__ void named_bar_arrive()
    {
        // ...
        // Softmax ends
        // Make sure barrier is not moving around
        if (_id == 0)
        {
            named_barrier_wait(_barrier_id, 256);
        }
    }

    inline __device__ void named_bar_wait()
    {
        // Make sure barrier is not moving around
        if (_id == 1)
        {
            named_barrier_wait(_barrier_id, 256);
        }
        // Softmax starts
        // ...
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ComputeGroupBarrier
{
    uint64_t barrier;

    inline __device__ void init(int tid0, int threads)
    {
        if (tid0)
        {
            fmha::bar_create(&barrier, threads);
        }
    }
};

class ComputeGroupBarrierAccessor
{
private:
    int _phase;
    fmha::Arrive_wait _barrier;

public:
    inline __device__ ComputeGroupBarrierAccessor(ComputeGroupBarrier& m)
        : _phase(0)
        , _barrier(&m.barrier)
    {
    }

    inline __device__ void arrive()
    {
        _barrier.bar_arrive_normal(0);
    }

    inline __device__ void wait()
    {
        int ready = _barrier.bar_peek(0, _phase);
        if (!ready)
        {
            _barrier.bar_wait(0, _phase);
        }
        _phase ^= 1;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace ws
} // namespace fmha
