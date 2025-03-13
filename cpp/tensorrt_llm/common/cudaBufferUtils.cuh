/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaUtils.h"
#include <assert.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp8.h>
#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <type_traits>

namespace tensorrt_llm
{
namespace common
{
static __host__ __device__ int hash(int val)
{
    val ^= val >> 1;
    val ^= val >> 3;
    val ^= val >> 7;
    return val;
}

static int global_seed = 0xfeedcafe; // updates on every call to gen_random.

template <typename T>
__global__ void randomizeKernel(T* data, size_t sz, int seed)
{
    size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    using VType = typename UpperType<T>::Type;
    VType val;
    curandStateXORWOW_t state;
    curand_init(seed, sz, offset, &state);
    val = (VType) ((int32_t) curand(&state));
    if constexpr (std::is_same_v<VType, double>)
    {
        val /= (VType) (INT_MAX / 4);
    }
    if (offset < sz)
        data[offset] = (T) val;
}

template <typename T>
void gen_random(T* ptr, size_t num_elements)
{

    if constexpr (std::is_integral_v<T> || std::is_same_v<T, __nv_bfloat16> || std::is_same_v<T, float>
        || std::is_same_v<T, half>)
    {

        T* tmp;
        CUDA_CALL(cudaMalloc(&tmp, num_elements * sizeof(T)));

        int num_blocks = (num_elements + 1023) / 1024;
        int num_threads = 1024;

        randomizeKernel<<<num_blocks, num_threads>>>(tmp, num_elements, global_seed);

        CUDA_CALL(cudaMemcpy(ptr, tmp, num_elements * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaFree(tmp));

        global_seed = hash(global_seed);
    }
    else
    {
        std::cerr << "I don't know how to randomize " << typeid(T).name() << std::endl;
        assert(false);
    }
}

template <typename T>
class Buffer
{
public:
    Buffer(size_t num_elements, bool randomize = true)
    {
        _sz = num_elements * sizeof(T);
        CUDA_CALL(cudaMalloc(&_d_buffer, _sz));
        _h_buffer = static_cast<T*>(malloc(_sz));
        assert(_h_buffer);
        if (randomize)
            gen_random(_h_buffer, num_elements);
    }

    Buffer(Buffer const&) = delete;

    ~Buffer()
    {
        CUDA_CALL(cudaFree(_d_buffer));
        free(_h_buffer);
    }

    T* devicePtr()
    {
        return _d_buffer;
    }

    T* hostPtr()
    {
        return _h_buffer;
    }

    void sendToDevice(cudaStream_t stream = 0)
    {
        CUDA_CALL(cudaMemcpyAsync(_d_buffer, _h_buffer, _sz, cudaMemcpyHostToDevice, stream));
    }

    void readFromDevice(cudaStream_t stream = 0)
    {
        CUDA_CALL(cudaMemcpyAsync(_h_buffer, _d_buffer, _sz, cudaMemcpyDeviceToHost, stream));
    }

    void fill_host(const T val)
    {
        for (int i = 0; i < _sz / sizeof(T); i++)
        {
            _h_buffer[i] = val;
        }
    }

    void clear(cudaStream_t stream = 0)
    {
        memset(_h_buffer, 0, _sz);
        CUDA_CALL(cudaMemsetAsync(_d_buffer, 0, _sz, stream));
    }

    size_t hash()
    {
        std::hash<std::string> string_hash;
        return string_hash(std::string{(char*) _h_buffer, _sz});
    }

private:
    T* _d_buffer;
    T* _h_buffer;
    size_t _sz;
};

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

/* Producer class */
template <int DEPTH, int CGA_SIZE>
class CircularBufferWriter
{
protected:
    uint32_t _wptr;
    uint32_t _phase;
    Arrive_wait _entryConsumedBarriers;
    Arrive_wait _entryProducedBarriers;

public:
    __device__ CircularBufferWriter(CircularBufferBarriers<DEPTH>* barriers)
        : _entryProducedBarriers(barriers->entryProducedBarriers)
        , _entryConsumedBarriers(barriers->entryConsumedBarriers)
        , _wptr(0)
        , _phase(0xffffffff)
    {
    }

    __device__ int ptr()
    {
        return _wptr;
    }

    // Return the equivalent read phase.
    __device__ int phase()
    {
        return _phase ^ 0xffffffff;
    }

    /* Reserve space in the buffer for TMA */
    __device__ int tmaReserve(int transactioncnt, int tid0 = 1)
    {
        int ptr = threadReserve();
        _entryProducedBarriers.bar_arrive_set_transactioncnt(ptr, transactioncnt, tid0);
        return ptr;
    }

    /* Reserve space in the buffer for producer threads */
    __device__ int threadReserve()
    {
        wait();
        return advance();
    }

    __device__ int advance()
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

    __device__ void wait(int wptr, int phase)
    {
        // int ready = _entryConsumedBarriers.bar_peek(wptr, phase);
        // if (!ready)
        _entryConsumedBarriers.bar_wait(wptr, phase);
    }

    __device__ void wait(int wptr)
    {
        wait(wptr, _phase >> wptr);
    }

    __device__ int wait()
    {
        wait(_wptr);
        return _wptr;
    }

    /* Signal that data is ready */
    __device__ void threadCommit(int id)
    {
        _entryProducedBarriers.bar_arrive_normal(id);
    }

    __device__ int push()
    {
        int ptr = this->threadReserve();
        this->threadCommit(ptr);
        return ptr;
    }

    /* Get the barrier address, needed by TMA */
    __device__ uint64_t* barrier_ptr(int id)
    {
        return _entryProducedBarriers.get_bar_addr(id);
    }

    __device__ void setPtr(int ptr)
    {
        _wptr = ptr;
    }

    __device__ void setPhase(int phase)
    {
        _phase = phase;
    }
};

/* Consumer class */
template <int DEPTH, int CGA_SIZE>
class CircularBufferReader
{
protected:
    uint32_t _rptr;
    uint32_t _phase;

public:
    Arrive_wait _entryProducedBarriers;
    Arrive_wait _entryConsumedBarriers;

    __device__ CircularBufferReader(CircularBufferBarriers<DEPTH>* barriers)
        : _entryProducedBarriers(barriers->entryProducedBarriers)
        , _entryConsumedBarriers(barriers->entryConsumedBarriers)
        , _rptr(0)
        , _phase(0)
    {
    }

    __device__ void setProducerCta(int cta_id)
    {
        _entryConsumedBarriers.set_bar_base_dsmem(cta_id);
    }

    /* Peek at the head */
    __device__ int peek()
    {
        return _entryProducedBarriers.bar_peek(_rptr, _phase >> _rptr);
    }

    /* Wait for the head to be ready */
    __device__ int wait()
    {
        _entryProducedBarriers.bar_wait(_rptr, _phase >> _rptr);
        return _rptr;
    }

    /* Advance the head pointer */
    __device__ void advance()
    {
        _phase ^= (1 << _rptr);
        _rptr += 1;
        if (_rptr >= DEPTH)
        {
            _rptr = 0;
        }
    }

    __device__ int ptr()
    {
        return _rptr;
    }

    __device__ uint32_t phase()
    {
        return _phase;
    }

    /* Indicate consumption of data at specified pointer.  The producer is now free to overwrite it */
    __device__ void complete(int ptr)
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

    /* Simplification of complete and advance for cases where they don't need to be reordered/separated for performance
     */
    __device__ void pop()
    {
        complete(_rptr);
        advance();
    }

    /* Overrides for pointer and phase.  Used for shared buffers */
    __device__ void setPtr(int ptr)
    {
        _rptr = ptr;
    }

    __device__ void setPhase(uint32_t phase)
    {
        _phase = phase;
    }
};

template <int DEPTH, int CGA_SIZE = 1>
class CircularBuffer
{
protected:
    CircularBufferBarriers<DEPTH> _barriers;

public:
    __device__ void init(int tid0, int producer_thread_count, int consumer_thread_count)
    {
        if (tid0)
        {
            for (int i = 0; i < DEPTH; i++)
            {
                bar_create(&_barriers.entryProducedBarriers[i], producer_thread_count);
                bar_create(&_barriers.entryConsumedBarriers[i], consumer_thread_count);
            }
        }
    }

    using Reader = CircularBufferReader<DEPTH, CGA_SIZE>;
    using Writer = CircularBufferWriter<DEPTH, CGA_SIZE>;

    __device__ Reader createReader()
    {
        return Reader(&_barriers);
    }

    __device__ Writer createWriter()
    {
        return Writer(&_barriers);
    }

    __device__ int depth()
    {
        return DEPTH;
    }

    CircularBuffer() = default;
    // CircularBuffer must live in __shared__ -- cannot copy
    CircularBuffer(CircularBuffer const& other) = delete;
};

template <int DEPTH, typename T, int CGA_SIZE>
class CircularBufferWithDataReader : public CircularBufferReader<DEPTH, CGA_SIZE>
{
protected:
    T* _data;

public:
    using Base = CircularBufferReader<DEPTH, CGA_SIZE>;

    __device__ CircularBufferWithDataReader(CircularBufferBarriers<DEPTH>* barriers, T* data)
        : Base(barriers)
        , _data(data)
    {
    }

    __device__ T read()
    {
        return _data[this->ptr()];
    }

    __device__ T pop(bool read_data = true)
    {
        T rval;
        // int ready = this->peek();
        // if (!ready)
        this->wait();
        if (read_data)
        {
            rval = read();
            fence_view_async_shared();
        }
        this->complete(this->ptr());
        this->advance();
        return rval;
    }
};

template <int DEPTH, typename T, int CGA_SIZE>
class CircularBufferWithDataWriter : public CircularBufferWriter<DEPTH, CGA_SIZE>
{
protected:
    T* _data;

public:
    using Base = CircularBufferWriter<DEPTH, CGA_SIZE>;

    __device__ CircularBufferWithDataWriter(CircularBufferBarriers<DEPTH>* barriers, T* data)
        : Base(barriers)
        , _data(data)
    {
    }

    __device__ void write(int ptr, T const& wrdat)
    {
        _data[ptr] = wrdat;
    }

    __device__ int push(T const& wrdat, bool writeData = true)
    {
        int ptr = this->threadReserve();
        if (writeData)
        {
            write(ptr, wrdat);
            __threadfence_block();
        }
        this->threadCommit(ptr);
        return ptr;
    }

    template <bool NEED_EXPLICIT_COMMITMENT = false>
    __device__ void push_to_cta(T const& wrdat, int cta_id, int offset)
    {
        if constexpr (CGA_SIZE == 1)
        {
            write(offset, wrdat);
            if constexpr (!NEED_EXPLICIT_COMMITMENT)
            {
                __threadfence_block();
                this->threadCommit(offset);
            }
        }
        else
        {

            uint64_t* bar_ptr = this->barrier_ptr(offset);
            stas<!NEED_EXPLICIT_COMMITMENT>(&_data[offset], bar_ptr, cta_id, wrdat);
        }
    }

    template <bool NEED_EXPLICIT_COMMITMENT = false, int SKIP_CTA_ID = -1>
    __device__ int broadcast(T const& wrdat)
    {
        int offset = this->threadReserve();
        for (int i = 0; i < CGA_SIZE; i++)
        {
            if constexpr (SKIP_CTA_ID != -1)
            {
                if (i == SKIP_CTA_ID)
                {
                    continue;
                }
            }
            push_to_cta<NEED_EXPLICIT_COMMITMENT>(wrdat, i, offset);
        }
        return offset;
    }

    __device__ void commit(int ptr, int ctaid)
    {
        if constexpr (CGA_SIZE == 1)
        {
            __threadfence_block();
            this->threadCommit(ptr);
        }
        else
        {
            // Set transaction cnt after the data transmission.
            uint64_t* bar_ptr = this->barrier_ptr(ptr);
            arrive_DSMEM_barrier_and_set_tx_cnt(bar_ptr, ctaid, sizeof(T));
        }
    }

    template <int SKIP_CTA_ID = -1>
    __device__ void commit(int ptr)
    {
        for (int i = 0; i < CGA_SIZE; i++)
        {
            if constexpr (SKIP_CTA_ID != -1)
            {
                if (i == SKIP_CTA_ID)
                {
                    continue;
                }
            }
            commit(ptr, i);
        }
    }
};

template <int DEPTH, typename T, int CTAS_PER_CGA = 1>
class CircularBufferWithData : public CircularBuffer<DEPTH, CTAS_PER_CGA>
{
protected:
    T _data[DEPTH];

public:
    __device__ T* data()
    {
        return _data;
    }

    using Reader = CircularBufferWithDataReader<DEPTH, T, CTAS_PER_CGA>;
    using Writer = CircularBufferWithDataWriter<DEPTH, T, CTAS_PER_CGA>;

    __device__ Reader createReader()
    {
        return Reader(&this->_barriers, _data);
    }

    __device__ Writer createWriter()
    {
        return Writer(&this->_barriers, _data);
    }

    CircularBufferWithData() = default;
    // Must live in __shared__ -- cannot copy
    CircularBufferWithData(CircularBufferWithData const& other) = delete;
};

__device__ __forceinline__ void namedBarrierSync(int name, int numThreads)
{
    asm volatile("bar.sync %0, %1;" : : "r"(name), "r"(numThreads) : "memory");
}

// Multi Producer, Single Consumer FIFO for Completer.
template <int DEPTH, int CTAS_PER_CGA>
struct MultiProducerCircularBuffer : public CircularBuffer<DEPTH, CTAS_PER_CGA>
{

    using Base = CircularBuffer<DEPTH, CTAS_PER_CGA>;

    struct Reader : public Base::Reader
    {
        using Base = typename Base::Reader;

        __device__ Reader(CircularBufferBarriers<DEPTH>* barriers)
            : Base(barriers)
        {
        }

        __device__ void setProducerCta(int) = delete;

        __device__ void complete(int ptr)
        {
            // Signal all producers.
            if constexpr (CTAS_PER_CGA == 1)
            {
                Base::_entryConsumedBarriers.bar_arrive_normal(ptr);
            }
            else
            {
                for (int i = 0; i < CTAS_PER_CGA; i++)
                {
                    Base::_entryConsumedBarriers.set_bar_base_dsmem(i);
                    Base::_entryConsumedBarriers.bar_arrive_dsmem(ptr);
                }
            }
        }

        __device__ void pop()
        {
            complete(this->_rptr);
            Base::advance();
        }
    };

    struct Writer : public Base::Writer
    {
        using Base = typename Base::Writer;

        __device__ Writer(CircularBufferBarriers<DEPTH>* barriers)
            : Base(barriers)
        {
        }

        __device__ void setConsumerCta(int cta_id)
        {
            if constexpr (CTAS_PER_CGA > 1)
            {
                Base::_entryProducedBarriers.set_bar_base_dsmem(cta_id);
            }
        }

        __device__ void threadCommit(int id)
        {
            if constexpr (CTAS_PER_CGA == 1)
            {
                Base::_entryProducedBarriers.bar_arrive_normal(id);
            }
            else
            {
                Base::_entryProducedBarriers.bar_arrive_dsmem(id);
            }
        }

        __device__ int push()
        {
            int ptr = this->threadReserve();
            threadCommit(ptr);
            return ptr;
        }
    };

    __device__ Reader createReader()
    {
        return Reader(&this->_barriers);
    }

    __device__ Writer createWriter()
    {
        return Writer(&this->_barriers);
    }
};

} // namespace common
} // namespace tensorrt_llm
