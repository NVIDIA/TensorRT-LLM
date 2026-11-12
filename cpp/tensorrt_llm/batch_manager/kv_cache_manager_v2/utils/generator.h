/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/assert.h"

#include <coroutine>
#include <cstddef>
#include <exception>
#include <iterator>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Generator — a coroutine that yields values, optionally receiving one back.
//
// `Generator<T>` is the familiar lazy sequence, close enough to C++23's
// std::generator to be replaced by it later: nothing runs until the first
// iteration, and iterating resumes the body to its next `co_yield`.
//
//     Generator<int> evens(int n) { for (int i = 0; i < n; i += 2) co_yield i; }
//     for (int v : evens(10)) { ... }
//
// `Generator<T, Send>` additionally carries a value back in, because `co_yield v`
// is `co_await promise.yield_value(v)` and so has a value of its own:
//
//     Generator<size_t, int32_t*> layout() {
//         int32_t* base = co_yield elements;   // yields a count, resumes with memory
//         ...
//     }
//
// That direction is what lets a caller collect what several bodies need, satisfy
// them all at once, and let each continue -- which a plain sequence of calls
// cannot express, since the first would have to be satisfied before the last had
// been asked. Each body stays one function, so what its first half computed is
// still in scope for its second: the values live in the coroutine frame.
//
// `Generator<T, Send, Return>` additionally carries a result out of `co_return`, so a
// body that was a function returning Return stays one:
//
//     Generator<SlotCount, Slots, bool> resize(...) {
//         for (lc ...) { auto slots = co_yield needed[lc]; ... }
//         co_return true;
//     }
//
// A body may yield any number of times; each suspension is a separate request, and
// value()/send() service them one at a time. That is what lets a caller drive several
// bodies round by round -- collecting every body's request for one life cycle, meeting
// them together, and only then letting each continue -- which is how allocations across
// bodies can be coalesced, or what one body released reused by another. A driver that
// expects a single round should check done() afterwards rather than assume it.
//
// A body's parameters are copied into its frame, so they are safe to hold across a
// suspension. A lambda's captures are not: the frame only refers to the closure, so an
// immediately-invoked capturing lambda leaves the frame pointing at a destroyed object.
// Write such a body as a function taking what it needs.
//
// Not std::generator: no recursive `elements_of`, no allocator customization, and
// the iterator is single-pass. Move-only, and an exception from the body is
// rethrown by whichever call resumed it.
// ---------------------------------------------------------------------------
namespace detail
{

//! Supplies the promise with return_value or return_void, which a promise may have one
//! of but never both. Both clear the pending value, so a finished body is told apart
//! from one suspended on a request by asking for that value.
template <typename Promise, typename Return>
struct GeneratorReturn
{
    void return_value(Return result)
    {
        auto* self = static_cast<Promise*>(this);
        self->mValue.reset();
        mReturn = std::move(result);
    }

    std::optional<Return> mReturn;
};

template <typename Promise>
struct GeneratorReturn<Promise, void>
{
    void return_void() noexcept
    {
        static_cast<Promise*>(this)->mValue.reset();
    }
};

} // namespace detail

template <typename T, typename Send = void, typename Return = void>
class Generator
{
    static constexpr bool kSends = !std::is_void_v<Send>;
    static constexpr bool kReturns = !std::is_void_v<Return>;

public:
    struct promise_type : detail::GeneratorReturn<promise_type, Return>
    {
        Generator get_return_object()
        {
            return Generator{std::coroutine_handle<promise_type>::from_promise(*this)};
        }

        //! Lazy, as std::generator is: the body runs on the first pull, not on the call.
        [[nodiscard]] std::suspend_always initial_suspend() const noexcept
        {
            return {};
        }

        //! Suspends at the end so the frame outlives the body, which is what lets a
        //! caller read a final exception out of the promise.
        [[nodiscard]] std::suspend_always final_suspend() const noexcept
        {
            return {};
        }

        void unhandled_exception() noexcept
        {
            mError = std::current_exception();
        }

        auto yield_value(T value) noexcept
        {
            mValue = std::move(value);
            if constexpr (kSends)
            {
                struct Resumption
                {
                    promise_type* promise;

                    [[nodiscard]] bool await_ready() const noexcept
                    {
                        return false;
                    }

                    void await_suspend(std::coroutine_handle<>) const noexcept {}

                    [[nodiscard]] Send await_resume() const
                    {
                        // Resumed without a value: the body would read whatever this
                        // returned and carry on as if it were real.
                        TLLM_CHECK_WITH_INFO(
                            promise->mSent.has_value(), "A two-way generator was resumed without being sent a value");
                        return std::move(*promise->mSent);
                    }
                };

                return Resumption{this};
            }
            else
            {
                return std::suspend_always{};
            }
        }

        std::optional<T> mValue;
        std::exception_ptr mError;
        // Only the two-way form carries one; std::monostate keeps the one-way form from
        // paying for it.
        std::conditional_t<kSends, std::optional<std::remove_reference_t<Send>>, std::monostate> mSent;
    };

    explicit Generator(std::coroutine_handle<promise_type> handle)
        : mHandle(handle)
    {
    }

    Generator(Generator&& other) noexcept
        : mHandle(std::exchange(other.mHandle, {}))
        , mStarted(std::exchange(other.mStarted, false))
    {
    }

    Generator(Generator const&) = delete;
    Generator& operator=(Generator const&) = delete;
    Generator& operator=(Generator&&) = delete;

    //! Destroys the frame, running the destructors of a suspended body's locals.
    ~Generator()
    {
        if (mHandle)
        {
            mHandle.destroy();
        }
    }

    //! Whether the body has run to completion.
    [[nodiscard]] bool done() const
    {
        _rethrow();
        return !mHandle || mHandle.done();
    }

    //! The value the body is suspended on, priming it if it has not started.
    //! Empty once the body has returned.
    [[nodiscard]] std::optional<T> const& value()
    {
        _prime();
        return mHandle.promise().mValue;
    }

    //! What the body returned, once it has finished. Empty until then.
    //!
    //! A template so that the signature is formed only where it is called. A
    //! requires-clause gates overload resolution, not the declaration, so a void Return
    //! would otherwise reach std::optional here and make the class ill-formed.
    template <typename R = Return>
    [[nodiscard]] std::optional<R> const& result() const requires(kReturns)
    {
        _rethrow();
        return mHandle.promise().mReturn;
    }

    //! Resume a one-way body. Available only when nothing is sent back.
    void next() requires(!kSends)
    {
        _prime();
        _step();
    }

    //! Hand the body a value and resume it. The `co_yield` it was suspended on
    //! evaluates to `sent`.
    //!
    //! A template for the same reason as result(), with S kept out of deduction so the
    //! parameter stays exactly Send rather than whatever the caller passed.
    template <typename S = Send>
    void send(std::type_identity_t<S> sent) requires(kSends)
    {
        _prime();
        if (mHandle.done())
        {
            return;
        }
        mHandle.promise().mSent = std::move(sent);
        _step();
    }

    // ---- one-way iteration, so `for (auto v : gen)` works -------------------

    class Iterator
    {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = T;
        using difference_type = std::ptrdiff_t;

        explicit Iterator(Generator* owner) noexcept
            : mOwner(owner)
        {
        }

        T const& operator*() const
        {
            return *mOwner->mHandle.promise().mValue;
        }

        Iterator& operator++()
        {
            mOwner->_step();
            if (mOwner->mHandle.done())
            {
                mOwner = nullptr;
            }
            return *this;
        }

        void operator++(int)
        {
            ++*this;
        }

        bool operator==(std::default_sentinel_t) const noexcept
        {
            return mOwner == nullptr;
        }

    private:
        Generator* mOwner;
    };

    [[nodiscard]] Iterator begin() requires(!kSends)
    {
        _prime();
        return Iterator{mHandle.done() ? nullptr : this};
    }

    [[nodiscard]] std::default_sentinel_t end() const noexcept requires(!kSends)
    {
        return {};
    }

private:
    //! Run to the first yield. Idempotent, so a caller need not track whether the body
    //! has started.
    void _prime()
    {
        if (mHandle && !mStarted)
        {
            mStarted = true;
            mHandle.resume();
            _rethrow();
        }
    }

    void _step()
    {
        if (mHandle && !mHandle.done())
        {
            mHandle.resume();
            _rethrow();
        }
    }

    void _rethrow() const
    {
        if (mHandle && mHandle.promise().mError)
        {
            std::rethrow_exception(mHandle.promise().mError);
        }
    }

    std::coroutine_handle<promise_type> mHandle;
    bool mStarted = false;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
