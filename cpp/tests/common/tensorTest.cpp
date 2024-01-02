/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>

#include "tensorrt_llm/common/tensor.h"

using namespace tensorrt_llm::common;

namespace
{

#define EXPECT_EQUAL_TENSORS(t1, t2)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        EXPECT_TRUE(t1.where == t2.where);                                                                             \
        EXPECT_TRUE(t1.type == t2.type);                                                                               \
        EXPECT_TRUE(t1.shape == t2.shape);                                                                             \
        EXPECT_TRUE(t1.data == t2.data);                                                                               \
    } while (false)

TEST(TensorMapTest, HasKeyCorrectness)
{
    bool* v1 = new bool(true);
    float* v2 = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{MEMORY_CPU, TYPE_BOOL, {1}, v1};
    Tensor t2 = Tensor{MEMORY_CPU, TYPE_FP32, {3, 2}, v2};

    TensorMap map({{"t1", t1}, {"t2", t2}});
    EXPECT_TRUE(map.contains("t1"));
    EXPECT_TRUE(map.contains("t2"));
    EXPECT_FALSE(map.contains("t3"));

    delete v1;
    delete[] v2;
}

TEST(TensorMapTest, InsertCorrectness)
{
    int* v1 = new int[4]{1, 10, 20, 30};
    float* v2 = new float[2]{1.0f, 2.0f};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, {2}, v2);

    TensorMap map({{"t1", t1}});
    EXPECT_TRUE(map.size() == 1);
    EXPECT_TRUE(map.contains("t1"));
    EXPECT_EQUAL_TENSORS(map.at("t1"), t1);
    EXPECT_FALSE(map.contains("t2"));

    delete[] v1;
    delete[] v2;
}

TEST(TensorMapTest, InsertDoesNotAllowNoneTensor)
{
    TensorMap map;
    EXPECT_TRUE(map.size() == 0);
    // forbid a none tensor.
    EXPECT_THROW(map.insert("none", {}), std::runtime_error);

    // forbid a tensor having null data pointer.
    Tensor none_data_tensor = Tensor(MEMORY_CPU, TYPE_INT32, {}, nullptr);
    EXPECT_THROW(map.insert("empty", none_data_tensor), std::runtime_error);
}

TEST(TensorMapTest, InsertDoesNotAllowDuplicatedKey)
{
    int* v1 = new int[4]{1, 10, 20, 30};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, {2}, v1);
    TensorMap map({{"t1", t1}});
    EXPECT_TRUE(map.size() == 1);
    // forbid a duplicated key.
    EXPECT_THROW(map.insert("t1", t2), std::runtime_error);
    delete[] v1;
}

TEST(TensorMapTest, GetValCorrectness)
{
    int* v1 = new int[4]{1, 10, 20, 30};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);

    TensorMap map({{"t1", t1}});
    EXPECT_TRUE(map.size() == 1);
    // throw exception since the map doesn't have a key "t3".
    EXPECT_THROW(map.getVal<int>("t3"), std::runtime_error);
    EXPECT_TRUE(map.getVal<int>("t1") == 1);
    EXPECT_TRUE(map.getVal<int>("t1", 3) == 1);

    // map doesn't have t2 so return the default value 3.
    EXPECT_TRUE(map.getVal<int>("t2", 3) == 3);

    v1[0] += 1; // update value.
    EXPECT_TRUE(map.getVal<int>("t1") == 2);
    EXPECT_TRUE(map.getVal<int>("t1", 3) == 2);

    size_t index = 2;
    EXPECT_TRUE(map.getValWithOffset<int>("t1", index) == 20);
    EXPECT_TRUE(map.getValWithOffset<int>("t1", index, 3) == 20);
    EXPECT_TRUE(map.getValWithOffset<int>("t2", index, 3) == 3);
    delete[] v1;
}

TEST(TensorMapTest, GetTensorCorrectness)
{
    bool* t1_val = new bool(true);
    float* t2_val = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{MEMORY_CPU, TYPE_BOOL, {1}, t1_val};
    Tensor t2 = Tensor{MEMORY_CPU, TYPE_FP32, {3, 2}, t2_val};

    int* default_val = new int[4]{0, 1, 2, 3};
    Tensor default_tensor = Tensor{MEMORY_CPU, TYPE_INT32, {4}, default_val};

    TensorMap map({{"t1", t1}, {"t2", t2}});
    EXPECT_THROW(map.at("t3"), std::runtime_error);
    EXPECT_EQUAL_TENSORS(map.at("t1", default_tensor), t1);
    EXPECT_EQUAL_TENSORS(map.at("t2", default_tensor), t2);
    EXPECT_EQUAL_TENSORS(map.at("t3", default_tensor), default_tensor);
    EXPECT_EQUAL_TENSORS(map.at("t3", {}), Tensor());

    delete[] default_val;
    delete[] t2_val;
    delete t1_val;
}

TEST(TensorMapTest, GetTensorCorrectnessAtConstTensorMap)
{
    bool* t1_val = new bool(true);
    float* t2_val = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{MEMORY_CPU, TYPE_BOOL, {1}, t1_val};
    Tensor t2 = Tensor{MEMORY_CPU, TYPE_FP32, {3, 2}, t2_val};

    int* default_val = new int[4]{0, 1, 2, 3};
    Tensor default_tensor = Tensor{MEMORY_CPU, TYPE_INT32, {4}, default_val};

    const TensorMap map({{"t1", t1}, {"t2", t2}});
    EXPECT_THROW(map.at("t3"), std::runtime_error);
    EXPECT_EQUAL_TENSORS(map.at("t1", default_tensor), t1);
    EXPECT_EQUAL_TENSORS(map.at("t2", default_tensor), t2);
    EXPECT_EQUAL_TENSORS(map.at("t3", default_tensor), default_tensor);
    EXPECT_EQUAL_TENSORS(map.at("t3", {}), Tensor());

    delete[] default_val;
    delete[] t2_val;
    delete t1_val;
}

TEST(TensorTest, EmptyTensorMinMaxRaiseError)
{
    Tensor t1;
    EXPECT_THROW(t1.min<int>(), std::runtime_error);
    EXPECT_THROW(t1.max<int>(), std::runtime_error);

    Tensor t2 = Tensor{MEMORY_CPU, TYPE_INT32, {}, nullptr};
    EXPECT_THROW(t2.min<int>(), std::runtime_error);
    EXPECT_THROW(t2.max<int>(), std::runtime_error);
}

using TensorTypes = testing::Types<int8_t, int, float>;

template <typename T>
class TensorFuncTest : public testing::Test
{
};

TYPED_TEST_SUITE(TensorFuncTest, TensorTypes);

TYPED_TEST(TensorFuncTest, MaxCorrectness)
{
    using T = TypeParam;

    size_t size = 4;

    T* v1 = new T[size]{T(1), T(2), T(3), T(4)};
    T* v2 = new T[size]{T(4), T(3), T(2), T(1)};
    T* v3 = new T[size]{T(1), T(2), T(4), T(3)};

    Tensor t1 = Tensor(MEMORY_CPU, getTensorType<T>(), {size}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, getTensorType<T>(), {size}, v2);
    Tensor t3 = Tensor(MEMORY_CPU, getTensorType<T>(), {size}, v3);

    EXPECT_EQ(t1.max<T>(), T(4));
    EXPECT_EQ(t2.max<T>(), T(4));
    EXPECT_EQ(t3.max<T>(), T(4));

    delete[] v1;
    delete[] v2;
    delete[] v3;
}

TYPED_TEST(TensorFuncTest, MinCorrectness)
{
    using T = TypeParam;

    size_t size = 4;

    T* v1 = new T[size]{T(1), T(2), T(3), T(4)};
    T* v2 = new T[size]{T(4), T(3), T(2), T(1)};
    T* v3 = new T[size]{T(1), T(2), T(4), T(3)};

    Tensor t1 = Tensor(MEMORY_CPU, getTensorType<T>(), {size}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, getTensorType<T>(), {size}, v2);
    Tensor t3 = Tensor(MEMORY_CPU, getTensorType<T>(), {size}, v3);

    EXPECT_EQ(t1.min<T>(), T(1));
    EXPECT_EQ(t2.min<T>(), T(1));
    EXPECT_EQ(t3.min<T>(), T(1));

    delete[] v1;
    delete[] v2;
    delete[] v3;
}

TYPED_TEST(TensorFuncTest, AnyCorrectness)
{
    using T = TypeParam;

    T* v = new T[4]{T(1), T(2), T(3), T(4)};
    Tensor t = Tensor{MEMORY_CPU, getTensorType<T>(), {4}, v};
    EXPECT_TRUE(t.any<T>(T(1)));
    EXPECT_FALSE(t.any<T>(T(5)));
    delete[] v;
}

TYPED_TEST(TensorFuncTest, AllCorrectness)
{
    using T = TypeParam;

    constexpr size_t size = 4;
    T* v1 = new T[size]{T(1), T(1), T(1), T(1)};
    T* v2 = new T[size]{T(1), T(1), T(1), T(2)};
    Tensor t1 = Tensor{MEMORY_CPU, getTensorType<T>(), {size}, v1};
    Tensor t2 = Tensor{MEMORY_CPU, getTensorType<T>(), {size}, v2};
    EXPECT_TRUE(t1.all<T>(T(1)));
    EXPECT_FALSE(t2.all<T>(T(2)));
    delete[] v1;
    delete[] v2;
}

TYPED_TEST(TensorFuncTest, SliceCorrectness)
{
    using T = TypeParam;

    constexpr int size = 12;
    T* v = new T[size];
    for (int i = 0; i < size; ++i)
    {
        v[i] = i;
    }

    DataType dtype = getTensorType<T>();
    Tensor t1 = Tensor(MEMORY_CPU, dtype, {3, 4}, v);
    Tensor t2 = t1.slice({2, 4}, 4);

    EXPECT_EQUAL_TENSORS(t2, Tensor(MEMORY_CPU, dtype, {2, 4}, &v[4]));
    // An overflowed tensor throws an exception.
    EXPECT_THROW(t1.slice({2, 4}, 5), std::runtime_error);

    delete[] v;
}

} // end of namespace
