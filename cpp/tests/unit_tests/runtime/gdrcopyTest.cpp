#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/gdrwrap.h"

// Skip all tests on Windows
#ifndef _WIN32

namespace tensorrt_llm::runtime::gdrcopy
{

namespace
{

class GdrCopyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Try to initialize GDRCopy
        gdrCopyInitialized = gdrcopy::initialize();
        if (!gdrCopyInitialized)
        {
            GTEST_SKIP() << "GDRCopy library not found or failed to initialize. Skipping all tests.";
        }

        gdrHandle = gdrcopy::open();
        ASSERT_NE(gdrHandle, nullptr) << "gdr_open() failed after successful initialization.";
    }

    void TearDown() override
    {
        if (gdrCopyInitialized && gdrHandle)
        {
            ASSERT_EQ(gdrcopy::close(gdrHandle), 0);
        }
    }

    bool gdrCopyInitialized = false;
    gdr_t gdrHandle = nullptr;
    const size_t TEST_SIZE = 4096; // Use a page size for testing
};

TEST_F(GdrCopyTest, GetVersionInfo)
{
    int libMajor = 0, libMinor = 0;
    gdrcopy::runtime_get_version(&libMajor, &libMinor);
    TLLM_LOG_INFO("GDRCopy library version: %d.%d", libMajor, libMinor);
    EXPECT_GE(libMajor, 2);

    int drvMajor = 0, drvMinor = 0;
    gdrcopy::driver_get_version(gdrHandle, &drvMajor, &drvMinor);
    TLLM_LOG_INFO("GDRCopy driver version: %d.%d", drvMajor, drvMinor);
    EXPECT_GE(drvMajor, 2);
}

TEST_F(GdrCopyTest, RawApiLifecycle)
{
    // 1. Allocate aligned device memory
    char* d_buffer;
    TLLM_CUDA_CHECK(cudaMalloc(&d_buffer, TEST_SIZE + GPU_PAGE_SIZE));
    unsigned long d_addr_aligned = ((unsigned long) d_buffer + GPU_PAGE_OFFSET) & GPU_PAGE_MASK;

    // 2. Pin buffer
    gdr_mh_t mh;
    ASSERT_EQ(gdrcopy::pin_buffer(gdrHandle, d_addr_aligned, TEST_SIZE, 0, 0, &mh), 0);

    // 3. Get Info before map
    gdr_info_t info;
    ASSERT_EQ(gdrcopy::get_info(gdrHandle, mh, &info), 0);
    EXPECT_EQ(info.mapped, 0);
    EXPECT_EQ(info.va, d_addr_aligned);

    // 4. Map buffer
    void* map_ptr = nullptr;
    ASSERT_EQ(gdrcopy::map(gdrHandle, mh, &map_ptr, TEST_SIZE), 0);
    ASSERT_NE(map_ptr, nullptr);

    // 5. Get Info after map
    ASSERT_EQ(gdrcopy::get_info(gdrHandle, mh, &info), 0);
    EXPECT_EQ(info.mapped, 1);
    EXPECT_TRUE(info.wc_mapping);

    // 6. Test copy functions
    std::vector<char> h_src(TEST_SIZE);
    std::vector<char> h_dst(TEST_SIZE, 0);
    for (size_t i = 0; i < TEST_SIZE; ++i)
    {
        h_src[i] = static_cast<char>(i % 256);
    }

    // Host -> Mapped GPU memory
    ASSERT_EQ(gdrcopy::copy_to_mapping(mh, map_ptr, h_src.data(), TEST_SIZE), 0);

    // Mapped GPU memory -> Host
    ASSERT_EQ(gdrcopy::copy_from_mapping(mh, h_dst.data(), map_ptr, TEST_SIZE), 0);

    // Verify
    EXPECT_EQ(memcmp(h_src.data(), h_dst.data(), TEST_SIZE), 0);

    // 7. Unmap buffer
    ASSERT_EQ(gdrcopy::unmap(gdrHandle, mh, map_ptr, TEST_SIZE), 0);

    // 8. Unpin buffer
    ASSERT_EQ(gdrcopy::unpin_buffer(gdrHandle, mh), 0);

    // 9. Free device memory
    TLLM_CUDA_CHECK(cudaFree(d_buffer));
}

TEST_F(GdrCopyTest, HelperLifecycle)
{
    char* gdr_ptr = nullptr;
    char* dev_ptr = nullptr;
    GdrMemDesc* mem_desc = nullptr;

    // 1. Allocate using helper
    ASSERT_NO_THROW(gdrcopy::gdrCudaMalloc(
        reinterpret_cast<void**>(&gdr_ptr), reinterpret_cast<void**>(&dev_ptr), TEST_SIZE, &mem_desc, gdrHandle));
    ASSERT_NE(gdr_ptr, nullptr);
    ASSERT_NE(dev_ptr, nullptr);
    ASSERT_NE(mem_desc, nullptr);

    // 2. Prepare host data and copy to GDR mapped pointer
    std::vector<char> h_src(TEST_SIZE);
    for (size_t i = 0; i < TEST_SIZE; ++i)
    {
        h_src[i] = static_cast<char>((i + 1) % 256);
    }
    memcpy(gdr_ptr, h_src.data(), TEST_SIZE);

    // 3. IMPORTANT: Flush WC buffer to ensure data is written to the device
    gdrcopy::wc_store_fence();

    // 4. Verify by copying back from the *device* pointer
    std::vector<char> h_dst(TEST_SIZE, 0);
    TLLM_CUDA_CHECK(cudaMemcpy(h_dst.data(), dev_ptr, TEST_SIZE, cudaMemcpyDeviceToHost));
    EXPECT_EQ(memcmp(h_src.data(), h_dst.data(), TEST_SIZE), 0);

    // 5. Free using helper
    ASSERT_NO_THROW(gdrcopy::gdrCudaFree(mem_desc, gdrHandle));
}

} // namespace
} // namespace tensorrt_llm::runtime::gdrcopy

#endif // _WIN32
