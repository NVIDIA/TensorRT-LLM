/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>
#include <string>

#include "tensorrt_llm/common/packageLocation.h"

namespace fs = std::filesystem;
using tensorrt_llm::common::findPackageLocationFromLibraryPath;
using tensorrt_llm::common::parsePipShowLocation;

namespace
{

class PackageLocationTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        std::random_device rd;
        mRoot = fs::temp_directory_path() / ("trtllm_package_location_test_" + std::to_string(rd()));
        fs::create_directories(mRoot);
    }

    void TearDown() override
    {
        std::error_code ec;
        fs::remove_all(mRoot, ec);
    }

    //! Create an empty file (and its parent directories) and return its path.
    static fs::path touch(fs::path const& path)
    {
        fs::create_directories(path.parent_path());
        std::ofstream{path};
        return path;
    }

    fs::path mRoot;
};

} // namespace

TEST_F(PackageLocationTest, WheelInstall)
{
    auto const sitePackages = mRoot / "venv" / "lib" / "python3.12" / "site-packages";
    fs::create_directories(sitePackages / "tensorrt_llm" / "include" / "deep_gemm");
    auto const library = touch(sitePackages / "tensorrt_llm" / "libs" / "libtensorrt_llm.so");

    auto const location = findPackageLocationFromLibraryPath(library);
    ASSERT_TRUE(location.has_value());
    EXPECT_TRUE(fs::equivalent(*location, sitePackages));
}

TEST_F(PackageLocationTest, BindingsNextToPackage)
{
    auto const sitePackages = mRoot / "site-packages";
    fs::create_directories(sitePackages / "tensorrt_llm" / "include");
    auto const bindings = touch(sitePackages / "tensorrt_llm" / "bindings.cpython-312-x86_64-linux-gnu.so");

    auto const location = findPackageLocationFromLibraryPath(bindings);
    ASSERT_TRUE(location.has_value());
    EXPECT_TRUE(fs::equivalent(*location, sitePackages));
}

TEST_F(PackageLocationTest, SymlinkedLibraryOutsidePackage)
{
    auto const sitePackages = mRoot / "site-packages";
    fs::create_directories(sitePackages / "tensorrt_llm" / "include");
    auto const library = touch(sitePackages / "tensorrt_llm" / "libs" / "libtensorrt_llm.so");
    auto const link = mRoot / "opt" / "lib" / "libtensorrt_llm.so";
    fs::create_directories(link.parent_path());
    fs::create_symlink(library, link);

    auto const location = findPackageLocationFromLibraryPath(link);
    ASSERT_TRUE(location.has_value());
    EXPECT_TRUE(fs::equivalent(*location, sitePackages));
}

TEST_F(PackageLocationTest, LibraryOutsidePackage)
{
    auto const library = touch(mRoot / "opt" / "tensorrt_llm" / "lib" / "libtensorrt_llm.so");

    EXPECT_FALSE(findPackageLocationFromLibraryPath(library).has_value());
}

TEST(PackageLocation, ParsePipShowLocation)
{
    auto const location = parsePipShowLocation(
        "Name: tensorrt_llm\nVersion: 1.3.0rc29\nLocation: /usr/local/lib/python3.12/dist-packages\nRequires: "
        "torch\n");
    ASSERT_TRUE(location.has_value());
    EXPECT_EQ(*location, fs::path("/usr/local/lib/python3.12/dist-packages"));
}

TEST(PackageLocation, ParsePipShowPrefersEditableLocation)
{
    auto const location = parsePipShowLocation(
        "Name: tensorrt_llm\r\nLocation: /venv/lib/python3.12/site-packages\r\nEditable project location: /code  \r\n");
    ASSERT_TRUE(location.has_value());
    EXPECT_EQ(*location, fs::path("/code"));
}

TEST(PackageLocation, ParsePipShowWithoutLocation)
{
    EXPECT_FALSE(parsePipShowLocation("").has_value());
    EXPECT_FALSE(parsePipShowLocation("WARNING: Package(s) not found: tensorrt_llm\n").has_value());
}
