/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "cutlass/gemm_coord.h"
#include "cutlass/layout/layout.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////
/// Result structure
struct Result
{
    double avg_runtime_ms;
    double gflops;
    cutlass::Status status;
    cudaError_t error;
    bool passed;

    Result(double avg_runtime_ms = 0, double gflops = 0, cutlass::Status status = cutlass::Status::kSuccess,
        cudaError_t error = cudaSuccess)
        : avg_runtime_ms(avg_runtime_ms)
        , gflops(gflops)
        , status(status)
        , error(error)
        , passed(false)
    {
    }
};

/// Command line options parsing
struct Options
{
    std::string command_name;
    bool help;
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord problem_size_out;
    float alpha;
    float beta;
    float scale_d0;
    float scale_d1;
    float scale_output;
    bool has_bias;
    int split_k_factor;
    int avail_sms;
    int iterations;
    bool real;
    bool debug;
    bool no_check;

    Options(std::string command_name)
        : command_name(command_name)
        , help(false)
        , problem_size({32, 96, 128})
        , alpha(1.0f)
        , beta(1.0f)
        , scale_d0(1.0f)
        , scale_d1(1.0f)
        , scale_output(1.0f)
        , has_bias(false)
        , split_k_factor(1)
        , avail_sms(-1) // Number of device SMs to use is unlimited
        , real(false)
        , iterations(10)
        , debug(false)
        , no_check(false)
    {
        parse(0, nullptr);
    }

    Options()
        : Options("")
    {
    }

    Options(Options const& other)
        : command_name(other.command_name)
        , help(other.help)
        , problem_size((other.problem_size))
        , problem_size_out((other.problem_size_out))
        , alpha(other.alpha)
        , beta(other.beta)
        , scale_d0(other.scale_d0)
        , scale_d1(other.scale_d1)
        , scale_output(other.scale_output)
        , has_bias(other.has_bias)
        , split_k_factor(other.split_k_factor)
        , avail_sms(other.avail_sms) // Number of device SMs to use is unlimited
        , real(other.real)
        , iterations(other.iterations)
        , debug(other.debug)
        , no_check(other.no_check)
    {
    }

    bool valid() const
    {
        return true;
    }

    void parse(int argc, char const** args)
    {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help"))
        {
            help = true;
        }

        cmd.get_cmd_line_argument("m", problem_size.m());
        cmd.get_cmd_line_argument("n", problem_size.n());
        cmd.get_cmd_line_argument("k", problem_size.k());
        cmd.get_cmd_line_argument("alpha", alpha);
        cmd.get_cmd_line_argument("beta", beta);
        cmd.get_cmd_line_argument("scale_d0", scale_d0);
        cmd.get_cmd_line_argument("scale_d1", scale_d1);
        cmd.get_cmd_line_argument("scale_output", scale_output);
        cmd.get_cmd_line_argument("split", split_k_factor);
        cmd.get_cmd_line_argument("iterations", iterations);
        real = cmd.check_cmd_line_flag("real");
        debug = cmd.check_cmd_line_flag("debug");
        no_check = cmd.check_cmd_line_flag("nocheck");
        has_bias = cmd.check_cmd_line_flag("bias");

        problem_size_out = cutlass::gemm::GemmCoord(problem_size.m(), problem_size.n() / 2, problem_size.k());
    }

    /// Prints the usage statement.
    std::ostream& print_usage(std::ostream& out) const
    {
        out << "Performs a GEMM computation.\n"
            << "\n"
            << "Options:\n"
            << "\n"
            << "  --help                      If specified, displays this usage statement.\n\n"
            << "  --m=<int>                   GEMM M dimension\n"
            << "  --n=<int>                   GEMM N dimension\n"
            << "  --k=<int>                   GEMM K dimension\n"
            << "  --alpha=<f32>               Epilogue scalar alpha\n"
            << "  --beta=<f32>                Epilogue scalar beta\n\n"
            << "  --scale_d0=<f32>            Epilogue scalar scale_d0\n"
            << "  --scale_d1=<f32>            Epilogue scalar scale_d1\n\n"
            << "  --scale_output=<f32>        Epilogue scalar scale_output\n\n"
            << "  --split=<int>               Split-K factor to emulate\n\n"
            << "  --real                      If specified, initializes with real values instead of whole numbers. "
               "Errors are to be expected.\n\n"
            << "  --iterations=<int>          Number of profiling iterations to perform.\n\n";

        out << "\n\nExamples:\n\n"
            << "$ " << command_name << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

        return out;
    }

    /// Compute performance in GFLOP/s
    double gflops(double runtime_s) const
    {
        // Two flops per multiply-add
        return 2.0 * double(problem_size.product()) / double(1.0e9) / runtime_s;
    }
};
