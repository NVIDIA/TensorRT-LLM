// Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include "triton/core/tritonbackend.h"
#include "triton/core/tritonserver.h"
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace triton::backend::inflight_batcher_llm::custom_metrics_reporter
{

/// TritonMetricGroups are handled by the CustomMetricsReporter class
/// and encapsulate the creation/update functionality for a
/// group of TRT LLM statistics to be reported as custom Triton metrics.
/// The statistics (or custom metrics) handled by this class should
/// not be confused with Triton base metrics.
class TritonMetricGroup
{
public:
    TritonMetricGroup(std::string const& metric_family_label, std::string const& metric_family_description,
        std::string const& category_label, std::vector<std::string> const& json_keys,
        std::vector<std::string> const& labels);
    ~TritonMetricGroup(){};

    /// Create a new Triton metric family with corresponding metric
    /// pointers and parameters.
    ///
    /// \param model_name The name of the model to provide a metrics
    /// group for.
    /// \param version The version of the model to provide a metrics
    /// group for.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* CreateGroup(std::string const& model_name, const uint64_t version,
        TRITONSERVER_MetricKind kind = TRITONSERVER_METRIC_KIND_GAUGE,
        std::optional<const std::vector<double>> buckets = std::nullopt);

    /// Update the Triton metrics associated with this group using
    /// the parsed TRT LLM backend statistics values.
    ///
    /// \param values Values parsed from the TRT LLM backend
    /// statistics output, filtered by this group's JSON keys.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* UpdateGroup(std::vector<double>& values);

    /// Return a list of JSON keys that correspond to the TRT LLM
    /// statistics handled by this metric group.
    ///
    /// \return A const reference to vector of strings corresponding
    /// to the JSON keys associated with this group.
    std::vector<std::string> const& JsonKeys() const;

    /// Custom deleter for a unique TRITONSERVER_MetricFamily pointer
    struct MetricFamilyDeleter
    {
        void operator()(TRITONSERVER_MetricFamily* family)
        {
            if (family != nullptr)
            {
                TRITONSERVER_MetricFamilyDelete(family);
            }
        }
    };

    /// Custom deleter for a unique TRITONSERVER_Metric pointer
    struct MetricDeleter
    {
        void operator()(TRITONSERVER_Metric* metric)
        {
            if (metric != nullptr)
            {
                TRITONSERVER_MetricDelete(metric);
            }
        }
    };

    /// Custom deleter for a unique TRITONSERVER_Parameter pointer
    struct ParameterDeleter
    {
        void operator()(TRITONSERVER_Parameter* parameter)
        {
            if (parameter != nullptr)
            {
                TRITONSERVER_ParameterDelete(parameter);
            }
        }
    };

private:
    std::unique_ptr<TRITONSERVER_MetricFamily, MetricFamilyDeleter> metric_family_;
    std::vector<std::unique_ptr<TRITONSERVER_Metric, MetricDeleter>> metrics_;
    std::function<struct TRITONSERVER_Error*(struct TRITONSERVER_Metric*, double)> update_function_;
    std::string metric_family_label_;
    std::string metric_family_description_;
    std::string category_label_;
    std::vector<std::string> json_keys_;
    std::vector<std::string> sub_labels_;
};

/// CustomMetricsReporter is an interface class meant to facilitate the
/// connection between TRT LLM backend statistics and Triton custom metrics.
/// It functions by passing BatchManager statistics data from
/// the TRT LLM backend to the multiple TritonMetricsGroup objects
/// it handles.
class CustomMetricsReporter
{
public:
    CustomMetricsReporter(){};
    ~CustomMetricsReporter(){};

    /// Initialize the various TritonMetricGroups handled by
    /// by this class using the static key/label members below.
    ///
    /// \param model The name of the model to provide metrics for.
    /// \param version The version of the model to provide metrics for.
    /// \param is_v1_model Whether the model type is v1 or an inflight
    /// batching model.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* InitializeReporter(std::string const& model, const uint64_t version, bool const is_v1_model);

    /// Updates the vector of TritonMetricGroup objects with a
    /// JSON-formatted statistics string.
    ///
    /// \param statistics A JSON-formatted string of TRT LLM backend
    /// statistics.
    /// \return a TRITONSERVER_Error indicating success or failure.
    TRITONSERVER_Error* UpdateCustomMetrics(std::string const& custom_metrics);

    static const std::vector<std::string> request_keys_;
    static const std::vector<std::string> request_labels_;

    static const std::vector<std::string> runtime_memory_keys_;
    static const std::vector<std::string> runtime_memory_labels_;

    static const std::vector<std::string> kv_cache_keys_;
    static const std::vector<std::string> kv_cache_labels_;

    static const std::vector<std::string> dis_serving_keys_;
    static const std::vector<std::string> dis_serving_labels_;

    static const std::vector<std::string> v1_specific_keys_;
    static const std::vector<std::string> v1_specific_labels_;

    static const std::vector<std::string> IFB_specific_keys_;
    static const std::vector<std::string> IFB_specific_labels_;

    static const std::vector<std::string> general_metric_keys_;
    static const std::vector<std::string> general_metric_labels_;

    static const std::vector<std::string> response_metric_type_keys_;
    static const std::vector<std::string> response_metric_type_labels_;

    static const std::vector<std::string> input_metric_type_keys_;
    static const std::vector<std::string> input_metric_type_labels_;

private:
    std::vector<std::unique_ptr<TritonMetricGroup>> metric_groups_;
    std::unique_ptr<TritonMetricGroup> request_metric_family_;
    std::unique_ptr<TritonMetricGroup> runtime_memory_metric_family_;
    std::unique_ptr<TritonMetricGroup> kv_cache_metric_family_;
    std::unique_ptr<TritonMetricGroup> dis_serving_metric_family_;
    std::unique_ptr<TritonMetricGroup> model_type_metric_family_;
    std::unique_ptr<TritonMetricGroup> general_metric_family_;
    std::unique_ptr<TritonMetricGroup> response_tokens_metric_family_;
    std::unique_ptr<TritonMetricGroup> input_tokens_metric_family_;
};

} // namespace triton::backend::inflight_batcher_llm::custom_metrics_reporter
