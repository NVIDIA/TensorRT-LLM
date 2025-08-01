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
#include "custom_metrics_reporter.h"
#include "triton/backend/backend_common.h"
#include <iomanip>
#include <sstream>
#include <vector>

using namespace ::triton::common; // TritonJson

namespace triton::backend::inflight_batcher_llm::custom_metrics_reporter
{

const std::vector<std::string> CustomMetricsReporter::request_keys_{
    "Active Request Count", "Max Request Count", "Scheduled Requests", "Context Requests", "Waiting Requests"};
const std::vector<std::string> CustomMetricsReporter::request_labels_{
    "active", "max", "scheduled", "context", "waiting"};

const std::vector<std::string> CustomMetricsReporter::runtime_memory_keys_{
    "Runtime CPU Memory Usage", "Runtime GPU Memory Usage", "Runtime Pinned Memory Usage"};
const std::vector<std::string> CustomMetricsReporter::runtime_memory_labels_{"cpu", "gpu", "pinned"};

const std::vector<std::string> CustomMetricsReporter::kv_cache_keys_{"Max KV cache blocks", "Free KV cache blocks",
    "Used KV cache blocks", "Tokens per KV cache block", "Reused KV cache blocks", "Fraction used KV cache blocks"};
const std::vector<std::string> CustomMetricsReporter::kv_cache_labels_{
    "max", "free", "used", "tokens_per", "reused", "fraction"};

const std::vector<std::string> CustomMetricsReporter::dis_serving_keys_{"KV cache transfer time", "Request count"};
const std::vector<std::string> CustomMetricsReporter::dis_serving_labels_{"kv_cache_transfer_ms", "request_count"};

const std::vector<std::string> CustomMetricsReporter::v1_specific_keys_{
    "Total Context Tokens", "Total Generation Tokens", "Empty Generation Slots"};
const std::vector<std::string> CustomMetricsReporter::v1_specific_labels_{
    "total_context_tokens", "total_generation_tokens", "empty_generation_slots"};

const std::vector<std::string> CustomMetricsReporter::IFB_specific_keys_{
    "Total Context Tokens", "Generation Requests", "MicroBatch ID", "Paused Requests"};
const std::vector<std::string> CustomMetricsReporter::IFB_specific_labels_{
    "total_context_tokens", "generation_requests", "micro_batch_id", "paused_requests"};

const std::vector<std::string> CustomMetricsReporter::general_metric_keys_{"Timestamp", "Iteration Counter"};
const std::vector<std::string> CustomMetricsReporter::general_metric_labels_{"timestamp", "iteration_counter"};

const std::vector<std::string> CustomMetricsReporter::response_metric_type_keys_{"Total Output Tokens"};
const std::vector<std::string> CustomMetricsReporter::response_metric_type_labels_{"total_output_tokens"};

const std::vector<std::string> CustomMetricsReporter::input_metric_type_keys_{"Total Input Tokens"};
const std::vector<std::string> CustomMetricsReporter::input_metric_type_labels_{"total_input_tokens"};

uint64_t convertTimestampToMicroseconds(std::string const& ts)
{
    std::tm tm = {};
    std::stringstream ss(ts);
    ss >> std::get_time(&tm, "%m-%d-%Y %H:%M:%S");
    auto timestamp = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    auto epoch = std::chrono::time_point_cast<std::chrono::microseconds>(timestamp).time_since_epoch();
    uint64_t seconds_to_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(epoch).count();

    // std::get_time does not support microsecond resolution, so we must manually add
    // microseconds to our converted value.
    uint64_t microseconds = std::strtol(ts.substr(ts.rfind(".") + 1).c_str(), NULL, 10);
    return (seconds_to_microseconds + microseconds);
}

TritonMetricGroup::TritonMetricGroup(std::string const& metric_family_label,
    std::string const& metric_family_description, std::string const& category_label,
    std::vector<std::string> const& json_keys, std::vector<std::string> const& sub_labels)
    : metric_family_label_(metric_family_label)
    , metric_family_description_(metric_family_description)
    , category_label_(category_label)
    , json_keys_(json_keys)
    , sub_labels_(sub_labels)
{
}

TRITONSERVER_Error* TritonMetricGroup::CreateGroup(std::string const& model_name, const uint64_t version,
    TRITONSERVER_MetricKind kind, std::optional<const std::vector<double>> buckets_)
{
    TRITONSERVER_MetricFamily* metric_family = nullptr;
    RETURN_IF_ERROR(TRITONSERVER_MetricFamilyNew(
        &metric_family, kind, metric_family_label_.c_str(), metric_family_description_.c_str()));
    metric_family_.reset(metric_family);
    TRITONSERVER_MetricArgs* args = nullptr;
    std::vector<double> buckets;
    if (buckets_.has_value())
    {
        buckets = buckets_.value();
    }
    // overloading metris kind to determine update action as well
    switch (kind)
    {
    case TRITONSERVER_METRIC_KIND_GAUGE: update_function_ = decltype(update_function_){TRITONSERVER_MetricSet}; break;
    case TRITONSERVER_METRIC_KIND_COUNTER:
        update_function_ = decltype(update_function_){TRITONSERVER_MetricIncrement};
        break;
    case TRITONSERVER_METRIC_KIND_HISTOGRAM:
        update_function_ = decltype(update_function_){TRITONSERVER_MetricObserve};
        TRITONSERVER_MetricArgsNew(&args);
        TRITONSERVER_MetricArgsSetHistogram(args, buckets.data(), buckets.size());
        break;
    default: return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, "unsupported Triton metrics kind"); break;
    }
    std::vector<TRITONSERVER_Parameter const*> labels;
    std::unique_ptr<TRITONSERVER_Parameter, ParameterDeleter> model_label(
        TRITONSERVER_ParameterNew("model", TRITONSERVER_PARAMETER_STRING, model_name.c_str()));
    std::unique_ptr<TRITONSERVER_Parameter, ParameterDeleter> model_version(
        TRITONSERVER_ParameterNew("version", TRITONSERVER_PARAMETER_STRING, std::to_string(version).c_str()));
    labels.emplace_back(model_label.get());
    labels.emplace_back(model_version.get());

    for (size_t i = 0; i < sub_labels_.size(); ++i)
    {
        TRITONSERVER_Metric* metric;
        std::unique_ptr<TRITONSERVER_Parameter, ParameterDeleter> sub_label(
            TRITONSERVER_ParameterNew(category_label_.c_str(), TRITONSERVER_PARAMETER_STRING, sub_labels_[i].c_str()));
        labels.emplace_back(sub_label.get());
        RETURN_IF_ERROR(
            TRITONSERVER_MetricNewWithArgs(&metric, metric_family_.get(), labels.data(), labels.size(), args));
        std::unique_ptr<TRITONSERVER_Metric, MetricDeleter> unique_metric(metric);
        metrics_.push_back(std::move(unique_metric));
        labels.pop_back();
    }

    return nullptr; // success
}

TRITONSERVER_Error* TritonMetricGroup::UpdateGroup(std::vector<double>& values)
{
    for (size_t i = 0; i < values.size(); ++i)
    {
        RETURN_IF_ERROR(update_function_(metrics_[i].get(), values[i]));
    }
    return nullptr; // success
}

std::vector<std::string> const& TritonMetricGroup::JsonKeys() const
{
    return json_keys_;
}

TRITONSERVER_Error* CustomMetricsReporter::InitializeReporter(
    std::string const& model_name, const uint64_t version, bool const is_v1_model)
{
    /* REQUEST METRIC GROUP */
    request_metric_family_ = std::make_unique<TritonMetricGroup>(
        "nv_trt_llm_request_metrics", "TRT LLM request metrics", "request_type", request_keys_, request_labels_);

    RETURN_IF_ERROR(request_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(request_metric_family_));

    /* RUNTIME MEMORY METRIC GROUP */
    runtime_memory_metric_family_ = std::make_unique<TritonMetricGroup>("nv_trt_llm_runtime_memory_metrics",
        "TRT LLM runtime memory metrics", "memory_type", runtime_memory_keys_, runtime_memory_labels_);

    RETURN_IF_ERROR(runtime_memory_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(runtime_memory_metric_family_));

    /* KV CACHE METRIC GROUP */
    kv_cache_metric_family_ = std::make_unique<TritonMetricGroup>("nv_trt_llm_kv_cache_block_metrics",
        "TRT LLM KV cache block metrics", "kv_cache_block_type", kv_cache_keys_, kv_cache_labels_);

    RETURN_IF_ERROR(kv_cache_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(kv_cache_metric_family_));

    /* DISAGGREGATED SERVING METRIC GROUP */
    dis_serving_metric_family_ = std::make_unique<TritonMetricGroup>("nv_trt_llm_disaggregated_serving_metrics",
        "TRT LLM disaggregated serving metrics", "disaggregated_serving_type", dis_serving_keys_, dis_serving_labels_);

    // This group is created as counter because it is aggregation on request statistics
    RETURN_IF_ERROR(dis_serving_metric_family_->CreateGroup(model_name, version, TRITONSERVER_METRIC_KIND_COUNTER));
    metric_groups_.push_back(std::move(dis_serving_metric_family_));

    /* MODEL-TYPE METRIC GROUP (V1 / IFB) */
    std::string model = (is_v1_model) ? "v1" : "inflight_batcher";
    std::string model_metric_family_label = "nv_trt_llm_" + model + "_metrics";
    std::string model_metric_family_description = "TRT LLM " + model + "-specific metrics";
    std::string model_metric_family_category = model + "_specific_metric";

    if (is_v1_model)
    {
        model_type_metric_family_ = std::make_unique<TritonMetricGroup>(model_metric_family_label,
            model_metric_family_description, model_metric_family_category, v1_specific_keys_, v1_specific_labels_);
    }
    else
    {
        model_type_metric_family_ = std::make_unique<TritonMetricGroup>(model_metric_family_label,
            model_metric_family_description, model_metric_family_category, IFB_specific_keys_, IFB_specific_labels_);
    }

    RETURN_IF_ERROR(model_type_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(model_type_metric_family_));

    /* GENERAL METRIC GROUP */
    general_metric_family_ = std::make_unique<TritonMetricGroup>("nv_trt_llm_general_metrics",
        "General TRT LLM metrics", "general_type", general_metric_keys_, general_metric_labels_);

    RETURN_IF_ERROR(general_metric_family_->CreateGroup(model_name, version));
    metric_groups_.push_back(std::move(general_metric_family_));

    /* GENERAL METRIC GROUP */
    response_tokens_metric_family_ = std::make_unique<TritonMetricGroup>("nv_llm_output_token_len",
        "TRT LLM response metrics", "response_metric_type", response_metric_type_keys_, response_metric_type_labels_);
    std::vector<double> buckets = {10.0, 50.0, 100.0, 500.0, 1000.0};
    RETURN_IF_ERROR(
        response_tokens_metric_family_->CreateGroup(model_name, version, TRITONSERVER_METRIC_KIND_HISTOGRAM, buckets));
    metric_groups_.push_back(std::move(response_tokens_metric_family_));

    input_tokens_metric_family_ = std::make_unique<TritonMetricGroup>("nv_llm_input_token_len",
        "TRT LLM response metrics", "response_metric_type", input_metric_type_keys_, input_metric_type_labels_);
    RETURN_IF_ERROR(
        input_tokens_metric_family_->CreateGroup(model_name, version, TRITONSERVER_METRIC_KIND_HISTOGRAM, buckets));
    metric_groups_.push_back(std::move(input_tokens_metric_family_));

    return nullptr; // success
}

TRITONSERVER_Error* CustomMetricsReporter::UpdateCustomMetrics(std::string const& custom_metrics)
{
    triton::common::TritonJson::Value metrics;
    std::vector<std::string> members;
    metrics.Parse(custom_metrics);
    metrics.Members(&members);

    for (auto const& metric_group : metric_groups_)
    {
        std::vector<std::string> metric_group_keys = metric_group->JsonKeys();
        std::vector<double> metric_group_values;
        for (auto const& key : metric_group_keys)
        {
            triton::common::TritonJson::Value value_json;
            double value;
            if (!metrics.Find(key.c_str(), &value_json))
            {
                std::string errStr = std::string("Failed to find " + key + " in metrics.");
                continue;
            }
            if (key == "Timestamp")
            {
                std::string timestamp;
                value_json.AsString(&timestamp);
                value = convertTimestampToMicroseconds(timestamp);
            }
            else
            {
                value_json.AsDouble(&value);
            }

            metric_group_values.push_back(value);
        }
        if (metric_group_values.size() > 0)
        {
            // Update only when something to update
            RETURN_IF_ERROR(metric_group->UpdateGroup(metric_group_values));
        }
    }

    return nullptr;
}

} // namespace triton::backend::inflight_batcher_llm::custom_metrics_reporter
