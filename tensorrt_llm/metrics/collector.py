"""Utilities for Prometheus Metrics Collection."""

import time
from typing import Dict, Optional, Union

from .enums import MetricNames


# Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0rc1/vllm/engine/metrics.py#L30
class MetricsCollector:
    labelname_finish_reason = "finished_reason"

    def __init__(self, labels: Dict[str, str]) -> None:
        from prometheus_client import Counter, Histogram
        self.last_log_time = time.time()
        self.labels = labels

        self.finish_reason_label = {
            MetricsCollector.labelname_finish_reason: "unknown"
        }
        self.labels_with_finished_reason = {
            **self.labels,
            **self.finish_reason_label
        }

        self.counter_request_success = Counter(
            name="request_success_total",
            documentation="Count of successfully processed requests.",
            labelnames=self.labels_with_finished_reason.keys())

        self.histogram_e2e_time_request = Histogram(
            name="e2e_request_latency_seconds",
            documentation="Histogram of end to end request latency in seconds.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_to_first_token = Histogram(
            name="time_to_first_token_seconds",
            documentation="Histogram of time to first token in seconds.",
            buckets=[
                0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5,
                0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0,
                2560.0
            ],
            labelnames=self.labels.keys())

        self.histogram_time_per_output_token = Histogram(
            name="time_per_output_token_seconds",
            documentation="Histogram of time per output token in seconds.",
            buckets=[
                0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75,
                1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0
            ],
            labelnames=self.labels.keys())

        self.histogram_queue_time_request = Histogram(
            name="request_queue_time_seconds",
            documentation=
            "Histogram of time spent in WAITING phase for request.",
            buckets=[
                0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0,
                40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0
            ],
            labelnames=self.labels.keys())

    def _label_merge(self, labels: Dict[str, str]) -> Dict[str, str]:
        if labels is None or len(labels) == 0:
            return self.labels
        return {**self.labels, **labels}

    def _log_counter(self, counter, labels: Dict[str, str],
                     data: Union[int, float]) -> None:
        # Convenience function for logging to counter.
        counter.labels(**self._label_merge(labels)).inc(data)

    def _log_histogram(self, histogram, data: Union[int, float]) -> None:
        # Convenience function for logging to histogram.
        histogram.labels(**self.labels).observe(data)

    def log_request_success(self, data: Union[int, float],
                            labels: Dict[str, str]) -> None:
        self._log_counter(self.counter_request_success, labels, data)
        self.last_log_time = time.time()

    def log_histogram(self, data: Optional[dict[str, float]]) -> None:
        if e2e := data.get(MetricNames.E2E, 0):
            self._log_histogram(self.histogram_e2e_time_request, e2e)
        if ttft := data.get(MetricNames.TTFT, 0):
            self._log_histogram(self.histogram_time_to_first_token, ttft)
        if tpot := data.get(MetricNames.TPOT, 0):
            self._log_histogram(self.histogram_time_per_output_token, tpot)
        if request_queue_time := data.get(MetricNames.REQUEST_QUEUE_TIME, 0):
            self._log_histogram(self.histogram_queue_time_request,
                                request_queue_time)
        self.last_log_time = time.time()

    def log_metrics_dict(self, metrics_dict: dict[str, float]) -> None:
        if finish_reason := metrics_dict.get(
                MetricsCollector.labelname_finish_reason):
            self.log_request_success(
                1, {MetricsCollector.labelname_finish_reason: finish_reason})
            self.log_histogram(metrics_dict)
