#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for time_breakdown module

Run tests with:
    python -m pytest tests/unittest/others/test_time_breakdown.py -v
    or
    python -m unittest tests.unittest.others.test_time_breakdown
"""

import json
import math
import os
import tempfile
import unittest
from unittest.mock import patch

from tensorrt_llm.serve.scripts.time_breakdown import (RequestDataParser,
                                                       RequestTimeBreakdown,
                                                       TimingMetric,
                                                       TimingMetricsConfig)


class TestTimingMetric(unittest.TestCase):
    """Test TimingMetric class."""

    def test_timing_metric_creation(self):
        """Test basic TimingMetric creation."""
        metric = TimingMetric(name='test_metric',
                              display_name='Test Metric',
                              color='blue',
                              description='Test description',
                              start_field='start_time',
                              end_field='end_time',
                              server_type='ctx')

        self.assertEqual(metric.name, 'test_metric')
        self.assertEqual(metric.display_name, 'Test Metric')
        self.assertEqual(metric.color, 'blue')
        self.assertEqual(metric.description, 'Test description')
        self.assertEqual(metric.start_field, 'start_time')
        self.assertEqual(metric.end_field, 'end_time')
        self.assertEqual(metric.server_type, 'ctx')

    def test_calculate_duration_valid(self):
        """Test duration calculation with valid timestamps."""
        metric = TimingMetric(name='test',
                              display_name='Test',
                              color='blue',
                              description='Test',
                              start_field='start_time',
                              end_field='end_time')

        timing_data = {'start_time': 1.0, 'end_time': 3.5}

        duration = metric.calculate_duration(timing_data)
        self.assertEqual(duration, 2.5)

    def test_calculate_duration_missing_start(self):
        """Test duration calculation with missing start time."""
        metric = TimingMetric(name='test',
                              display_name='Test',
                              color='blue',
                              description='Test',
                              start_field='start_time',
                              end_field='end_time')

        timing_data = {'end_time': 3.5}

        duration = metric.calculate_duration(timing_data)
        self.assertEqual(duration, 0.0)

    def test_calculate_duration_missing_end(self):
        """Test duration calculation with missing end time."""
        metric = TimingMetric(name='test',
                              display_name='Test',
                              color='blue',
                              description='Test',
                              start_field='start_time',
                              end_field='end_time')

        timing_data = {'start_time': 1.0, 'end_time': 0}

        duration = metric.calculate_duration(timing_data)
        self.assertEqual(duration, 0.0)

    def test_calculate_duration_negative(self):
        """Test duration calculation doesn't produce negative values."""
        metric = TimingMetric(name='test',
                              display_name='Test',
                              color='blue',
                              description='Test',
                              start_field='start_time',
                              end_field='end_time')

        timing_data = {'start_time': 5.0, 'end_time': 3.5}

        duration = metric.calculate_duration(timing_data)
        self.assertEqual(duration, 0.0)


class TestTimingMetricsConfig(unittest.TestCase):
    """Test TimingMetricsConfig class."""

    def test_default_metrics_loaded(self):
        """Test that default metrics are loaded."""
        config = TimingMetricsConfig()

        # Should have multiple default metrics
        self.assertGreater(len(config.metrics), 0)

        # Check for expected metric names
        metric_names = [m.name for m in config.metrics]
        self.assertIn('ctx_preprocessing', metric_names)
        self.assertIn('ctx_processing', metric_names)

    def test_get_metric_by_name(self):
        """Test retrieving a metric by name."""
        config = TimingMetricsConfig()

        metric = config.get_metric_by_name('ctx_preprocessing')
        self.assertIsNotNone(metric)
        self.assertEqual(metric.name, 'ctx_preprocessing')

        # Test non-existent metric
        metric = config.get_metric_by_name('non_existent')
        self.assertIsNone(metric)

    def test_disagg_relay_metric_exists(self):
        """Test that disagg_relay metric exists in default configuration."""
        config = TimingMetricsConfig()

        metric = config.get_metric_by_name('disagg_relay')
        self.assertIsNotNone(metric)
        self.assertEqual(metric.name, 'disagg_relay')
        self.assertEqual(metric.display_name, 'Disagg Relay')
        self.assertEqual(metric.start_field, 'ctx_server_first_token_time')
        self.assertEqual(metric.end_field, 'gen_server_arrival_time')
        self.assertEqual(metric.server_type, 'disagg')

    def test_disagg_relay_calculation(self):
        """Test disagg_relay duration calculation."""
        config = TimingMetricsConfig()
        metric = config.get_metric_by_name('disagg_relay')

        # Test with valid timing data
        timing_data = {
            'ctx_server_first_token_time': 1.5,
            'gen_server_arrival_time': 2.0
        }
        duration = metric.calculate_duration(timing_data)
        self.assertAlmostEqual(duration, 0.5, places=5)

        # Test with missing fields
        timing_data_missing = {'ctx_server_first_token_time': 1.5}
        duration = metric.calculate_duration(timing_data_missing)
        self.assertEqual(duration, 0.0)

    def test_metrics_list_modification(self):
        """Test that metrics list can be modified directly."""
        config = TimingMetricsConfig()
        initial_count = len(config.metrics)

        # Add a metric directly to the list
        new_metric = TimingMetric(name='custom_metric',
                                  display_name='Custom Metric',
                                  color='red',
                                  description='Custom test metric',
                                  start_field='start',
                                  end_field='end')

        config.metrics.append(new_metric)
        self.assertEqual(len(config.metrics), initial_count + 1)
        self.assertIsNotNone(config.get_metric_by_name('custom_metric'))


class TestRequestDataParser(unittest.TestCase):
    """Test RequestDataParser class."""

    def test_parse_aggregated_format(self):
        """Test parsing aggregated (non-disaggregated) format."""
        parser = RequestDataParser()

        request_data = {
            'request_id': 'req123',
            'perf_metrics': {
                'timing_metrics': {
                    'server_arrival_time': 1.0,
                    'arrival_time': 1.1,
                    'first_scheduled_time': 1.2,
                    'first_token_time': 1.5,
                    'server_first_token_time': 1.6
                }
            }
        }

        parsed = parser.parse_request(request_data, 0)

        self.assertEqual(parsed['request_index'], 'req123')
        self.assertEqual(parsed['ctx_server_arrival_time'], 1.0)
        self.assertEqual(parsed['ctx_arrival_time'], 1.1)
        self.assertEqual(parsed['ctx_first_scheduled_time'], 1.2)
        self.assertEqual(parsed['ctx_first_token_time'], 1.5)
        self.assertEqual(parsed['ctx_server_first_token_time'], 1.6)

        # Gen metrics should be NaN in aggregated format
        self.assertTrue(math.isnan(parsed['gen_server_arrival_time']))
        self.assertTrue(math.isnan(parsed['disagg_server_arrival_time']))

    def test_parse_disaggregated_format(self):
        """Test parsing disaggregated format."""
        parser = RequestDataParser()

        request_data = {
            'ctx_perf_metrics': {
                'request_id': 'req456',
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': 1.0,
                        'arrival_time': 1.1,
                        'first_scheduled_time': 1.2,
                        'first_token_time': 1.5,
                        'server_first_token_time': 1.6
                    }
                }
            },
            'gen_perf_metrics': {
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': 2.0,
                        'arrival_time': 2.1,
                        'first_scheduled_time': 2.2,
                        'first_token_time': 2.5,
                        'server_first_token_time': 2.6
                    }
                }
            },
            'disagg_server_arrival_time': 0.5,
            'disagg_server_first_token_time': 3.0
        }

        parsed = parser.parse_request(request_data, 0)

        self.assertEqual(parsed['request_index'], 'req456')

        # Context metrics
        self.assertEqual(parsed['ctx_server_arrival_time'], 1.0)
        self.assertEqual(parsed['ctx_arrival_time'], 1.1)

        # Generation metrics
        self.assertEqual(parsed['gen_server_arrival_time'], 2.0)
        self.assertEqual(parsed['gen_arrival_time'], 2.1)

        # Disaggregation metrics
        self.assertEqual(parsed['disagg_server_arrival_time'], 0.5)
        self.assertEqual(parsed['disagg_server_first_token_time'], 3.0)

    def test_parse_missing_fields(self):
        """Test parsing with missing fields (should default to 0)."""
        parser = RequestDataParser()

        request_data = {
            'request_id': 'req789',
            'perf_metrics': {
                'timing_metrics': {}
            }
        }

        parsed = parser.parse_request(request_data, 0)

        # All timing fields should default to NaN
        self.assertTrue(math.isnan(parsed['ctx_server_first_token_time']))
        self.assertTrue(math.isnan(parsed['ctx_arrival_time']))
        self.assertTrue(math.isnan(parsed['gen_server_arrival_time']))

    def test_parse_uses_index_as_fallback(self):
        """Test that index is used when request_id is missing."""
        parser = RequestDataParser()

        request_data = {'perf_metrics': {'timing_metrics': {}}}

        parsed = parser.parse_request(request_data, 42)

        self.assertEqual(parsed['request_index'], 42)

    def test_parse_disagg_relay_fields(self):
        """Test parsing disaggregated format includes disagg_relay timing fields."""
        parser = RequestDataParser()

        request_data = {
            'ctx_perf_metrics': {
                'request_id': 'req_relay',
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': 1.0,
                        'arrival_time': 1.1,
                        'first_scheduled_time': 1.2,
                        'first_token_time': 1.5,
                        'server_first_token_time': 1.6  # End of disagg_relay
                    }
                }
            },
            'gen_perf_metrics': {
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time':
                        2.0,  # Start of disagg_relay (from gen perspective)
                        'arrival_time': 2.1,
                        'first_scheduled_time': 2.2,
                        'first_token_time': 2.5,
                        'server_first_token_time': 2.6
                    }
                }
            },
            'disagg_server_arrival_time': 0.5,
            'disagg_server_first_token_time': 3.0
        }

        parsed = parser.parse_request(request_data, 0)

        # Verify that both fields required for disagg_relay are present
        self.assertEqual(parsed['ctx_server_first_token_time'], 1.6)
        self.assertEqual(parsed['gen_server_arrival_time'], 2.0)

        # The relay time should be: gen_server_arrival_time - ctx_server_first_token_time
        # = 2.0 - 1.6 = 0.4 seconds
        config = TimingMetricsConfig()
        disagg_relay_metric = config.get_metric_by_name('disagg_relay')
        relay_duration = disagg_relay_metric.calculate_duration(parsed)
        self.assertAlmostEqual(relay_duration, 0.4, places=5)


class TestRequestTimeBreakdown(unittest.TestCase):
    """Test RequestTimeBreakdown class."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = RequestTimeBreakdown()

        # Create a temporary JSON file for testing
        self.test_data = [{
            'request_id': 0,
            'perf_metrics': {
                'timing_metrics': {
                    'server_arrival_time': 1.0,
                    'arrival_time': 1.1,
                    'first_scheduled_time': 1.2,
                    'first_token_time': 1.5,
                    'server_first_token_time': 1.6
                }
            }
        }, {
            'request_id': 1,
            'perf_metrics': {
                'timing_metrics': {
                    'server_arrival_time': 2.0,
                    'arrival_time': 2.1,
                    'first_scheduled_time': 2.3,
                    'first_token_time': 2.7,
                    'server_first_token_time': 2.8
                }
            }
        }]

    def test_parse_json_file(self):
        """Test parsing a JSON file."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump(self.test_data, f)
            temp_file = f.name

        try:
            timing_data = self.analyzer.parse_json_file(temp_file)

            self.assertEqual(len(timing_data), 2)

            # Check first request
            self.assertEqual(timing_data[0]['request_index'], 0)
            self.assertEqual(timing_data[0]['ctx_server_arrival_time'], 1.0)

            # Check that durations were calculated
            self.assertIn('ctx_preprocessing_time', timing_data[0])
            self.assertIn('ctx_queue_time', timing_data[0])

            # Verify a specific duration calculation
            # ctx_preprocessing = ctx_arrival_time - ctx_server_arrival_time
            expected_preprocessing = 1.1 - 1.0
            self.assertAlmostEqual(timing_data[0]['ctx_preprocessing_time'],
                                   expected_preprocessing,
                                   places=5)
        finally:
            os.unlink(temp_file)

    def test_parse_json_file_not_found(self):
        """Test parsing a non-existent file."""
        with self.assertRaises(SystemExit):
            self.analyzer.parse_json_file('non_existent_file.json')

    def test_parse_json_file_invalid_json(self):
        """Test parsing an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            f.write("{ invalid json")
            temp_file = f.name

        try:
            with self.assertRaises(SystemExit):
                self.analyzer.parse_json_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_create_timing_diagram(self):
        """Test creating a timing diagram."""
        # Create sample timing data with all required fields
        timing_data = [{
            'request_index': 0,
            'ctx_server_arrival_time': 1.0,
            'ctx_arrival_time': 1.1,
            'ctx_first_scheduled_time': 1.2,
            'ctx_first_token_time': 1.5,
            'ctx_server_first_token_time': 1.6,
            'gen_server_arrival_time': float('nan'),
            'disagg_server_arrival_time': float('nan'),
            'ctx_preprocessing_time': 0.1,
            'ctx_queue_time': 0.2,
            'ctx_processing_time': 0.3,
            'ctx_postprocessing_time': 0.05,
            'gen_preprocessing_time': 0,
            'gen_queue_time': 0,
            'gen_postprocessing_time': 0,
            'disagg_preprocessing_time': 0,
            'disagg_postprocessing_time': 0,
            'step_metrics': None,
        }, {
            'request_index': 1,
            'ctx_server_arrival_time': 2.0,
            'ctx_arrival_time': 2.1,
            'ctx_first_scheduled_time': 2.2,
            'ctx_first_token_time': 2.5,
            'ctx_server_first_token_time': 2.6,
            'gen_server_arrival_time': float('nan'),
            'disagg_server_arrival_time': float('nan'),
            'ctx_preprocessing_time': 0.15,
            'ctx_queue_time': 0.25,
            'ctx_processing_time': 0.35,
            'ctx_postprocessing_time': 0.06,
            'gen_preprocessing_time': 0,
            'gen_queue_time': 0,
            'gen_postprocessing_time': 0,
            'disagg_preprocessing_time': 0,
            'disagg_postprocessing_time': 0,
            'step_metrics': None,
        }]

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            # Run the diagram creation - it will write to the temp file
            self.analyzer.create_timing_diagram(timing_data, temp_file)

            # Verify that the output file was created and has content
            self.assertTrue(os.path.exists(temp_file))
            with open(temp_file, 'r') as f:
                content = f.read()
                self.assertIn('<html>', content)
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_create_timing_diagram_empty_data(self):
        """Test creating a diagram with empty data."""
        # Should handle gracefully without creating a file
        with patch('builtins.print') as mock_print:
            self.analyzer.create_timing_diagram([])
            mock_print.assert_called_with("No timing data to visualize.")

    def test_show_statistics(self):
        """Test showing statistics."""
        timing_data = [{
            'ctx_preprocessing_time': 0.1,
            'ctx_queue_time': 0.2,
            'ctx_processing_time': 0.3,
        }, {
            'ctx_preprocessing_time': 0.15,
            'ctx_queue_time': 0.25,
            'ctx_processing_time': 0.35,
        }]

        # Capture printed output
        with patch('builtins.print') as mock_print:
            self.analyzer.show_statistics(timing_data)

            # Should have printed something
            self.assertTrue(mock_print.called)

            # Check for expected content in printed output
            printed_output = ' '.join(
                [str(call[0][0]) for call in mock_print.call_args_list])
            self.assertIn('Total requests', printed_output)

    def test_show_statistics_empty_data(self):
        """Test showing statistics with empty data."""
        with patch('builtins.print') as mock_print:
            self.analyzer.show_statistics([])
            mock_print.assert_called_with("No timing data to analyze.")

    def test_custom_config(self):
        """Test using a custom configuration."""
        custom_config = TimingMetricsConfig()
        custom_config.metrics.append(
            TimingMetric(name='custom_metric',
                         display_name='Custom',
                         color='red',
                         description='Custom metric',
                         start_field='custom_start',
                         end_field='custom_end'))

        analyzer = RequestTimeBreakdown(config=custom_config)

        # Verify custom config is used
        self.assertIsNotNone(
            analyzer.config.get_metric_by_name('custom_metric'))


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """Test the complete workflow from file to diagram."""
        # Create test data with step_metrics for proper sorting
        test_data = [{
            'request_id': i,
            'perf_metrics': {
                'timing_metrics': {
                    'server_arrival_time': float(i),
                    'arrival_time': float(i) + 0.1,
                    'first_scheduled_time': float(i) + 0.2,
                    'first_token_time': float(i) + 0.5,
                    'server_first_token_time': float(i) + 0.6
                }
            },
            'time_breakdown_metrics': {
                'step_metrics': [{
                    'iter': 1,
                    'token_time': float(i) + 0.55
                }]
            }
        } for i in range(5)]

        # Create temporary files and run the complete workflow
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as json_f, \
             tempfile.NamedTemporaryFile(suffix='.html', delete=True) as html_f:
            # Write test data to JSON file
            json.dump(test_data, json_f)
            json_f.flush()  # Ensure data is written before reading

            analyzer = RequestTimeBreakdown()
            timing_data = analyzer.parse_json_file(json_f.name)

            # Verify parsing
            self.assertEqual(len(timing_data), 5)

            # Create the diagram
            analyzer.create_timing_diagram(timing_data, html_f.name)

    def test_full_workflow_with_disagg_relay(self):
        """Test the complete workflow with disaggregated data including disagg_relay."""
        # Create disaggregated test data with step_metrics for proper sorting
        test_data = [{
            'ctx_perf_metrics': {
                'request_id': i,
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': float(i),
                        'arrival_time': float(i) + 0.1,
                        'first_scheduled_time': float(i) + 0.2,
                        'first_token_time': float(i) + 0.5,
                        'server_first_token_time': float(i) + 0.6
                    }
                }
            },
            'gen_perf_metrics': {
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': float(i) + 1.0,
                        'arrival_time': float(i) + 1.1,
                        'first_scheduled_time': float(i) + 1.2,
                        'first_token_time': float(i) + 1.5,
                        'server_first_token_time': float(i) + 1.6
                    }
                },
                'time_breakdown_metrics': {
                    'step_metrics': [{
                        'iter': 1,
                        'prev_batch_token_time': float(i) + 1.55
                    }]
                }
            },
            'disagg_server_arrival_time': float(i) - 0.5,
            'disagg_server_first_token_time': float(i) + 2.0
        } for i in range(3)]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=True) as json_f, \
             tempfile.NamedTemporaryFile(suffix='.html', delete=True) as html_f:
            # Write test data to JSON file
            json.dump(test_data, json_f)
            json_f.flush()

            analyzer = RequestTimeBreakdown()
            timing_data = analyzer.parse_json_file(json_f.name)

            # Verify parsing
            self.assertEqual(len(timing_data), 3)

            # Verify disagg_relay_time is calculated
            for data in timing_data:
                self.assertIn('disagg_relay_time', data)
                # Expected relay time: gen_server_arrival_time - ctx_server_first_token_time
                # = (i + 1.0) - (i + 0.6) = 0.4
                self.assertAlmostEqual(data['disagg_relay_time'], 0.4, places=5)

            # Create the diagram
            analyzer.create_timing_diagram(timing_data, html_f.name)


class TestStepMetricsParsing(unittest.TestCase):
    """Test step_metrics parsing functionality."""

    def test_parse_step_metrics_non_disagg(self):
        """Test parsing step_metrics in non-disaggregated format."""
        parser = RequestDataParser()

        request_data = {
            'request_id': 'req_step',
            'perf_metrics': {
                'timing_metrics': {
                    'server_arrival_time': 1.0,
                    'arrival_time': 1.1,
                    'first_scheduled_time': 1.2,
                    'first_token_time': 1.5,
                    'server_first_token_time': 1.6
                }
            },
            'time_breakdown_metrics': {
                'step_metrics': [{
                    'iter': 1,
                    'forward_start_time': 1.21,
                    'forward_end_time': 1.35,
                    'sample_start_time': 1.36,
                    'sample_end_time': 1.40,
                    'gpu_forward_time': 10.5,
                    'gpu_sample_time': 0.5,
                    'token_time': 1.45
                }, {
                    'iter': 2,
                    'forward_start_time': 1.50,
                    'forward_end_time': 1.55,
                    'sample_start_time': 1.56,
                    'sample_end_time': 1.58,
                    'gpu_forward_time': 5.0,
                    'gpu_sample_time': 0.3,
                    'token_time': 1.60
                }],
                'ctx_gpu_forward_time':
                12.0,
                'ctx_gpu_sample_time':
                0.8
            }
        }

        parsed = parser.parse_request(request_data, 0)

        # Verify step_metrics is parsed
        self.assertIsNotNone(parsed['step_metrics'])
        self.assertEqual(len(parsed['step_metrics']), 2)

        # Verify first step content
        first_step = parsed['step_metrics'][0]
        self.assertEqual(first_step['iter'], 1)
        self.assertEqual(first_step['forward_start_time'], 1.21)
        self.assertEqual(first_step['gpu_forward_time'], 10.5)

        # Verify ctx GPU times
        self.assertEqual(parsed['ctx_gpu_forward_time'], 12.0)
        self.assertEqual(parsed['ctx_gpu_sample_time'], 0.8)

    def test_parse_step_metrics_disagg(self):
        """Test parsing step_metrics in disaggregated format."""
        parser = RequestDataParser()

        request_data = {
            'ctx_perf_metrics': {
                'request_id': 'req_disagg_step',
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': 1.0,
                        'arrival_time': 1.1,
                        'first_scheduled_time': 1.2,
                        'first_token_time': 1.5,
                        'server_first_token_time': 1.6
                    }
                }
            },
            'gen_perf_metrics': {
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': 2.0,
                        'arrival_time': 2.1,
                        'first_scheduled_time': 2.2,
                        'first_token_time': 2.5,
                        'server_first_token_time': 2.6
                    }
                },
                'time_breakdown_metrics': {
                    'step_metrics': [{
                        'iter': 1,
                        'forward_start_time': 2.25,
                        'forward_end_time': 2.30,
                        'sample_start_time': 2.31,
                        'sample_end_time': 2.35,
                        'gpu_forward_time': 4.0,
                        'gpu_sample_time': 0.2,
                        'prev_batch_token_time': 2.40
                    }],
                    'ctx_gpu_forward_time':
                    20.0,
                    'ctx_gpu_sample_time':
                    1.0
                }
            },
            'disagg_server_arrival_time': 0.5,
            'disagg_server_first_token_time': 3.0
        }

        parsed = parser.parse_request(request_data, 0)

        # Verify step_metrics from gen_perf_metrics
        self.assertIsNotNone(parsed['step_metrics'])
        self.assertEqual(len(parsed['step_metrics']), 1)

        # Verify overlap mode token time field
        first_step = parsed['step_metrics'][0]
        self.assertEqual(first_step['prev_batch_token_time'], 2.40)

        # Verify ctx GPU times
        self.assertEqual(parsed['ctx_gpu_forward_time'], 20.0)
        self.assertEqual(parsed['ctx_gpu_sample_time'], 1.0)

    def test_parse_step_metrics_legacy_format(self):
        """Test parsing step_metrics from legacy format (inside perf_metrics)."""
        parser = RequestDataParser()

        request_data = {
            'request_id': 'req_legacy',
            'perf_metrics': {
                'timing_metrics': {
                    'server_arrival_time': 1.0,
                    'arrival_time': 1.1,
                    'first_scheduled_time': 1.2,
                    'first_token_time': 1.5,
                    'server_first_token_time': 1.6
                },
                'time_breakdown_metrics': {
                    'step_metrics': [{
                        'iter': 1,
                        'forward_start_time': 1.21,
                        'forward_end_time': 1.35,
                        'gpu_forward_time': 8.0
                    }]
                }
            }
        }

        parsed = parser.parse_request(request_data, 0)

        # Verify step_metrics is parsed from legacy location
        self.assertIsNotNone(parsed['step_metrics'])
        self.assertEqual(len(parsed['step_metrics']), 1)


class TestSortingAndFiltering(unittest.TestCase):
    """Test sorting and filtering functionality in create_timing_diagram."""

    def setUp(self):
        """Set up parsed timing data with different arrival times and E2E latencies."""
        # Pre-parsed timing data format (as returned by parse_json_file)
        self.timing_data = [
            {
                'request_index': 0,
                'ctx_server_arrival_time': 3.0,  # Latest arrival
                'ctx_arrival_time': 3.1,
                'ctx_first_scheduled_time': 3.2,
                'ctx_first_token_time': 3.5,
                'ctx_server_first_token_time': 3.6,
                'gen_server_arrival_time': float('nan'),
                'disagg_server_arrival_time': float('nan'),
                'step_metrics': [{
                    'iter': 1,
                    'token_time': 4.0
                }],  # E2E = 1.0s
                'ctx_preprocessing_time': 0.1,
                'ctx_queue_time': 0.1,
                'ctx_processing_time': 0.3,
                'ctx_postprocessing_time': 0.1,
            },
            {
                'request_index': 1,
                'ctx_server_arrival_time': 1.0,  # Earliest arrival
                'ctx_arrival_time': 1.1,
                'ctx_first_scheduled_time': 1.2,
                'ctx_first_token_time': 1.5,
                'ctx_server_first_token_time': 1.6,
                'gen_server_arrival_time': float('nan'),
                'disagg_server_arrival_time': float('nan'),
                'step_metrics': [{
                    'iter': 1,
                    'token_time': 3.5
                }],  # E2E = 2.5s (longest)
                'ctx_preprocessing_time': 0.1,
                'ctx_queue_time': 0.1,
                'ctx_processing_time': 0.3,
                'ctx_postprocessing_time': 0.1,
            },
            {
                'request_index': 2,
                'ctx_server_arrival_time': 2.0,  # Middle arrival
                'ctx_arrival_time': 2.1,
                'ctx_first_scheduled_time': 2.2,
                'ctx_first_token_time': 2.5,
                'ctx_server_first_token_time': 2.6,
                'gen_server_arrival_time': float('nan'),
                'disagg_server_arrival_time': float('nan'),
                'step_metrics': [{
                    'iter': 1,
                    'token_time': 2.8
                }],  # E2E = 0.8s (shortest)
                'ctx_preprocessing_time': 0.1,
                'ctx_queue_time': 0.1,
                'ctx_processing_time': 0.3,
                'ctx_postprocessing_time': 0.1,
            }
        ]

    def test_create_diagram_with_sort_by_arrival(self):
        """Test that create_timing_diagram accepts sort_by='arrival' parameter."""
        analyzer = RequestTimeBreakdown()

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            # Should not raise any errors
            analyzer.create_timing_diagram(self.timing_data,
                                           temp_file,
                                           sort_by='arrival')
            self.assertTrue(os.path.exists(temp_file))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_create_diagram_with_sort_by_e2e(self):
        """Test that create_timing_diagram accepts sort_by='e2e' parameter."""
        analyzer = RequestTimeBreakdown()

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            # Should not raise any errors
            analyzer.create_timing_diagram(self.timing_data,
                                           temp_file,
                                           sort_by='e2e')
            self.assertTrue(os.path.exists(temp_file))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_create_diagram_with_max_requests(self):
        """Test that create_timing_diagram accepts max_requests parameter."""
        analyzer = RequestTimeBreakdown()

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            # Should not raise any errors
            analyzer.create_timing_diagram(self.timing_data,
                                           temp_file,
                                           max_requests=2)
            self.assertTrue(os.path.exists(temp_file))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_create_diagram_with_max_requests_and_sort(self):
        """Test that create_timing_diagram accepts both max_requests and sort_by."""
        analyzer = RequestTimeBreakdown()

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            # Should not raise any errors
            analyzer.create_timing_diagram(self.timing_data,
                                           temp_file,
                                           max_requests=2,
                                           sort_by='e2e')
            self.assertTrue(os.path.exists(temp_file))
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestPagination(unittest.TestCase):
    """Test pagination functionality."""

    def test_pagination_threshold_calculation(self):
        """Test that pagination is triggered based on data point threshold."""
        # Create data with many steps to trigger pagination
        test_data = []
        for i in range(100):
            test_data.append({
                'request_id': i,
                'perf_metrics': {
                    'timing_metrics': {
                        'server_arrival_time': float(i),
                        'arrival_time': float(i) + 0.1,
                        'first_scheduled_time': float(i) + 0.2,
                        'first_token_time': float(i) + 0.5,
                        'server_first_token_time': float(i) + 0.6
                    }
                },
                'time_breakdown_metrics': {
                    'step_metrics': [
                        {
                            'iter': j,
                            'token_time': float(i) + 0.5 + j * 0.01
                        } for j in range(1, 101)  # 100 steps per request
                    ]
                }
            })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        try:
            analyzer = RequestTimeBreakdown()
            timing_data = analyzer.parse_json_file(temp_file)

            # Total data points = 100 requests * 100 steps + 100 context = 10100
            # Should exceed 10000 threshold
            total_steps = sum(
                len(data.get('step_metrics', []) or []) for data in timing_data)
            total_data_points = total_steps + len(timing_data)
            self.assertGreater(total_data_points, 10000)
        finally:
            os.unlink(temp_file)


class TestContextGPUFallback(unittest.TestCase):
    """Test context GPU time fallback for non-disagg mode."""

    def test_ctx_gpu_from_first_step(self):
        """Test that ctx_gpu_forward_time uses first step's GPU time when not provided."""
        parser = RequestDataParser()

        # Non-disagg mode without ctx_gpu times but with step_metrics
        request_data = {
            'request_id': 'req_fallback',
            'perf_metrics': {
                'timing_metrics': {
                    'server_arrival_time': 1.0,
                    'arrival_time': 1.1,
                    'first_scheduled_time': 1.2,
                    'first_token_time': 1.5,
                    'server_first_token_time': 1.6
                }
            },
            'time_breakdown_metrics': {
                'step_metrics': [
                    {
                        'iter': 1,
                        'forward_start_time': 1.21,
                        'forward_end_time': 1.35,
                        'gpu_forward_time': 15.0,  # This should be used as ctx
                        'gpu_sample_time': 0.5,
                        'token_time': 1.45
                    },
                    {
                        'iter': 2,
                        'gpu_forward_time': 5.0,
                        'gpu_sample_time': 0.3,
                        'token_time': 1.60
                    }
                ]
                # Note: no ctx_gpu_forward_time or ctx_gpu_sample_time
            }
        }

        parsed = parser.parse_request(request_data, 0)

        # ctx_gpu_forward_time should be None (fallback happens in create_figure)
        self.assertIsNone(parsed['ctx_gpu_forward_time'])

        # step_metrics should be present for fallback logic
        self.assertIsNotNone(parsed['step_metrics'])
        self.assertEqual(parsed['step_metrics'][0]['gpu_forward_time'], 15.0)


if __name__ == '__main__':
    unittest.main()
