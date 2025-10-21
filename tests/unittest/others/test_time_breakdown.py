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

    def test_get_metrics_by_server(self):
        """Test retrieving metrics by server type."""
        config = TimingMetricsConfig()

        ctx_metrics = config.get_metrics_by_server('ctx')
        self.assertGreater(len(ctx_metrics), 0)

        # All returned metrics should be for 'ctx' server
        for metric in ctx_metrics:
            self.assertEqual(metric.server_type, 'ctx')

    def test_add_metric(self):
        """Test adding a new metric."""
        config = TimingMetricsConfig()
        initial_count = len(config.metrics)

        new_metric = TimingMetric(name='custom_metric',
                                  display_name='Custom Metric',
                                  color='red',
                                  description='Custom test metric',
                                  start_field='start',
                                  end_field='end')

        config.add_metric(new_metric)
        self.assertEqual(len(config.metrics), initial_count + 1)
        self.assertIsNotNone(config.get_metric_by_name('custom_metric'))

    def test_remove_metric(self):
        """Test removing a metric."""
        config = TimingMetricsConfig()
        initial_count = len(config.metrics)

        # Add a test metric first
        test_metric = TimingMetric(name='test_to_remove',
                                   display_name='Test',
                                   color='blue',
                                   description='Test',
                                   start_field='start',
                                   end_field='end')
        config.add_metric(test_metric)

        # Remove it
        config.remove_metric('test_to_remove')
        self.assertEqual(len(config.metrics), initial_count)
        self.assertIsNone(config.get_metric_by_name('test_to_remove'))


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

        # Gen metrics should be 0 in aggregated format
        self.assertEqual(parsed['gen_server_arrival_time'], 0)
        self.assertEqual(parsed['disagg_server_arrival_time'], 0)

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

        # All timing fields should default to 0
        self.assertEqual(parsed['ctx_server_arrival_time'], 0)
        self.assertEqual(parsed['ctx_arrival_time'], 0)
        self.assertEqual(parsed['gen_server_arrival_time'], 0)

    def test_parse_uses_index_as_fallback(self):
        """Test that index is used when request_id is missing."""
        parser = RequestDataParser()

        request_data = {'perf_metrics': {'timing_metrics': {}}}

        parsed = parser.parse_request(request_data, 42)

        self.assertEqual(parsed['request_index'], 42)


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
        # Create sample timing data
        timing_data = [{
            'request_index': 0,
            'ctx_preprocessing_time': 0.1,
            'ctx_queue_time': 0.2,
            'ctx_processing_time': 0.3,
            'ctx_postprocessing_time': 0.05,
            'gen_preprocessing_time': 0,
            'gen_queue_time': 0,
            'gen_postprocessing_time': 0,
            'disagg_preprocessing_time': 0,
            'disagg_postprocessing_time': 0,
        }, {
            'request_index': 1,
            'ctx_preprocessing_time': 0.15,
            'ctx_queue_time': 0.25,
            'ctx_processing_time': 0.35,
            'ctx_postprocessing_time': 0.06,
            'gen_preprocessing_time': 0,
            'gen_queue_time': 0,
            'gen_postprocessing_time': 0,
            'disagg_preprocessing_time': 0,
            'disagg_postprocessing_time': 0,
        }]

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            temp_file = f.name

        try:
            # Mock plotly to avoid actual file creation
            with patch(
                    'tensorrt_llm.serve.scripts.time_breakdown.time_breakdown.pyo.plot'
            ) as mock_plot:
                self.analyzer.create_timing_diagram(timing_data, temp_file)

                # Verify that plot was called
                mock_plot.assert_called_once()
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
        custom_config.add_metric(
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
        # Create test data
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

            # Mock the plot function to avoid actual file operations
            with patch(
                    'tensorrt_llm.serve.scripts.time_breakdown.time_breakdown.pyo.plot'
            ):
                analyzer.create_timing_diagram(timing_data, html_f.name)


if __name__ == '__main__':
    unittest.main()
