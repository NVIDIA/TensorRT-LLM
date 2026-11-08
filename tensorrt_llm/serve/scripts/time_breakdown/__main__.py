#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Entry point for running time_breakdown as a module.

Usage:
    python -m tensorrt_llm.serve.scripts.time_breakdown perf_metrics.json [options]
"""

from .time_breakdown import main

if __name__ == '__main__':
    main()
