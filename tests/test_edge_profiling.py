"""
Phase 10 — Tests: Edge Hardware Profiling
Tests the formatting and logical components of the edge profiling scripts
without requiring full model inference or TF compilation.
"""

import pytest
import numpy as np
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.evaluation.edge_profiling import format_bytes, profile_inference


class TestEdgeProfiling:
    def test_format_bytes_kb(self):
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(500 * 1024) == "500.0 KB"

    def test_format_bytes_mb(self):
        assert format_bytes(1024 * 1024) == "1.00 MB"
        assert format_bytes(2.5 * 1024 * 1024) == "2.50 MB"

    def test_profile_inference_outputs(self):
        # Dummy predict function that sleeps slightly
        import time
        def dummy_predict(x):
            time.sleep(0.001)
            return x * 2

        x_sample = np.ones((1, 5))
        latency, ram = profile_inference(dummy_predict, x_sample, n_runs=10)
        
        # Latency should be at least 1ms (0.001s * 1000)
        assert latency >= 1.0
        # Peak RAM should be non-negative and likely very small
        assert ram >= 0.0
