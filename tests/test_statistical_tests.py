"""
Phase 11 — Tests: Statistical Tests & Results Compilation
Tests McNemar test math and table compilation structure.
"""

import pytest
import numpy as np
import pandas as pd
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.evaluation.statistical_tests import (
    mcnemar_test,
    compile_table_I,
    compile_table_II,
    compile_table_III
)


class TestStatisticalTests:
    def test_mcnemar_test_identical(self):
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1])
        # Two identical models
        stat, pval = mcnemar_test(y_true, y_pred, y_pred)
        assert stat == 0.0
        assert pval == 1.0

    def test_mcnemar_test_different(self):
        y_true  = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1])
        y_pred1 = np.array([1, 1, 1, 0, 0, 1, 0, 0, 0, 0]) # 5 correct
        y_pred2 = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1]) # 10 correct
        
        stat, pval = mcnemar_test(y_true, y_pred1, y_pred2)
        # Model 2 gets 5 right that Model 1 got wrong. Model 1 gets 0 right that Model 2 got wrong.
        # b = 0, c = 5
        # chi2 = (|0 - 5| - 1)^2 / 5 = 16 / 5 = 3.2
        assert stat == 3.2
        assert 0.05 < pval < 0.10 # chi2 dist cdf

    def test_compile_table_I_returns_df(self):
        df = compile_table_I()
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "Method" in df.columns
            
    def test_compile_table_II_returns_df(self):
        df = compile_table_II()
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "Method" in df.columns

    def test_compile_table_III_returns_df(self):
        df = compile_table_III()
        assert isinstance(df, pd.DataFrame)
        if len(df) > 0:
            assert "Method" in df.columns
            assert "Hyperparameters" in df.columns
