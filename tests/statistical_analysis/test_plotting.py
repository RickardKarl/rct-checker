"""Tests for plotting module."""

import numpy as np

from src.statistical_analysis.plotting import plot_test_output


def test_plot_test_output_saves_figure(tmp_path):
    """Test that plot_test_output saves a valid figure to disk."""
    test_output = {
        "cont_chi_squared_variance": {
            "p_value": 0.5,
            "test_statistic": 10.0,
            "zscores": np.random.randn(20),
        },
        "fisher_method-combined": {"p_value": 0.3, "test_statistic": 5.0},
    }

    save_path = tmp_path / "test_plot.png"
    plot_test_output(test_output, save_path=str(save_path))

    assert save_path.exists()
    assert save_path.stat().st_size > 0
