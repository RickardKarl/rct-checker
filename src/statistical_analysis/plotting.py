import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, probplot

logger = logging.getLogger(__name__)


def _plot_zscore_histogram(ax, zscores):
    """
    Plot histogram of z-scores with standard normal curve overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    zscores : array-like
        Array of z-scores to visualize.
    """
    ax.hist(zscores, bins=20, density=True, alpha=0.6, label="Z-scores")
    x = np.linspace(min(zscores), max(zscores), 300)
    y = norm.pdf(x, 0, 1)
    ax.plot(x, y, "r-", linewidth=2, label="Standard Normal")
    ax.set_xlabel("Z-score")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Z-scores")
    ax.legend()


def _plot_zscore_qq(ax, zscores):
    """
    Plot Q-Q plot for z-scores against standard normal distribution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    zscores : array-like
        Array of z-scores to visualize.
    """
    probplot(zscores, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot for Z-scores")
    ax.grid(True, alpha=0.3)


def _plot_odds_ratios(ax, log_odds_ratios):
    """
    Plot histogram of log odds ratios with standard normal curve overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on.
    log_odds_ratios : array-like
        Array of log odds ratios to visualize.
    """
    ax.hist(
        log_odds_ratios,
        bins=20,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Observed values",
    )
    x = np.linspace(min(log_odds_ratios), max(log_odds_ratios), 300)
    y = norm.pdf(x, 0, 1)
    ax.plot(x, y, "r-", linewidth=2, label="Standard Normal")
    ax.set_xlabel("Logarithm of odds ratios")
    ax.set_ylabel("Count")
    ax.set_title("Odds ratios from Fisher's exact test")
    ax.tick_params(axis="x", rotation=45)


def plot_test_output(test_output: dict, save_path: str = None):
    """
    Plot visualization of statistical test results.

    Generates histograms and Q-Q plots for z-scores from chi-squared variance
    tests, and histograms for log odds ratios from Fisher's exact tests.

    Parameters
    ----------
    test_output : dict
        Dictionary containing test results from run_test_pipeline.
    save_path : str, optional
        Path to save the plot image. If None, displays the plot interactively.

    Raises
    ------
    RuntimeError
        If no tests are found or no plots can be generated.
    """
    tests_performed = list(test_output.keys())

    if len(tests_performed) == 0:
        raise RuntimeError("No tests found")

    plot_zscores = "cont_chi_squared_variance" in tests_performed

    fisher_test_keys = [
        key
        for key in tests_performed
        if key.startswith("fisher_test") and test_output[key]["test_statistic_is_odds_ratio"]
    ]
    plot_fisher_test = len(fisher_test_keys) > 0

    if not plot_zscores and not plot_fisher_test:
        raise RuntimeError(
            f"No valid tests found in test_output. If you see fisher_test-*, it is likely because we have multiple groups in data and thus have not computed any odds ratio to plot. Check test_output = {test_output.keys()}."
        )

    # Determine number of subplots needed
    num_plots = 0
    if plot_zscores:
        num_plots += 2  # Z-score histogram + Q-Q plot
    if plot_fisher_test:
        num_plots += 1  # Odds ratios histogram

    if num_plots == 0:
        raise RuntimeError("No plots to generate")

    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    ax_idx = 0

    if plot_zscores:
        zscores = test_output["cont_chi_squared_variance"]["zscores"]
        _plot_zscore_histogram(axes[ax_idx], zscores)
        ax_idx += 1
        _plot_zscore_qq(axes[ax_idx], zscores)
        ax_idx += 1

    if plot_fisher_test:
        odds_ratios = np.array([test_output[test]["test_statistic"] for test in fisher_test_keys])
        n_total = len(odds_ratios)

        # Filter out invalid values (zeros and infinities)
        n_zeros = np.sum(odds_ratios == 0)
        n_inf = np.sum(np.isinf(odds_ratios))
        valid_mask = (odds_ratios > 0) & ~np.isinf(odds_ratios)
        odds_ratios = odds_ratios[valid_mask]

        if n_zeros > 0:
            logger.warning(
                f"Removed {n_zeros} zero odds ratio(s) from plot (out of {n_total} total)"
            )
        if n_inf > 0:
            logger.warning(
                f"Removed {n_inf} infinite odds ratio(s) from plot (out of {n_total} total)"
            )

        if len(odds_ratios) == 0:
            logger.warning("All odds ratios are zero or infinite, skipping odds ratio plot")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                logger.info(f"Plot saved to {save_path}")
            else:
                plt.show()
            return

        log_odds_ratios = np.log(odds_ratios)
        _plot_odds_ratios(axes[ax_idx], log_odds_ratios)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()
