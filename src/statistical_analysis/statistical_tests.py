import numpy as np
from scipy.stats import chi2


def chi_square_variance_test(zscores, sigma0: float = 1.0):
    """
    Assumes zscores are i.i.d. and test variance.

    Args:
        zscores (_type_): z-scores assumed to be normally distributed
        sigma0 (float, optional): hypothesized standard deviation (e.g., 1 for standard normal). Defaults to 1.0.
    """
    if zscores is None or len(zscores) == 0:
        raise ValueError("zscores must not be empty")
    if len(zscores) < 2:
        raise ValueError("zscores must have at least 2 elements to compute variance")
    if np.any(np.isnan(zscores)) or np.any(np.isinf(zscores)):
        raise ValueError("zscores contains NaN or infinite values")
    if sigma0 <= 0:
        raise ValueError("sigma0 must be positive")

    sample_var = np.var(zscores, ddof=1)  # sample variance

    chi2_stat = (len(zscores) - 1) * sample_var / sigma0**2

    # p-value for two-tailed test
    p_value = 2 * min(
        chi2.cdf(chi2_stat, df=len(zscores) - 1),
        1 - chi2.cdf(chi2_stat, df=len(zscores) - 1),
    )

    return p_value, chi2_stat
