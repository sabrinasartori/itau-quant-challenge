import pandas as pd
from scipy.stats import norm
import numpy as np

def get_returns_pvalue(
    returns : pd.Series,
    expected_sr : float = 0
)-> float:
    """Given the returns series, this method returns the pvalue of
    the null hypothesis sharpe ratio > expected)_sr.

    Based on probabilistic sharpe ratio from Lopez de Prado's book

    Args:
        returns (pd.Series): _description_
        expected_sr (float, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """    
    sr = returns.mean() / returns.std()
    T = len(returns)

    gamma_3 = returns.skew()
    gamma_4 = returns.kurtosis()

    denominator =  1 - gamma_3 * sr + (gamma_4 - 1)/4 * sr**2
    denominator = np.sqrt(denominator)

    psr = (sr - expected_sr) * np.sqrt(T - 1) / denominator
    psr = norm.cdf(psr)

    return 1- psr