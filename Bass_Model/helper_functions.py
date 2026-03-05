"""
Helper functions for Bass diffusion analysis.
"""

import numpy as np


def bass_model(t, p, q, M):
    """
    Bass diffusion model for number of adopters per period.

    Parameters
    ----------
    t : array-like
        Time periods since product launch.
    p : float
        Coefficient of innovation.
    q : float
        Coefficient of imitation.
    M : float
        Market potential.

    Returns
    -------
    numpy array
        Predicted adopters per period.
    """
    exp_term = np.exp(-(p + q) * t)
    return M * ((p + q) ** 2 / p) * exp_term / (1 + (q / p) * exp_term) ** 2


def peak_adoption_time(p, q):
    """
    Calculate the time of peak adoption in the Bass model.

    Returns
    -------
    float
        Time of maximum adoption.
    """
    return np.log(q / p) / (p + q)