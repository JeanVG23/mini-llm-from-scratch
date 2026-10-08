"""Contrôle de gradient : compare un gradient écrit à la main à des différences finies centrées."""

from collections.abc import Callable

import numpy as np


def numerical_grad(loss_fn: Callable[[], float], param: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Gradient de loss_fn() par rapport à param, par différences finies centrées.

    loss_fn ne prend aucun argument : elle relit param, que l'on perturbe en place, un coefficient à la fois.
    """
    grad = np.zeros_like(param)
    for idx in np.ndindex(param.shape):
        original = param[idx]
        param[idx] = original + eps
        loss_plus = loss_fn()
        param[idx] = original - eps
        loss_minus = loss_fn()
        param[idx] = original
        grad[idx] = (loss_plus - loss_minus) / (2 * eps)
    return grad


def relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """Écart relatif entre deux gradients, en norme : robuste aux coefficients proches de zéro."""
    gap = np.linalg.norm(analytic - numeric)
    scale = max(np.linalg.norm(analytic) + np.linalg.norm(numeric), 1e-12)
    return float(gap / scale)
