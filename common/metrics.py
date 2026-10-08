"""Métriques : log-softmax stable, entropie croisée, perplexité, plancher uniforme."""

from collections.abc import Callable

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """log-softmax sur le dernier axe. On retire le maximum avant exp pour éviter tout débordement."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> float:
    """Entropie croisée moyenne, en nats. logits (N, V), targets (N,)."""
    log_probs = log_softmax(logits)
    return float(-log_probs[np.arange(len(targets)), targets].mean())


def perplexity(loss: float) -> float:
    return float(np.exp(loss))


def uniform_loss(vocab_size: int) -> float:
    """Loss d'un modèle qui donne la même probabilité à chaque caractère : le premier plancher."""
    return float(np.log(vocab_size))


def mean_loss(logits_fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray, chunk: int = 65536) -> float:
    """Entropie croisée moyenne de logits_fn sur (x, y), calculée par morceaux pour limiter la mémoire."""
    total = 0.0
    for start in range(0, len(y), chunk):
        y_chunk = y[start : start + chunk]
        total += cross_entropy(logits_fn(x[start : start + chunk]), y_chunk) * len(y_chunk)
    return total / len(y)
