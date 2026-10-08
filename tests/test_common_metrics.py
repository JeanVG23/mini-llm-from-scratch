import numpy as np

from common.metrics import (
    cross_entropy,
    log_softmax,
    mean_loss,
    perplexity,
    uniform_loss,
)


def test_log_softmax_rows_are_normalised():
    logits = np.random.default_rng(0).normal(size=(5, 7))
    assert np.allclose(np.exp(log_softmax(logits)).sum(axis=-1), 1.0)


def test_log_softmax_survives_huge_logits():
    huge = np.array([[1000.0, 1001.0, 999.0]])
    result = log_softmax(huge)
    assert np.all(np.isfinite(result))
    assert np.allclose(result, log_softmax(np.array([[0.0, 1.0, -1.0]])))  # invariant par translation


def test_cross_entropy_of_uniform_logits_is_log_vocab():
    logits = np.zeros((4, 10))
    targets = np.array([0, 3, 5, 9])
    assert np.isclose(cross_entropy(logits, targets), np.log(10))


def test_uniform_loss_and_perplexity_are_consistent():
    assert np.isclose(perplexity(uniform_loss(65)), 65.0)


def test_mean_loss_does_not_depend_on_the_chunk_size():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(23, 6))
    y = rng.integers(0, 6, size=23)
    whole = mean_loss(lambda xc: xc, x, y, chunk=1000)
    chunked = mean_loss(lambda xc: xc, x, y, chunk=5)
    assert np.isclose(whole, chunked)
    assert np.isclose(whole, cross_entropy(x, y))
