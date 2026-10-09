import numpy as np

from steps.s02_bigram_neural import evaluate, loss_and_grad, train
from tests.gradcheck import numerical_grad, relative_error


def synthetic_ids(n=600, vocab_size=4, seed=0):
    return np.random.default_rng(seed).integers(0, vocab_size, size=n)


def mle_loss(ids, vocab_size):
    """Entropie croisée du meilleur bigramme sur ids, calculée directement à partir des comptes."""
    counts = np.zeros((vocab_size, vocab_size))
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    probs = counts / counts.sum(axis=1, keepdims=True)
    return float(-np.log(probs[ids[:-1], ids[1:]]).mean())


def test_gradient_matches_finite_differences():
    W = np.random.default_rng(0).normal(size=(5, 5))
    x = np.array([0, 1, 1, 2, 4, 1, 0])  # doublons exprès : np.add.at doit additionner les lignes
    y = np.array([1, 2, 0, 3, 1, 4, 2])
    _, dW = loss_and_grad(W, x, y)
    numeric = numerical_grad(lambda: loss_and_grad(W, x, y)[0], W)
    assert dW.shape == W.shape
    assert relative_error(dW, numeric) < 1e-6


def test_zero_weights_give_the_uniform_loss():
    loss, _ = loss_and_grad(np.zeros((6, 6)), np.array([0, 1, 2]), np.array([3, 4, 5]))
    assert np.isclose(loss, np.log(6))


def test_full_batch_descent_decreases_the_loss_every_step():
    ids = synthetic_ids()
    W = np.zeros((4, 4))
    losses = train(W, ids[:-1], ids[1:], steps=200, lr=1.0, batch_size=None, rng=np.random.default_rng(0))
    assert np.all(np.diff(losses) <= 0)


def test_converges_to_the_count_model_loss_and_not_below():
    ids = synthetic_ids()
    W = np.zeros((4, 4))
    train(W, ids[:-1], ids[1:], steps=3000, lr=1.0, batch_size=None, rng=np.random.default_rng(0))
    final, _ = loss_and_grad(W, ids[:-1], ids[1:])
    best = mle_loss(ids, 4)
    assert final >= best - 1e-9  # aucun bigramme ne fait mieux que le maximum de vraisemblance
    assert final - best < 1e-3


def test_minibatch_training_is_deterministic_for_a_seed():
    ids = synthetic_ids()
    results = []
    for _ in range(2):
        W = np.zeros((4, 4))
        train(W, ids[:-1], ids[1:], steps=50, lr=1.0, batch_size=32, rng=np.random.default_rng(5))
        results.append(W)
    assert np.array_equal(results[0], results[1])


def test_evaluate_scores_zero_weights_at_the_uniform_loss():
    assert np.isclose(evaluate(np.zeros((4, 4)), synthetic_ids(100)), np.log(4))
