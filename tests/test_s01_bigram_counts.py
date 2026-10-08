import numpy as np
import pytest

from common.data import CharTokenizer
from steps.s01_bigram_counts import bigram_log_probs, bigram_loss, count_bigrams, sample

IDS = np.array([0, 1, 0, 1, 1])  # paires : (0,1) (1,0) (0,1) (1,1)


def test_count_bigrams_counts_each_pair():
    counts = count_bigrams(IDS, vocab_size=2)
    assert counts.tolist() == [[0, 2], [1, 1]]


def test_rows_are_probability_distributions():
    for alpha in (0.0, 1.0):
        log_probs = bigram_log_probs(count_bigrams(IDS, 2), alpha)
        assert np.allclose(np.exp(log_probs).sum(axis=1), 1.0)


def test_maximum_likelihood_matches_the_hand_computation():
    probs = np.exp(bigram_log_probs(count_bigrams(IDS, 2), alpha=0.0))
    assert np.allclose(probs, [[0.0, 1.0], [0.5, 0.5]])


def test_uniform_log_probs_give_log_vocab_loss():
    uniform = np.full((4, 4), np.log(1 / 4))
    loss = bigram_loss(uniform, np.array([0, 1, 2]), np.array([3, 2, 1]))
    assert np.isclose(loss, np.log(4))


def test_unseen_pair_is_infinite_without_smoothing_and_finite_with_it():
    counts = count_bigrams(IDS, 2)  # la paire (0, 0) n'apparaît jamais
    previous, following = np.array([0]), np.array([0])
    assert bigram_loss(bigram_log_probs(counts, 0.0), previous, following) == np.inf
    assert np.isfinite(bigram_loss(bigram_log_probs(counts, 1.0), previous, following))


def test_maximum_likelihood_is_the_best_bigram_on_train():
    counts = count_bigrams(IDS, 2)
    previous, following = IDS[:-1], IDS[1:]
    mle_loss = bigram_loss(bigram_log_probs(counts, 0.0), previous, following)
    smoothed_loss = bigram_loss(bigram_log_probs(counts, 1.0), previous, following)
    assert mle_loss < smoothed_loss


def test_negative_alpha_is_rejected():
    with pytest.raises(ValueError, match="alpha"):
        bigram_log_probs(count_bigrams(IDS, 2), alpha=-1.0)


def test_sample_is_deterministic_for_a_seed_and_stays_in_vocabulary():
    tokenizer = CharTokenizer("ab")
    log_probs = bigram_log_probs(count_bigrams(IDS, 2), alpha=1.0)
    first = sample(log_probs, tokenizer, 50, np.random.default_rng(3), start_id=0)
    second = sample(log_probs, tokenizer, 50, np.random.default_rng(3), start_id=0)
    assert first == second
    assert len(first) == 50
    assert set(first) <= {"a", "b"}
