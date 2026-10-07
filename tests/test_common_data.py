import hashlib
from pathlib import Path

import numpy as np
import pytest

from common.data import (
    CharTokenizer,
    eval_windows,
    get_batch,
    load_splits,
    sha256_of,
    split_text,
    verify_sha256,
)

STEPS_DIR = Path(__file__).parent.parent / "steps"


def test_split_text_is_contiguous_80_10_10():
    text = "abcdefghij" * 100
    train, dev, test = split_text(text)
    assert (len(train), len(dev), len(test)) == (800, 100, 100)
    assert train + dev + test == text


def test_verify_sha256_accepts_matching_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("bonjour", encoding="utf-8")
    verify_sha256(path, hashlib.sha256(b"bonjour").hexdigest())


def test_verify_sha256_rejects_corrupted_file(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("bonjour", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_sha256(path, hashlib.sha256(b"autre chose").hexdigest())


def test_load_splits_returns_only_train_and_dev(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij" * 100, encoding="utf-8")
    splits = load_splits(path, expected_sha256=sha256_of(path))
    assert len(splits) == 2
    assert [len(s) for s in splits] == [800, 100]


def test_load_splits_refuses_a_truncated_corpus(tmp_path):
    path = tmp_path / "corpus.txt"
    path.write_text("abcdefghij" * 100, encoding="utf-8")
    expected = sha256_of(path)
    path.write_text("abcdefghij" * 50, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_splits(path, expected_sha256=expected)


def test_tokenizer_round_trip_and_sorted_vocab():
    tokenizer = CharTokenizer("banane")
    assert tokenizer.chars == ["a", "b", "e", "n"]
    assert tokenizer.vocab_size == 4
    ids = tokenizer.encode("banane")
    assert ids.dtype == np.int64
    assert tokenizer.decode(ids) == "banane"


def test_tokenizer_rejects_unknown_character():
    tokenizer = CharTokenizer("abc")
    with pytest.raises(ValueError, match="absents du vocabulaire"):
        tokenizer.encode("abz")


def test_get_batch_shapes_and_shift():
    ids = np.arange(100)
    x, y = get_batch(ids, batch_size=4, block_size=8, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (4, 8)
    assert np.array_equal(y, x + 1)  # ids = 0, 1, 2... : le caractère suivant vaut x + 1


def test_get_batch_can_reach_the_last_character():
    ids = np.arange(10)
    _, y = get_batch(ids, batch_size=2000, block_size=3, rng=np.random.default_rng(0))
    assert y.max() == 9  # la dernière fenêtre possible se termine sur le dernier identifiant


def test_get_batch_is_deterministic_for_a_seed():
    ids = np.arange(100)
    a = get_batch(ids, 4, 8, np.random.default_rng(7))
    b = get_batch(ids, 4, 8, np.random.default_rng(7))
    assert np.array_equal(a[0], b[0])


def test_get_batch_rejects_text_shorter_than_context():
    with pytest.raises(ValueError, match="trop court"):
        get_batch(np.arange(5), batch_size=2, block_size=8, rng=np.random.default_rng(0))


def test_eval_windows_covers_every_position_with_full_context():
    ids = np.arange(20)
    x, y = eval_windows(ids, block_size=4)
    assert x.shape == (16, 4)
    assert y.shape == (16,)
    assert y[0] == 4
    assert np.array_equal(y, x[:, -1] + 1)


def test_eval_windows_rejects_text_shorter_than_context():
    with pytest.raises(ValueError, match="trop court"):
        eval_windows(np.arange(8), block_size=8)


def test_only_step_9_may_read_the_test_split():
    offenders = [p.name for p in STEPS_DIR.glob("*.py") if "load_test" in p.read_text(encoding="utf-8") and not p.name.startswith("s09_")]
    assert offenders == []
