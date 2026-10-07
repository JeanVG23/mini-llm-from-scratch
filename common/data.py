"""Données : corpus Tiny Shakespeare vérifié, découpage fixe, tokenizer par caractères, lots."""

import hashlib
import urllib.request
from collections.abc import Iterable
from pathlib import Path

import numpy as np

CORPUS_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
CORPUS_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
CORPUS_PATH = Path("data/tinyshakespeare.txt")

# Toutes les mesures sur `dev` portent sur les mêmes positions : celles qui ont au moins
# EVAL_CONTEXT caractères de contexte. Un bigramme et un modèle à contexte 8 se comparent ainsi
# sur les mêmes cibles.
EVAL_CONTEXT = 8


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    actual = sha256_of(path)
    if actual != expected:
        raise ValueError(f"{path} : SHA-256 {actual}, attendu {expected} (fichier corrompu, tronqué ou modifié)")


def download_corpus(path: Path = CORPUS_PATH, expected_sha256: str = CORPUS_SHA256) -> Path:
    """Télécharge le corpus s'il est absent, puis vérifie sa somme SHA-256."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CORPUS_URL, timeout=30) as response:
            path.write_bytes(response.read())
    verify_sha256(path, expected_sha256)
    return path


def split_text(text: str) -> tuple[str, str, str]:
    """Trois blocs contigus : train 80 %, dev 10 %, test 10 %. Pas de mélange."""
    n = len(text)
    i, j = int(0.8 * n), int(0.9 * n)
    return text[:i], text[i:j], text[j:]


def load_splits(path: Path = CORPUS_PATH, expected_sha256: str = CORPUS_SHA256) -> tuple[str, str]:
    """Renvoie (train, dev). Le jeu test n'est volontairement pas renvoyé : voir load_test."""
    text = download_corpus(path, expected_sha256).read_text(encoding="utf-8")
    train, dev, _ = split_text(text)
    return train, dev


def load_test(path: Path = CORPUS_PATH, expected_sha256: str = CORPUS_SHA256) -> str:
    """Réservé à l'étape 9, une seule fois. Un test vérifie qu'aucune autre étape ne l'appelle."""
    text = download_corpus(path, expected_sha256).read_text(encoding="utf-8")
    return split_text(text)[2]


class CharTokenizer:
    """Un identifiant par caractère distinct du texte d'entraînement, dans l'ordre alphabétique."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self._index = {c: i for i, c in enumerate(self.chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> np.ndarray:
        unknown = sorted(set(text) - self._index.keys())
        if unknown:
            raise ValueError(f"caractères absents du vocabulaire : {unknown}")
        return np.array([self._index[c] for c in text], dtype=np.int64)

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.chars[i] for i in ids)


def get_batch(ids: np.ndarray, batch_size: int, block_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Tire batch_size fenêtres de block_size caractères. y est x décalé d'un cran."""
    if len(ids) <= block_size:
        raise ValueError(f"texte trop court ({len(ids)} identifiants) pour un contexte de {block_size}")
    starts = rng.integers(0, len(ids) - block_size, size=batch_size)
    positions = starts[:, None] + np.arange(block_size)  # (B, T)
    return ids[positions], ids[positions + 1]  # x (B, T), y (B, T) : y[b, t] suit x[b, t]


def eval_windows(ids: np.ndarray, block_size: int = EVAL_CONTEXT) -> tuple[np.ndarray, np.ndarray]:
    """Toutes les positions ayant block_size caractères de contexte : x (N, T) et la cible y (N,)."""
    n = len(ids) - block_size
    if n <= 0:
        raise ValueError(f"texte trop court ({len(ids)} identifiants) pour un contexte de {block_size}")
    x = ids[np.arange(n)[:, None] + np.arange(block_size)]  # (N, T)
    return x, ids[block_size:]  # y (N,) : le caractère qui suit chaque fenêtre
