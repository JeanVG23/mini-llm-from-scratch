"""Étape 1 : bigramme par comptage. Plancher de référence (équivalent du « v0 naïf » du RAG).

Un bigramme prédit le caractère suivant à partir du seul caractère précédent, en comptant
les paires du texte d'entraînement. Aucun apprentissage : juste des comptes.

Usage : uv run python -m steps.s01_bigram_counts --seed 0 --alpha 1.0
"""

import argparse

import numpy as np

from common.data import EVAL_CONTEXT, CharTokenizer, eval_windows, load_splits
from common.metrics import perplexity, uniform_loss
from common.results import append_result


def count_bigrams(ids: np.ndarray, vocab_size: int) -> np.ndarray:
    """counts[i, j] = nombre de fois où le caractère j suit le caractère i."""
    counts = np.zeros((vocab_size, vocab_size))
    np.add.at(counts, (ids[:-1], ids[1:]), 1)
    return counts


def bigram_log_probs(counts: np.ndarray, alpha: float) -> np.ndarray:
    """log P(suivant | précédent), avec un lissage additif alpha (chaque paire reçoit alpha comptes de plus).

    alpha = 0 donne le maximum de vraisemblance : une paire jamais vue a une probabilité nulle
    (log = -inf), et une ligne sans aucun compte donne nan. Il faut alpha > 0 pour échantillonner.
    """
    if alpha < 0:
        raise ValueError(f"alpha doit être positif ou nul, reçu {alpha}")
    smoothed = counts + alpha  # (V, V)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))


def bigram_loss(log_probs: np.ndarray, previous: np.ndarray, following: np.ndarray) -> float:
    """Entropie croisée moyenne (en nats) : -log P(following | previous), moyennée sur les paires."""
    return float(-log_probs[previous, following].mean())


def sample(log_probs: np.ndarray, tokenizer: CharTokenizer, length: int, rng: np.random.Generator, start_id: int) -> str:
    ids = [start_id]
    for _ in range(length - 1):
        probs = np.exp(log_probs[ids[-1]])  # (V,)
        ids.append(int(rng.choice(len(probs), p=probs)))
    return tokenizer.decode(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="ne sert qu'à l'échantillonnage : les comptes sont déterministes")
    parser.add_argument("--alpha", type=float, default=1.0, help="lissage additif (1.0 = lissage de Laplace)")
    args = parser.parse_args()

    train_text, dev_text = load_splits()
    tokenizer = CharTokenizer(train_text)
    train_ids, dev_ids = tokenizer.encode(train_text), tokenizer.encode(dev_text)

    counts = count_bigrams(train_ids, tokenizer.vocab_size)
    log_probs = bigram_log_probs(counts, args.alpha)
    mle = bigram_log_probs(counts, 0.0)

    result = {
        "seed": args.seed,
        "alpha": args.alpha,
        "vocab_size": tokenizer.vocab_size,
        "eval_context": EVAL_CONTEXT,
        "uniform_loss": uniform_loss(tokenizer.vocab_size),
    }
    for name, ids in (("train", train_ids), ("dev", dev_ids)):
        x, y = eval_windows(ids)
        loss = bigram_loss(log_probs, x[:, -1], y)  # le bigramme n'utilise que le dernier caractère du contexte
        result[f"{name}_loss"] = loss
        result[f"{name}_perplexity"] = perplexity(loss)
    x, y = eval_windows(train_ids)
    result["train_loss_mle"] = bigram_loss(mle, x[:, -1], y)  # le meilleur bigramme possible sur train

    rng = np.random.default_rng(args.seed)
    start_id = int(tokenizer.encode("\n")[0])
    for key, value in result.items():
        print(f"{key:>16} : {value}")
    print("\n--- échantillon (300 caractères) ---")
    print(sample(log_probs, tokenizer, 300, rng, start_id))
    print(f"\nrésultats ajoutés à {append_result('s01', result)}")


if __name__ == "__main__":
    main()
