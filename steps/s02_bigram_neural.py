"""Étape 2 : bigramme neuronal, gradient écrit à la main.

Même modèle que l'étape 1 (le caractère précédent prédit le suivant), mais appris par descente
de gradient : une table W (V, V) dont la ligne x donne les logits du caractère suivant.
Le gradient de l'entropie croisée est écrit à la main, puis contrôlé contre des différences finies
dans les tests.

Usage : uv run python -m steps.s02_bigram_neural --seed 0 --steps 20000 --lr 20 --batch-size 4096
"""

import argparse
import time

import numpy as np

from common.data import EVAL_CONTEXT, CharTokenizer, eval_windows, load_splits
from common.metrics import log_softmax, mean_loss, perplexity, uniform_loss
from common.results import append_result


def loss_and_grad(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """Entropie croisée moyenne et son gradient par rapport à W. x (N,) précédents, y (N,) suivants."""
    n = len(y)
    logits = W[x]  # (N, V) : une ligne de W par exemple
    log_probs = log_softmax(logits)  # (N, V)
    loss = float(-log_probs[np.arange(n), y].mean())

    dlogits = np.exp(log_probs)  # softmax
    dlogits[np.arange(n), y] -= 1  # softmax - one-hot : le gradient de l'entropie croisée
    dlogits /= n  # moyenne sur le lot
    dW = np.zeros_like(W)
    np.add.at(dW, x, dlogits)  # additionne les lignes quand un même x apparaît plusieurs fois
    return loss, dW


def train(W: np.ndarray, x: np.ndarray, y: np.ndarray, steps: int, lr: float, batch_size: int | None, rng: np.random.Generator) -> list[float]:
    """Descente de gradient. Modifie W en place. batch_size=None : lot complet à chaque pas."""
    losses = []
    for _ in range(steps):
        if batch_size is None:
            x_batch, y_batch = x, y
        else:
            picked = rng.integers(0, len(y), size=batch_size)
            x_batch, y_batch = x[picked], y[picked]
        loss, dW = loss_and_grad(W, x_batch, y_batch)
        W -= lr * dW
        losses.append(loss)
    return losses


def evaluate(W: np.ndarray, ids: np.ndarray) -> float:
    x, y = eval_windows(ids)
    return mean_loss(lambda x_chunk: W[x_chunk[:, -1]], x, y)  # le bigramme n'utilise que le dernier caractère


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=20.0)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    train_text, dev_text = load_splits()
    tokenizer = CharTokenizer(train_text)
    train_ids, dev_ids = tokenizer.encode(train_text), tokenizer.encode(dev_text)

    W = np.zeros((tokenizer.vocab_size, tokenizer.vocab_size))  # logits nuls : on part du modèle uniforme
    rng = np.random.default_rng(args.seed)
    started = time.perf_counter()
    losses = train(W, train_ids[:-1], train_ids[1:], args.steps, args.lr, args.batch_size, rng)
    elapsed = time.perf_counter() - started

    result = {
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_params": int(W.size),
        "train_seconds": round(elapsed, 2),
        "eval_context": EVAL_CONTEXT,
        "uniform_loss": uniform_loss(tokenizer.vocab_size),
        "last_batch_loss": losses[-1],
        "train_loss": evaluate(W, train_ids),
        "dev_loss": evaluate(W, dev_ids),
    }
    result["train_perplexity"] = perplexity(result["train_loss"])
    result["dev_perplexity"] = perplexity(result["dev_loss"])
    for key, value in result.items():
        print(f"{key:>16} : {value}")
    print(f"\nrésultats ajoutés à {append_result('s02', result)}")


if __name__ == "__main__":
    main()
