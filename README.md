# mini_llm : un LLM construit pas à pas

Un petit modèle de langage de type GPT, construit étape par étape, du bigramme à GPT, avec un rapport chiffré par étape, échecs compris. L'objectif est **pédagogique** : comprendre chaque mécanisme (rétropropagation, attention, optimiseur, tokenizer) avant de passer au suivant, puis valider l'architecture sur les poids de GPT-2 small.

**Accès rapide** : [Méthode de travail](#méthode-de-travail) · [Parcours](#parcours) · [Démarrage rapide](#démarrage-rapide) · [English summary](#english-summary) · [Limites](#limites)

## Méthode de travail

Je n'ai pas écrit le code de ce dépôt : il est écrit par un assistant IA sous ma direction. Ma part : la conception (ordre des étapes, frontière du « from scratch », choix de ce que l'on mesure), le protocole d'évaluation, la relecture critique des résultats, et la compréhension détaillée de chaque mécanisme à travers des échanges continus et approfondis à chaque étape. Les mesures sont à lire avec les limites indiquées dans chaque rapport.

Chaque étape suit le même cycle : une question et une prédiction chiffrée écrites avant toute mesure, des tests de correction écrits avant le code, des mesures sur un jeu de développement (`dev`) avec un jeu `test` lu une seule fois, puis un rapport dans [`experiments/`](experiments/) qui garde les échecs.

## Parcours

Spec de référence : [`docs/superpowers/specs/2026-10-06-mini-llm-design.md`](docs/superpowers/specs/2026-10-06-mini-llm-design.md).

| N° | Étape | Pile | Statut |
|---|---|---|---|
| 1 | Bigramme par comptage (plancher de référence) | numpy | à faire |
| 2 | Bigramme neuronal, gradient écrit à la main | numpy | à faire |
| 3 | MLP à contexte fixe, rétropropagation à la main | numpy | à faire |
| 4 | Une tête d'attention causale, forward et backward à la main | numpy | à faire |
| 5 | Pont : le modèle de l'étape 4 en PyTorch, gradients identiques | PyTorch | à venir |
| 6 | Tokenizer BPE à la main | Python | à venir |
| 7 | Bloc Transformer (multi-têtes, LayerNorm, résiduelles, positions) | PyTorch | à venir |
| 8 | GPT miniature entraîné (AdamW à la main, échantillonnage) | PyTorch | à venir |
| 9 | Ablations, jeu test lu une seule fois | PyTorch | à venir |
| 10 | Validation : logits de GPT-2 small reproduits par notre architecture | PyTorch | à venir |

## Démarrage rapide

Prérequis : Python 3.11 ou plus, [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                                  # tests hermétiques, aucun réseau
uv run ruff check .                            # lint
uv run python -m steps.s01_bigram_counts       # télécharge Tiny Shakespeare dans data/ au premier lancement
```

Le corpus (Tiny Shakespeare, environ 1,1 million de caractères) est vérifié par somme SHA-256 : un fichier corrompu ou modifié est refusé.

## English summary

A small GPT-style language model built step by step, from a count-based bigram to a GPT, with a measured write-up (in French) for every step, failures included. Steps 1 to 4 use numpy with hand-written backpropagation; from step 5 on, PyTorch provides tensors and autograd only, while layers, optimizer, tokenizer and training loop stay hand-written. The final step loads GPT-2 small weights into our own architecture and checks that the logits match the reference. The code is written by an AI assistant under the direction of the repository owner, who designed the curriculum and the evaluation protocol, inspects every mechanism in detail and reviews every result.

## Limites

* Un seul corpus d'environ 1 Mo : aucune conclusion ne se généralise à d'autres textes.
* Un seul auteur de l'évaluation, sans second relecteur.
* Un surapprentissage probable dès que le modèle grossit : la loss sur `dev` remontera. Ce sera documenté comme un résultat.
* Les chiffres des rapports sont donnés avec leur jeu (`train` ou `dev`) et leurs paramètres ; ne jamais en citer un sans eux.
