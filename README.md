# mini_llm : un LLM construit pas à pas

Un petit modèle de langage de type GPT, construit étape par étape, du bigramme à GPT, avec un rapport chiffré par étape, échecs compris. L'objectif est **pédagogique** : comprendre chaque mécanisme (rétropropagation, attention, optimiseur, tokenizer) avant de passer au suivant, puis valider l'architecture sur les poids de GPT-2 small.

**Accès rapide** : [Méthode de travail](#méthode-de-travail) · [Parcours](#parcours) · [Démarrage rapide](#démarrage-rapide) · [English summary](#english-summary) · [Limites](#limites)

## Méthode de travail

Je n'ai pas écrit le code de ce dépôt : il est écrit par un assistant IA sous ma direction. Ma part : la conception (ordre des étapes, frontière du « from scratch », choix de ce que l'on mesure), le protocole d'évaluation, la relecture critique des résultats, et la compréhension approfondie de chaque mécanisme.

### Le dialogue en parallèle : un apprentissage actif

Chaque étape s'accompagne d'un échange continu et approfondi avec l'assistant (qui adopte la posture d'un professeur exigeant, pas d'un simple exécutant) :
* **Décortiquer les fondements** : comprendre le sens mathématique et l'intuition derrière chaque formule (pourquoi une loss, pourquoi telle forme de tenseur, comment fonctionne l'entropie croisée).
* **Replacer dans l'histoire** : situer chaque progrès dans la chronologie de l'IA (des chaînes de Markov de 1913 aux Transformers de 2017) pour comprendre quelle impasse historique chaque brique vient débloquer.
* **Challenger les compromis** : analyser les limites concrètes de chaque choix technique (malédiction de la dimensionnalité, lissage de Laplace, absence de fuite entre jeux de données).
* **Porte de compréhension** : aucun composant n'est validé sans que ses rouages ne soient maîtrisés ; chaque étape se conclut par trois questions de synthèse auxquelles je dois savoir répondre sans consulter le dépôt.

### Le cycle d'une étape

Chaque étape suit un protocole expérimental strict en cinq temps :
1. **Question et prédiction chiffrée** : formulées et écrites avant toute mesure (une prédiction fausse est un résultat utile).
2. **Tests de correction d'abord** : écrits avant le code, exécutés sur CPU et données synthétiques sans accès réseau (gradient numérique contre analytique, contrôle de causalité, formes).
3. **Mesures étanches** : exécutées sur le jeu de développement (`dev`) sur 3 graines (moyenne et écart-type) ; le jeu `test` reste scellé via une fonction dédiée jusqu'à l'étape 9.
4. **Rapport d'expérience** : consigné dans [`experiments/`](experiments/), incluant les échecs, les anomalies observées et les limites de validité.
5. **Validation des acquis** : passage de la porte de compréhension avant de passer à l'étape suivante.

## Parcours

Spec de référence : [`docs/superpowers/specs/2026-10-06-mini-llm-design.md`](docs/superpowers/specs/2026-10-06-mini-llm-design.md).

| N° | Étape | Pile | Statut |
|---|---|---|---|
| 1 | Bigramme par comptage (plancher de référence) | numpy | fait ([rapport](experiments/s01-bigram-counts.md)) |
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

A small GPT-style language model built step by step, from a count-based bigram to a GPT, with a measured write-up (in French) for every step, failures included. Steps 1 to 4 use numpy with hand-written backpropagation; from step 5 on, PyTorch provides tensors and autograd only, while layers, optimizer, tokenizer and training loop stay hand-written. The final step loads GPT-2 small weights into our own architecture and checks that the logits match the reference. The code is written by an AI assistant under the direction of the repository owner, who designed the curriculum and the evaluation protocol, engages in continuous interactive dialogue to inspect every mechanism and mathematical foundation in detail, and reviews every result.

## Limites

* Un seul corpus d'environ 1 Mo : aucune conclusion ne se généralise à d'autres textes.
* Un seul auteur de l'évaluation, sans second relecteur.
* Un surapprentissage probable dès que le modèle grossit : la loss sur `dev` remontera. Ce sera documenté comme un résultat.
* Les chiffres des rapports sont donnés avec leur jeu (`train` ou `dev`) et leurs paramètres ; ne jamais en citer un sans eux.
