# Étape 1 : bigramme par comptage

Rapport d'expérience du 8 octobre 2026. Code : `steps/s01_bigram_counts.py`.

## 1. Question

Que vaut un modèle qui ne regarde que le caractère précédent, et de combien bat-il le modèle uniforme sur `dev` ?

## 2. Prédiction (écrite avant toute mesure)

Le modèle uniforme sur 65 caractères produit une loss théorique de ln(65) ≈ 4,17 nats (perplexité 65,0). Pour le bigramme par comptage, la dépendance locale de l'anglais shakespearien doit réduire substantiellement l'incertitude : on attend une loss entre 2,4 et 2,8 nats sur `dev` (perplexité entre 11 et 16), soit un gain d'environ 1,5 à 1,7 nat par rapport au modèle uniforme.

## 3. Dispositif

* Paramètres : matrice de comptage de taille 65 x 65 (4 225 transitions), exploration de l'hyperparamètre de lissage additif alpha ∈ {0.0, 0.1, 1.0}.
* Graines : non applicable (le comptage est entièrement déterministe ; la graine ne sert qu'à l'échantillon de texte généré).
* Device et versions : CPU, Python 3.13.13, numpy 2.5.3.
* Nombre de paramètres : 4 225 coefficients.
* Temps de calcul : 0,08 seconde (instantané).
* État du dépôt : Étape 1, branche `main`.

## 4. Résultats

Toutes les évaluations sont mesurées sur les positions ayant au moins 8 caractères de contexte (`EVAL_CONTEXT = 8`) pour assurer la stricte comparabilité avec les étapes suivantes.

| Modèle / Configuration | Jeu | Loss (nats) | Perplexité |
|---|---|---|---|
| Modèle uniforme (65 caractères) | théorique | 4,1744 | 65,00 |
| Bigramme MLE (alpha = 0,0) | train | 2,4485 | 11,57 |
| Bigramme MLE (alpha = 0,0) | dev | inf | inf |
| Bigramme lissé (alpha = 1,0, Laplace) | train | 2,4515 | 11,61 |
| Bigramme lissé (alpha = 1,0, Laplace) | dev | 2,4959 | 12,13 |
| **Bigramme lissé (alpha = 0,1, retenu)** | **train** | **2,4488** | **11,57** |
| **Bigramme lissé (alpha = 0,1, retenu)** | **dev** | **2,4954** | **12,13** |

Le modèle retenu pour servir de plancher aux étapes suivantes est celui avec **alpha = 0,1** (loss `dev` = 2,4954 nats, perplexité = 12,13).

Échantillon de texte généré (300 caractères, amorce `\n`, seed 0) :
```text
NG ETornetl
UChathan ashe:-wirin se!
Wine, ges o tilo g is LI ar?
As,
Nonthelyoounerdent stt two? tle tr, it sh s
Tht t gethalof finor:
Thuliou, vater h pent t hedve!
Vindy busitroue rl!
Whe me y,
Turais nfu mod Pl hooust V: oocldeano n'sey winorit te,-nd t sprat hee h
So, twndwrerthot, n h lear wi
```

## 5. Échecs et surprises

* **Loss infinie pour alpha = 0 sur `dev`** : le maximum de vraisemblance pur (MLE) attribue une probabilité nulle (donc -log(P) = +inf) à toute paire qui n'a jamais été observée dans `train`. Il suffit d'une seule paire inédite dans les 111 539 caractères de `dev` pour que l'entropie croisée devienne infinie. Le lissage additif (pseudo-comptes) est strictement obligatoire.
* **Sensibilité faible au choix d'alpha positif** : passer d'alpha = 0,1 à alpha = 1,0 ne fait varier la loss `dev` que de 2,4954 à 2,4958, ce qui montre que le corpus de 892 000 caractères d'entraînement est suffisamment dense pour que la plupart des transitions fréquentes dominent largement le pseudo-compte.

## 6. Limites

* Mémoire nulle au-delà d'un caractère : le modèle ne connaît ni le mot en cours, ni la syntaxe, ni le locuteur. Le texte produit n'a aucune cohérence lexicale au-delà de bigrammes fréquents (ex. `th`, `ou`, `in`).
* Plancher statistique, pas un réseau : aucun paramètre n'est appris par descente de gradient.
* Les 8 premiers caractères de `train` et de `dev` ne sont pas évalués (règle commune `EVAL_CONTEXT = 8`).

## 7. Ce qui motive l'étape suivante

Apprendre cette même matrice de transition via un réseau de neurones (un paramètre de poids par bigramme, fonction Softmax et optimisation par descente de gradient). Cela permettra de valider notre chaîne de rétropropagation manuelle en vérifiant que le modèle neuronal converge exactement vers la loss de ce plancher statistique.

## 8. Questions à savoir traiter sans le dépôt

1. Pourquoi un modèle qui donne la même probabilité à chacun des 65 caractères obtient-il une loss de ln(65), et que représente la perplexité correspondante en termes de choix équiprobables ?
2. Pourquoi la loss `dev` du bigramme non lissé (alpha = 0) est-elle infinie alors que sa loss `train` est la meilleure possible ? Que fait le lissage, et pourquoi règle-t-on alpha sur `dev` et non sur `train` ?
3. La loss `train_loss_mle` est-elle une borne inférieure pour tout modèle qui ne voit que le caractère précédent ? Un modèle qui voit 8 caractères peut-il la battre, et pourquoi ?
