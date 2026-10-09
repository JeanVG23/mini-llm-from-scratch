# Étape 2 : bigramme neuronal

Rapport d'expérience du 9 octobre 2026. Code : `steps/s02_bigram_neural.py`.

## 1. Question

Un bigramme appris par descente de gradient retrouve-t-il la loss du bigramme par comptage, et à quelle précision ?

## 2. Prédiction (écrite avant toute mesure)

Sur `train`, la descente de gradient stochastique par mini-lots de 4 096 caractères doit s'approcher très près du plancher statistique MLE calculé à l'étape 1 (`train_loss_mle` = 2,4485 nats), mais sans pouvoir le battre. On attend un écart inférieur à 0,02 nat (autour de 2,45 - 2,46 nats). Sur `dev`, la loss devrait être très proche du bigramme lissé (autour de 2,49 - 2,50 nats), car l'initialisation des poids à zéro (modèle uniforme) et l'arrêt précoce agissent comme une régularisation implicite naturelle.

## 3. Dispositif

* Paramètres : matrice W de taille 65 x 65 (4 225 paramètres), initialisée à zéro (logits nuls = modèle uniforme initial).
* Optimisation : descente de gradient stochastique (SGD), 20 000 pas, taille de lot B = 4 096.
* Exploration des hyperparamètres : comparaison de lr ∈ {20.0, 50.0} sur seed 0, puis rétention de lr = 20.0 répété sur 3 graines (0, 1, 2).
* Device et versions : CPU, Python 3.13.13, numpy 2.5.3.
* Nombre de paramètres : 4 225 poids.
* Temps d'entraînement : environ 57 secondes par exécution.
* État du dépôt : Étape 2, branche `main`.

## 4. Résultats

Toutes les évaluations sont mesurées sur les positions ayant au moins 8 caractères de contexte (`EVAL_CONTEXT = 8`) pour assurer la stricte comparabilité avec les étapes 1 et suivantes.

| Modèle / Configuration | Jeu | Loss (nats) | Perplexité |
|---|---|---|---|
| Modèle uniforme (65 caractères) | théorique | 4,1744 | 65,00 |
| Bigramme MLE par comptage (étape 1) | train | 2,4485 | 11,57 |
| Bigramme MLE par comptage (étape 1) | dev | inf | inf |
| Bigramme lissé par comptage (alpha = 0,1, étape 1) | dev | 2,4954 | 12,13 |
| Bigramme neuronal (lr = 50.0, seed 0, exploration) | dev | 2,4980 | 12,16 |
| **Bigramme neuronal (lr = 20.0, 3 graines)** | **train** | **2,4514 ± 0,0000** | **11,60** |
| **Bigramme neuronal (lr = 20.0, 3 graines)** | **dev** | **2,4970 ± 0,0003** | **12,15** |

Détail des 3 graines retenues (lr = 20.0) :
* Graine 0 : train_loss = 2,4514 | dev_loss = 2,4973 (54,7 s)
* Graine 1 : train_loss = 2,4514 | dev_loss = 2,4966 (58,1 s)
* Graine 2 : train_loss = 2,4515 | dev_loss = 2,4971 (57,6 s)

Écart au plancher statistique d'entraînement :
`train_loss` neuronal (2,4514) - `train_loss_mle` (2,4485) = **+0,0029 nat**.
Le réseau retrouve le plancher théorique à moins de 3 millièmes de nat près.

## 5. Échecs et surprises

* **Pourquoi le réseau ne touche-t-il pas exactement 2,4485 ?**
  1. *Échantillonnage stochastique* : les mini-lots de 4 096 introduisent un bruit de gradient résiduel constant (SGD avec learning rate constant).
  2. *Paires très rares* : certaines transitions n'apparaissent qu'une ou deux fois dans tout le corpus. Pour faire tendre leur probabilité vers 1 ou 0 dans le softmax, les logits correspondants devraient diverger vers l'infini, ce qui nécessiterait un nombre infini de pas avec un gradient qui décroît exponentiellement.
* **Stabilité remarquable entre graines** : l'écart-type sur `train` est quasi nul (0,00004 nat) et sur `dev` de seulement 0,00035 nat, montrant que 20 000 pas de 4 096 caractères (soit environ 81,9 millions de caractères observés, ~90 époques du corpus) garantissent une convergence quasi déterministe.

## 6. Limites

* Modèle linéaire sans contexte : comme à l'étape 1, le réseau ne consulte que le caractère précédent. Il est mathématiquement impossible de descendre en dessous de la matrice de comptage de l'étape 1.
* Coût de calcul : l'étape 1 a trouvé la solution exacte en 0,08 seconde par simple comptage, tandis que l'étape 2 nécessite près d'une minute d'optimisation par SGD pour obtenir une solution approchée à 0,0029 nat près. L'intérêt de l'étape 2 est méthodologique (valider la rétropropagation), pas computationnel.

## 7. Ce qui motive l'étape suivante

Maintenant que la chaîne `Logits -> Softmax -> Entropie Croisée -> Gradient analytique -> SGD` est rigoureusement validée et concorde avec les mathématiques au millième de nat près, nous pouvons introduire ce que le comptage direct ne sait plus faire : un **contexte plus large** (ex. 8 caractères) et une **couche cachée non-linéaire** (MLP à la Bengio 2003) pour casser la barrière des 2,45 nats.

## 8. Questions à savoir traiter sans le dépôt

1. Pourquoi le gradient de l'entropie croisée par rapport aux logits vaut-il `softmax - one-hot` ? Comment s'interprète intuitivement cette différence entre la prédiction et la cible ?
2. Pourquoi écrit-on `np.add.at(dW, x, dlogits)` et non `dW[x] += dlogits` ? Donne un exemple concret avec deux occurrences du même caractère dans un lot.
3. Pourquoi ce modèle ne peut-il mathématiquement jamais battre la loss de l'étape 1 sur le jeu d'entraînement (`train_loss_mle`), même en l'entraînant pendant un million de pas ?
