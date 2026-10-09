# Pipeline d'Entraînement : du Texte à la Descente de Gradient

Ce document détaille l'intégralité du pipeline d'apprentissage pour un modèle neuronal (à commencer par le bigramme de l'Étape 2, puis extensible aux étapes suivantes).

Il montre le cycle complet : échantillonnage stochastique, passe avant (*forward*), calcul de la perte, rétropropagation analytique (*backward*), et mise à jour des paramètres.

---

## 1. Vue d'Ensemble du Pipeline (Mermaid)

```mermaid
flowchart TD
    subgraph Data["1. Échantillonnage Stochastique (Mini-Batch)"]
        Corpus["Corpus d'entraînement (892 000 caractères)"]
        RNG["Tirage aléatoire uniforme"] --> BatchIndices["Indices de B paires au hasard"]
        Corpus --> BatchIndices
        BatchIndices --> X["Entrée x : caractères précédents<br/>Forme : (B,) ou one-hot (B, V)"]
        BatchIndices --> Y["Cible y : vrais caractères suivants<br/>Forme : (B,)"]
    end

    subgraph Forward["2. Passe Avant (Forward)"]
        W["Matrice de Poids W<br/>Forme : (V, V)"]
        X --> Matmul["Projection linéaire : z = x @ W<br/>(ou sélection de ligne W[x])"]
        W --> Matmul
        Matmul --> Logits["Logits z (scores bruts)<br/>Forme : (B, V)"]
        
        Logits --> Shift["Stabilité numérique : z_shift = z - max(z)"]
        Shift --> Exp["Exponentielle : exp_z = exp(z_shift)"]
        Exp --> SumExp["Somme par ligne : sum_exp = sum(exp_z)"]
        Exp --> Softmax["Normalisation Softmax : p = exp_z / sum_exp"]
        SumExp --> Softmax
        Softmax --> Probs["Probabilités prédites p<br/>Forme : (B, V)"]
    end

    subgraph LossCalculation["3. Calcul de la Perte (Entropie Croisée)"]
        Probs --> TargetProb["Sélection probabilité de la cible : p[i, y[i]]<br/>Forme : (B,)"]
        Y --> TargetProb
        TargetProb --> NLL["Log-vraisemblance négative : -log(p_cible)<br/>Forme : (B,)"]
        NLL --> MeanLoss["Moyenne sur le lot : Loss = mean(-log(p_cible))<br/>Scalaire unique ()"]
    end

    subgraph Backward["4. Passe Arrière (Rétropropagation Analytique)"]
        MeanLoss -.-> StartBack["Départ de l'erreur : dLoss / dLoss = 1"]
        StartBack --> GradLogits["Dérivée Loss par rapport aux Logits z :<br/>dz = (p - one_hot(y)) / B<br/>Forme : (B, V)"]
        GradLogits --> ChainRule["Règle de chaîne matricielle :<br/>dW = x.T @ dz<br/>(ou accumulation dans dW[x])"]
        X --> ChainRule
        ChainRule --> GradW["Matrice des gradients dW<br/>Forme : (V, V)"]
    end

    subgraph Optimizer["5. Mise à Jour (Descente de Gradient / SGD)"]
        GradW --> Step["Pas d'optimisation :<br/>W = W - lr * dW"]
        LR["Taux d'apprentissage lr"] --> Step
        W -.-> Step
        Step --> UpdatedW["Poids W mis à jour pour le pas suivant"]
        UpdatedW --> W
    end

    subgraph Validation["6. Contrôles et Filets de Sécurité"]
        GradW --> GradCheck["Contrôle unitaire (tests/gradcheck.py) :<br/>dW_num = (L(W + eps) - L(W - eps)) / (2 * eps)<br/>Différence relative < 1e-5"]
        MeanLoss --> MLEFloor["Contrôle global (Étape 1) :<br/>Convergence vers le plancher MLE (2,4485 nats sur train)"]
    end

    UpdatedW --> Data
```

---

## 2. Détail des Étapes et Formes des Tenseurs

### 1. Échantillonnage stochastique
* **Source** : Le texte d'entraînement complet tokenisé sous forme d'entiers (`train_ids`).
* **Mécanisme** : Pour chaque pas, on tire un sous-ensemble aléatoire de taille `B` (ex. `B = 32`).
* **Tenseurs générés** :
  * `x` : indices des caractères précédents `# (B,)`
  * `y` : indices des caractères cibles attendus `# (B,)`
* **Rôle** : Injecter le bruit d'échantillonnage qui permet à l'optimiseur d'échapper aux mauvais minima locaux et de converger vers des minima plats.

### 2. Passe avant (*Forward*)
* **Projection linéaire** :
  * En notation vectorielle complète avec encodage one-hot : `z = x_one_hot @ W` `# (B, V)`.
  * En pratique (sans gaspiller de mémoire) : sélection directe de lignes `z = W[x]` `# (B, V)`.
* **Softmax à stabilité numérique** :
  1. `z_shift = z - np.max(z, axis=1, keepdims=True)` `# (B, V)` pour éviter l'overflow exponentiel.
  2. `exp_z = np.exp(z_shift)` `# (B, V)`.
  3. `p = exp_z / np.sum(exp_z, axis=1, keepdims=True)` `# (B, V)`.

### 3. Calcul de la perte (*Loss*)
* **Entropie croisée** :
  1. On extrait pour chaque élément du lot la probabilité qu'il a accordée à la vraie réponse : `p_cible = p[np.arange(B), y]` `# (B,)`.
  2. On calcule la surprise individuelle : `-np.log(p_cible)` `# (B,)`.
  3. On calcule la note moyenne sur le lot : `loss = np.mean(-np.log(p_cible))` (scalaire).

### 4. Passe arrière (*Backward* / Rétropropagation)
* **Pas de simulation numérique en entraînement** : calcul analytique exact et instantané.
* **Gradient par rapport aux logits `z`** :
  * `dz = p.copy()` `# (B, V)`
  * `dz[np.arange(B), y] -= 1.0` (on soustrait 1 à la vraie cible)
  * `dz /= B` (on normalise par la taille du lot car la perte est une moyenne)
* **Gradient par rapport aux poids `W`** :
  * `dW = x_one_hot.T @ dz` `# (V, V)`
  * Ou par accumulation directe : `np.add.at(dW, x, dz)` `# (V, V)`.

### 5. Mise à jour des poids (*Optimizer Step*)
* Formule : `W -= lr * dW`
* Tous les 4 225 coefficients de `W` sont ajustés simultanément dans la direction qui réduit la perte moyenne.

### 6. Filets de validation indispensables
1. **Contrôle unitaire local** ([tests/gradcheck.py](tests/gradcheck.py)) : vérifie que le gradient analytique `dW` concorde avec la dérivée par différences finies numériques à `1e-5` près.
2. **Contrôle global de convergence** : le bigramme neuronal doit converger vers le plancher statistique calculé à l'Étape 1 dans [experiments/s01-bigram-counts.md](experiments/s01-bigram-counts.md) (`train_loss ≈ 2,4485 nats`).
