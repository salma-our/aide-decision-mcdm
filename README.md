# Aide à la décision multicritère (MCDM)

Application Streamlit générique d'aide à la décision multicritère.
L'utilisateur saisit le problème (alternatives, critères, matrice de décision),
choisit une méthode, et l'application calcule le résultat **en détaillant chaque étape**.

## Installation et lancement

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Méthodes implémentées

| Étape | Méthodes |
|---|---|
| Pondération subjective | Saisie directe, **AHP** (méthode approximative ou vecteur propre exact, λmax, CI, CR), **BWM** (programme linéaire, ξ*), **DEMATEL** |
| Pondération objective | **Entropie**, **CRITIC** |
| Classement (additives) | **WSM**, **WPM**, **WASPAS** (λ), **TOPSIS**, **VIKOR** (S, R, Q, conditions C1/C2), **AHP complet** |
| Robustesse | Comparaison des classements, **corrélation de Spearman**, **analyse de sensibilité** des poids |

L'application démarre sur un problème vierge (3 alternatives × 3 critères) à compléter
manuellement ou par import de fichier.

## Organisation du code

```
app.py              interface Streamlit (4 pages)
mcdm/
  utils.py          saisie (fractions 1/7...), rangs, contrôles
  weighting.py      AHP, BWM, DEMATEL, Entropie, CRITIC
  ranking.py        WSM, WPM, WASPAS, TOPSIS, VIKOR  + registre RANKING_METHODS
  analysis.py       Spearman, sensibilité
  examples.py       emplacement réservé aux jeux de données d'exemple
```

## Ajouter une méthode

1. Écrire la fonction dans `mcdm/ranking.py` (ou `weighting.py`). Elle renvoie
   `{"scores", "ranks", "result", "higher_is_better", "steps"}` (ou `{"weights", "steps"}`).
2. Classement : l'ajouter à `RANKING_METHODS` et sa fiche dans `METHOD_INFO` (app.py).
   Elle apparaît alors automatiquement dans les pages 3 et 4.
3. Pondération : l'ajouter à `WEIGHT_METHODS` et écrire son bloc dans `page_weighting()`.

## Import de données

CSV (séparateur `;`) ou Excel : première colonne = alternatives, première ligne = critères,
lignes facultatives `Type` (Max/Min) et `Poids`. Le bouton « Exporter » produit ce même format.
