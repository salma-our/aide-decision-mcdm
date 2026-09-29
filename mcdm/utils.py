"""Fonctions utilitaires communes à toutes les méthodes MCDM."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

BENEFIT = "Max (+)"   # critère positif : à maximiser
COST = "Min (−)"      # critère négatif : à minimiser


def parse_number(value) -> float:
    """Convertit une saisie en nombre : accepte 3, '3', '0,5', '1/7', '1 / 3'."""
    if value is None:
        raise ValueError("valeur vide")
    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            raise ValueError("valeur vide")
        return float(value)
    s = str(value).strip().replace(" ", "").replace(",", ".")
    if s == "":
        raise ValueError("valeur vide")
    if "/" in s:
        return float(Fraction(s))
    return float(s)


def to_float_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit un DataFrame saisi (texte/nombres) en DataFrame de floats.

    Lève ValueError avec la cellule fautive si une valeur est invalide.
    """
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for i in df.index:
        for j in df.columns:
            try:
                out.loc[i, j] = parse_number(df.loc[i, j])
            except (ValueError, ZeroDivisionError):
                raise ValueError(f"Valeur invalide en ligne « {i} », colonne « {j} » : {df.loc[i, j]!r}")
    return out


def is_benefit(types: list[str]) -> np.ndarray:
    return np.array([t == BENEFIT for t in types], dtype=bool)


def rank_desc(scores: np.ndarray) -> np.ndarray:
    """Rang 1 = score le plus élevé (ex æquo -> même rang minimal)."""
    return pd.Series(scores).rank(ascending=False, method="min").astype(int).to_numpy()


def rank_asc(scores: np.ndarray) -> np.ndarray:
    """Rang 1 = score le plus faible (VIKOR)."""
    return pd.Series(scores).rank(ascending=True, method="min").astype(int).to_numpy()


def normalize_weights(w) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("La somme des poids doit être strictement positive.")
    return w / s


def check_decision_matrix(X: np.ndarray, need_positive: bool = False, method: str = "") -> None:
    if not np.all(np.isfinite(X)):
        raise ValueError("La matrice de décision contient des valeurs manquantes ou invalides.")
    if need_positive and np.any(X <= 0):
        raise ValueError(f"{method} : toutes les valeurs de la matrice doivent être strictement positives.")
