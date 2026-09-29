"""Analyse de robustesse : comparaison des méthodes (Spearman) et analyse de sensibilité."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .ranking import RANKING_METHODS


def run_all(X, w, types, crit_labels, alt_labels, methods: list[str], params: dict | None = None) -> dict:
    """Exécute plusieurs méthodes de classement et renvoie {nom: résultat}."""
    params = params or {}
    out = {}
    for name in methods:
        fn = RANKING_METHODS[name]
        out[name] = fn(X, w, types, crit_labels, alt_labels, **params.get(name, {}))
    return out


def ranks_table(results: dict, alt_labels) -> pd.DataFrame:
    return pd.DataFrame({name: r["ranks"] for name, r in results.items()}, index=alt_labels)


def spearman_matrix(ranks: pd.DataFrame) -> pd.DataFrame:
    """Coefficient de corrélation de rang de Spearman entre méthodes :
    ρ = 1 − 6 Σ d_i² / (m (m² − 1))  (formule exacte sans ex æquo ; scipy gère les ex æquo)."""
    cols = list(ranks.columns)
    M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i < j:
                if ranks[a].nunique() < 2 or ranks[b].nunique() < 2:
                    rho = np.nan
                else:
                    rho = spearmanr(ranks[a], ranks[b]).statistic
                M.loc[a, b] = M.loc[b, a] = rho
    return M


def perturb_weights(w, j: int, new_wj: float) -> np.ndarray:
    """Fixe le poids du critère j à new_wj et redistribue 1 − new_wj sur les autres
    proportionnellement à leurs poids initiaux."""
    w = np.asarray(w, dtype=float)
    others = np.delete(np.arange(len(w)), j)
    rest = w[others].sum()
    out = np.empty_like(w)
    out[j] = new_wj
    if rest > 0:
        out[others] = w[others] / rest * (1 - new_wj)
    else:
        out[others] = (1 - new_wj) / len(others)
    return out


def sensitivity(X, w, types, crit_labels, alt_labels, method: str, j: int,
                grid=None, params: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fait varier le poids du critère j sur `grid` ; renvoie (scores, rangs) en fonction du poids."""
    if grid is None:
        grid = np.round(np.linspace(0.0, 1.0, 21), 3)
    fn = RANKING_METHODS[method]
    params = params or {}
    scores, ranks = {}, {}
    for g in grid:
        wg = perturb_weights(w, j, g)
        try:
            r = fn(X, wg, types, crit_labels, alt_labels, **params)
            scores[g] = r["scores"]
            ranks[g] = r["ranks"]
        except Exception:  # noqa: BLE001 — cas dégénérés (poids nuls, etc.)
            scores[g] = [np.nan] * len(alt_labels)
            ranks[g] = [np.nan] * len(alt_labels)
    df_s = pd.DataFrame(scores, index=alt_labels).T
    df_r = pd.DataFrame(ranks, index=alt_labels).T
    df_s.index.name = df_r.index.name = f"Poids de {crit_labels[j]}"
    return df_s, df_r
