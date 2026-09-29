"""Méthodes de classement des alternatives (méthodes additives / compensatoires).

WSM, WPM, WASPAS, TOPSIS, VIKOR  (AHP complet : voir weighting.ahp_full)

Chaque fonction renvoie un dict :
    {"result": DataFrame (score + rang, trié), "scores": np.ndarray, "ranks": np.ndarray,
     "higher_is_better": bool, "steps": [(titre, objet), ...]}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import check_decision_matrix, is_benefit, rank_asc, rank_desc


def _result_table(alt_labels, cols: dict, rank_col: str) -> pd.DataFrame:
    df = pd.DataFrame(cols, index=alt_labels)
    return df.sort_values(rank_col)


# ---------------------------------------------------------------------------
# Normalisation linéaire (méthodes de pondération simple)
# ---------------------------------------------------------------------------
def linear_normalize(X, types, mode: str = "max"):
    """mode = "max"   : r = x/max (+) ;  r = min/x (−)   [normalisation linéaire max]
    Retourne (R, x_plus, x_minus)."""
    X = np.asarray(X, dtype=float)
    ben = is_benefit(types)
    x_plus = np.where(ben, X.max(axis=0), X.min(axis=0))   # solution idéale positive A+
    x_minus = np.where(ben, X.min(axis=0), X.max(axis=0))  # solution idéale négative A−
    R = np.empty_like(X)
    for j in range(X.shape[1]):
        if ben[j]:
            R[:, j] = X[:, j] / x_plus[j]
        else:
            R[:, j] = x_plus[j] / X[:, j]
    return R, x_plus, x_minus


def _simple_common(X, w, types, crit_labels, alt_labels, method):
    X = np.asarray(X, dtype=float)
    check_decision_matrix(X, need_positive=True, method=method)
    R, xp, xm = linear_normalize(X, types)
    steps = [
        ("1) Solutions idéales positive A⁺ et négative A⁻",
         pd.DataFrame([xp, xm], index=["A⁺ (meilleure valeur)", "A⁻ (pire valeur)"], columns=crit_labels)),
        ("2) Matrice normalisée : r = x / x⁺ (critère +) ; r = x⁺ / x (critère −)",
         pd.DataFrame(R, index=alt_labels, columns=crit_labels)),
    ]
    return R, steps


def wsm(X, w, types, crit_labels, alt_labels) -> dict:
    """Weighted Sum Method : Q¹_i = Σ w_j r_ij"""
    w = np.asarray(w, dtype=float)
    R, steps = _simple_common(X, w, types, crit_labels, alt_labels, "WSM")
    V = R * w
    Q = V.sum(axis=1)
    steps.append(("3) Matrice pondérée w_j × r_ij", pd.DataFrame(V, index=alt_labels, columns=crit_labels)))
    ranks = rank_desc(Q)
    res = _result_table(alt_labels, {"Q¹ (WSM)": Q, "Rang": ranks}, "Rang")
    steps.append(("4) Score global Q¹_i = Σ w_j r_ij (le plus élevé est le meilleur)", res))
    return {"result": res, "scores": Q, "ranks": ranks, "higher_is_better": True, "steps": steps}


def wpm(X, w, types, crit_labels, alt_labels) -> dict:
    """Weighted Product Method : Q²_i = Π r_ij^w_j"""
    w = np.asarray(w, dtype=float)
    R, steps = _simple_common(X, w, types, crit_labels, alt_labels, "WPM")
    P = R ** w
    Q = P.prod(axis=1)
    steps.append(("3) Termes r_ij ^ w_j", pd.DataFrame(P, index=alt_labels, columns=crit_labels)))
    ranks = rank_desc(Q)
    res = _result_table(alt_labels, {"Q² (WPM)": Q, "Rang": ranks}, "Rang")
    steps.append(("4) Score global Q²_i = Π r_ij ^ w_j (le plus élevé est le meilleur)", res))
    return {"result": res, "scores": Q, "ranks": ranks, "higher_is_better": True, "steps": steps}


def waspas(X, w, types, crit_labels, alt_labels, lam: float = 0.5) -> dict:
    """WASPAS : Q_i = λ Q¹_i + (1 − λ) Q²_i"""
    w = np.asarray(w, dtype=float)
    R, steps = _simple_common(X, w, types, crit_labels, alt_labels, "WASPAS")
    Q1 = (R * w).sum(axis=1)
    Q2 = (R ** w).prod(axis=1)
    Q = lam * Q1 + (1 - lam) * Q2
    steps.append(("3) Scores WSM (Q¹) et WPM (Q²)",
                  pd.DataFrame({"Q¹ = Σ w_j r_ij": Q1, "Q² = Π r_ij^w_j": Q2}, index=alt_labels)))
    ranks = rank_desc(Q)
    res = _result_table(alt_labels, {"Q¹ (WSM)": Q1, "Q² (WPM)": Q2,
                                     f"Q = {lam:g}·Q¹ + {1-lam:g}·Q²": Q, "Rang": ranks}, "Rang")
    steps.append((f"4) Critère généralisé conjoint (λ = {lam:g}) — le plus élevé est le meilleur", res))
    return {"result": res, "scores": Q, "ranks": ranks, "higher_is_better": True, "steps": steps}


# ---------------------------------------------------------------------------
# TOPSIS
# ---------------------------------------------------------------------------
def topsis(X, w, types, crit_labels, alt_labels) -> dict:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    check_decision_matrix(X)
    ben = is_benefit(types)
    norms = np.sqrt((X ** 2).sum(axis=0))
    if np.any(norms == 0):
        raise ValueError("TOPSIS : une colonne est entièrement nulle.")
    R = X / norms
    steps = [("1) Matrice normalisée r_ij = x_ij / √(Σ_i x_ij²)", pd.DataFrame(R, index=alt_labels, columns=crit_labels))]
    V = R * w
    steps.append(("2) Matrice normalisée pondérée v_ij = w_j × r_ij", pd.DataFrame(V, index=alt_labels, columns=crit_labels)))
    Ip = np.where(ben, V.max(axis=0), V.min(axis=0))
    Im = np.where(ben, V.min(axis=0), V.max(axis=0))
    steps.append(("3) Solutions idéales positive I⁺ et négative I⁻",
                  pd.DataFrame([Ip, Im], index=["I⁺", "I⁻"], columns=crit_labels)))
    Sp = np.sqrt(((V - Ip) ** 2).sum(axis=1))
    Sm = np.sqrt(((V - Im) ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        RC = np.where(Sp + Sm > 0, Sm / (Sp + Sm), 0.0)
    steps.append(("4) Mesures de séparation", pd.DataFrame({"S⁺ (distance à I⁺)": Sp, "S⁻ (distance à I⁻)": Sm}, index=alt_labels)))
    ranks = rank_desc(RC)
    res = _result_table(alt_labels, {"S⁺": Sp, "S⁻": Sm, "RC* = S⁻/(S⁺+S⁻)": RC, "Rang": ranks}, "Rang")
    steps.append(("5) Coefficient de proximité RC* (le plus élevé est le meilleur)", res))
    return {"result": res, "scores": RC, "ranks": ranks, "higher_is_better": True, "steps": steps}


# ---------------------------------------------------------------------------
# VIKOR
# ---------------------------------------------------------------------------
def vikor(X, w, types, crit_labels, alt_labels, v: float = 0.5, strict: bool = True) -> dict:
    X = np.asarray(X, dtype=float)
    w = np.asarray(w, dtype=float)
    check_decision_matrix(X)
    m, n = X.shape
    if m < 2:
        raise ValueError("VIKOR nécessite au moins 2 alternatives.")
    ben = is_benefit(types)
    xb = np.where(ben, X.max(axis=0), X.min(axis=0))   # x_j*
    xw = np.where(ben, X.min(axis=0), X.max(axis=0))   # x_j−
    steps = [("1) Meilleures (x*) et pires (x⁻) valeurs de chaque critère",
              pd.DataFrame([xb, xw], index=["x* (meilleure)", "x⁻ (pire)"], columns=crit_labels))]
    rng = xb - xw
    with np.errstate(invalid="ignore", divide="ignore"):
        D = np.where(rng != 0, (xb - X) / rng, 0.0)
    WD = w * D
    steps.append(("2) Termes w_j (x*_j − x_ij)/(x*_j − x⁻_j)", pd.DataFrame(WD, index=alt_labels, columns=crit_labels)))
    S = WD.sum(axis=1)
    R = WD.max(axis=1)
    S_star, S_minus, R_star, R_minus = S.min(), S.max(), R.min(), R.max()
    qs = (S - S_star) / (S_minus - S_star) if S_minus > S_star else np.zeros(m)
    qr = (R - R_star) / (R_minus - R_star) if R_minus > R_star else np.zeros(m)
    Q = v * qs + (1 - v) * qr
    steps.append((f"3) S (utilité de groupe), R (regret individuel), Q (v = {v:g})",
                  pd.DataFrame({"S_i = Σ": S, "R_i = max": R, "Q_i": Q}, index=alt_labels)))
    steps.append(("   Valeurs extrêmes",
                  pd.DataFrame({"S* = min S": [S_star], "S⁻ = max S": [S_minus],
                                "R* = min R": [R_star], "R⁻ = max R": [R_minus]})))
    rS, rR, rQ = rank_asc(S), rank_asc(R), rank_asc(Q)
    res = _result_table(alt_labels, {"S": S, "R": R, "Q": Q, "Rang S": rS, "Rang R": rR, "Rang Q": rQ}, "Rang Q")
    steps.append(("4) Trois classements (plus la valeur est petite, meilleure est l'alternative)", res))

    # 5) solution de compromis
    order = np.argsort(Q, kind="stable")
    a1, a2 = order[0], order[1]
    DQ = 1.0 / (m - 1)
    c1 = (Q[a2] - Q[a1]) >= DQ - 1e-12
    # Condition de stabilité (version stricte) : A' doit être aussi le meilleur selon S ET R
    # (Opricovic accepte « S et/ou R » : option strict=False)
    c2 = (S[a1] == S.min()) and (R[a1] == R.min()) if strict else (S[a1] == S.min()) or (R[a1] == R.min())
    lines = [
        f"A' = {alt_labels[a1]} (meilleur Q = {Q[a1]:.4f}),  A'' = {alt_labels[a2]} (Q = {Q[a2]:.4f})",
        f"DQ = 1/(m − 1) = {DQ:.4f}",
        f"C1 — Avantage acceptable : Q(A'') − Q(A') = {Q[a2] - Q[a1]:.4f} {'≥' if c1 else '<'} {DQ:.4f} → "
        f"{'satisfaite' if c1 else 'non satisfaite'}",
        f"C2 — Stabilité acceptable : A' est aussi le meilleur selon S ({'oui' if S[a1] == S.min() else 'non'}) "
        f"{'et' if strict else 'et/ou'} selon R ({'oui' if R[a1] == R.min() else 'non'}) → {'satisfaite' if c2 else 'non satisfaite'}",
    ]
    if c1 and c2:
        compromise = [alt_labels[a1]]
        lines.append(f"Solution de compromis : **{alt_labels[a1]}**")
    elif c1 and not c2:
        compromise = [alt_labels[a1], alt_labels[a2]]
        lines.append(f"Seule C2 n'est pas satisfaite : solutions de compromis **{alt_labels[a1]}** et **{alt_labels[a2]}**")
    else:
        compromise = [alt_labels[k] for k in order if Q[k] - Q[a1] < DQ - 1e-12]
        lines.append("C1 n'est pas satisfaite : ensemble de compromis {A', …, A^k} avec Q(A^k) − Q(A') < DQ : **"
                     + ", ".join(compromise) + "**")
    steps.append(("5) Solution de compromis", "\n\n".join(lines)))
    return {"result": res, "scores": Q, "ranks": rQ, "higher_is_better": False, "steps": steps,
            "S": S, "R": R, "Q": Q, "compromise": compromise, "C1": bool(c1), "C2": bool(c2)}


RANKING_METHODS = {
    "WSM": wsm,
    "WPM": wpm,
    "WASPAS": waspas,
    "TOPSIS": topsis,
    "VIKOR": vikor,
}
