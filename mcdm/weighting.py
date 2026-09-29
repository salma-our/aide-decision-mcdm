"""Méthodes de pondération des critères.

Subjectives : AHP, BWM, DEMATEL
Objectives  : Entropie, CRITIC

Chaque fonction renvoie un dict :
    {"weights": np.ndarray, "steps": [(titre, objet), ...], ...infos spécifiques}
où « objet » est un DataFrame, un texte (str) ou un nombre, affiché tel quel par l'interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .utils import BENEFIT, check_decision_matrix, is_benefit

# ---------------------------------------------------------------------------
# AHP
# ---------------------------------------------------------------------------
# Indice aléatoire RI (valeurs usuelles pour n = 3..10 ; Saaty pour n > 10)
RI_TABLE = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
            9: 1.45, 10: 1.56, 11: 1.57, 12: 1.58, 13: 1.59, 14: 1.60, 15: 1.61}

SAATY_SCALE = pd.DataFrame({
    "Valeur": ["1", "3", "5", "7", "9", "2, 4, 6, 8", "1/x"],
    "Signification": [
        "Importance égale",
        "Importance modérée",
        "Importance forte",
        "Importance très forte",
        "Importance extrême",
        "Valeurs intermédiaires",
        "Réciproque : si Ci vaut x par rapport à Cj, alors Cj vaut 1/x par rapport à Ci",
    ],
})


def ahp_check_reciprocal(A: np.ndarray, tol: float = 0.02) -> list[str]:
    """Liste des avertissements si la matrice n'est pas réciproque (a_ji = 1/a_ij)."""
    warnings = []
    n = A.shape[0]
    for i in range(n):
        if abs(A[i, i] - 1) > 1e-9:
            warnings.append(f"La diagonale doit valoir 1 (case {i+1},{i+1} = {A[i, i]:g}).")
        for j in range(i + 1, n):
            if A[i, j] <= 0 or A[j, i] <= 0:
                warnings.append(f"Les valeurs doivent être > 0 (cases {i+1},{j+1}).")
            elif abs(A[i, j] * A[j, i] - 1) > tol:
                warnings.append(
                    f"Non réciproque : a[{i+1},{j+1}] = {A[i, j]:.4g} et a[{j+1},{i+1}] = {A[j, i]:.4g} "
                    f"(produit = {A[i, j] * A[j, i]:.3f} au lieu de 1)."
                )
    return warnings


def ahp_weights(A, labels: list[str], method: str = "approx") -> dict:
    """Poids AHP + test de cohérence.

    method = "approx"  : méthode approximative (normalisation par colonnes + moyenne des lignes)
    method = "eigen"   : vecteur propre exact associé à λmax
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("La matrice de comparaison doit être carrée.")
    if np.any(A <= 0):
        raise ValueError("Toutes les valeurs de la matrice de comparaison doivent être > 0.")

    steps = []
    col_sums = A.sum(axis=0)
    df_A = pd.DataFrame(A, index=labels, columns=labels)
    df_A_sum = pd.concat([df_A, pd.DataFrame([col_sums], index=["Somme (s_j)"], columns=labels)])
    steps.append(("1) Matrice de comparaison par paires et somme des colonnes", df_A_sum))

    norm = A / col_sums
    if method == "eigen":
        vals, vecs = np.linalg.eig(A)
        k = int(np.argmax(vals.real))
        w = np.abs(vecs[:, k].real)
        w = w / w.sum()
        lam = float(vals[k].real)
        df_norm = pd.DataFrame(norm, index=labels, columns=labels)
        steps.append(("2) Matrice normalisée (pour information)", df_norm))
        steps.append(("3) Vecteur propre exact associé à λmax (normalisé)",
                      pd.DataFrame({"Poids w": w}, index=labels)))
    else:
        w = norm.mean(axis=1)
        lam = float(np.dot(col_sums, w))
        df_norm = pd.DataFrame(norm, index=labels, columns=labels)
        df_norm["Priorité w (moyenne ligne)"] = w
        steps.append(("2) Matrice normalisée (a_ij / s_j) et moyenne de chaque ligne", df_norm))

    CI = (lam - n) / (n - 1) if n > 1 else 0.0
    RI = RI_TABLE.get(n, 1.59)
    CR = CI / RI if RI > 0 else 0.0
    consistent = CR < 0.1

    # vérification AW ≈ λmax W
    Aw = A @ w
    df_check = pd.DataFrame({"w": w, "A·w": Aw, "(A·w)/w": Aw / w}, index=labels)
    steps.append(("Vérification : A·W ≈ λmax·W", df_check))

    formula = "λmax = Σ s_j × w_j" if method != "eigen" else "λmax = plus grande valeur propre de A"
    cons = pd.DataFrame({
        "Indicateur": ["n", "λmax", "CI = (λmax − n)/(n − 1)", "RI", "CR = CI / RI", "Cohérente (CR < 0,1) ?"],
        "Valeur": [n, round(lam, 4), round(CI, 4), RI, round(CR, 4), "Oui" if consistent else "Non"],
    })
    steps.append((f"Test de cohérence ({formula})", cons))

    return {"weights": w, "lambda_max": lam, "CI": CI, "RI": RI, "CR": CR,
            "consistent": consistent, "steps": steps}


def ahp_full(criteria_matrix, alt_matrices: list, crit_labels: list[str], alt_labels: list[str],
             method: str = "approx") -> dict:
    """AHP complet : poids des critères + priorités locales + priorités globales."""
    crit = ahp_weights(criteria_matrix, crit_labels, method)
    local = np.zeros((len(alt_labels), len(crit_labels)))
    locals_detail = []
    for j, M in enumerate(alt_matrices):
        r = ahp_weights(M, alt_labels, method)
        local[:, j] = r["weights"]
        locals_detail.append(r)
    global_scores = local @ crit["weights"]
    df_local = pd.DataFrame(local, index=alt_labels, columns=crit_labels)
    df_final = df_local.copy()
    df_final.loc["Poids des critères"] = crit["weights"]
    df_final["Priorité globale Σ w_j·w_ij"] = list(global_scores) + [np.nan]
    return {"criteria": crit, "locals": locals_detail, "local_matrix": df_local,
            "scores": global_scores, "final_table": df_final}


# ---------------------------------------------------------------------------
# BWM (Best-Worst Method, version linéaire)
# ---------------------------------------------------------------------------
def bwm_weights(best: int, worst: int, a_BO, a_OW, labels: list[str]) -> dict:
    """BWM linéaire (Rezaei) :
        min ξ  s.c. |w_B − a_Bj w_j| ≤ ξ ,  |w_j − a_jW w_W| ≤ ξ ,  Σ w_j = 1,  w_j ≥ 0
    résolu par programmation linéaire (scipy.optimize.linprog, forme A_ub x ≤ b_ub).
    """
    a_BO = np.asarray(a_BO, dtype=float)
    a_OW = np.asarray(a_OW, dtype=float)
    n = len(labels)
    warnings = []
    if best == worst:
        raise ValueError("Le meilleur et le pire critère doivent être différents.")
    if abs(a_BO[best] - 1) > 1e-9:
        warnings.append("a_BB devrait valoir 1.")
    if abs(a_OW[worst] - 1) > 1e-9:
        warnings.append("a_WW devrait valoir 1.")
    if abs(a_BO[worst] - a_BO.max()) > 1e-9:
        warnings.append("La note la plus élevée du vecteur BO devrait porter sur le critère Worst.")
    if abs(a_OW[best] - a_OW.max()) > 1e-9:
        warnings.append("La note la plus élevée du vecteur OW devrait porter sur le critère Best.")

    # variables x = [w_1, ..., w_n, ξ]
    rows, b, names = [], [], []
    for j in range(n):
        # écart au meilleur :  w_B − a_Bj w_j − ξ ≤ 0   et   −w_B + a_Bj w_j − ξ ≤ 0
        r1 = np.zeros(n + 1); r1[best] += 1; r1[j] -= a_BO[j]; r1[n] = -1
        r2 = np.zeros(n + 1); r2[best] -= 1; r2[j] += a_BO[j]; r2[n] = -1
        # écart au pire :  w_j − a_jW w_W − ξ ≤ 0   et   −w_j + a_jW w_W − ξ ≤ 0
        r3 = np.zeros(n + 1); r3[j] += 1; r3[worst] -= a_OW[j]; r3[n] = -1
        r4 = np.zeros(n + 1); r4[j] -= 1; r4[worst] += a_OW[j]; r4[n] = -1
        for r, nm in [(r1, f"wB − aB{j+1}·w{j+1} − ξ ≤ 0"), (r2, f"−wB + aB{j+1}·w{j+1} − ξ ≤ 0"),
                      (r3, f"w{j+1} − a{j+1}W·wW − ξ ≤ 0"), (r4, f"−w{j+1} + a{j+1}W·wW − ξ ≤ 0")]:
            if np.any(np.abs(r[:n]) > 1e-12):   # on supprime les contraintes triviales (0 − ξ ≤ 0)
                rows.append(r); b.append(0.0); names.append(nm)
    A_ub = np.array(rows)
    b_ub = np.array(b)
    A_eq = np.array([[1.0] * n + [0.0]])
    b_eq = np.array([1.0])
    c = np.zeros(n + 1); c[n] = 1.0
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise ValueError(f"Le programme linéaire n'a pas pu être résolu : {res.message}")
    w = res.x[:n]
    xi = float(res.x[n])

    steps = []
    steps.append(("1) Vecteurs Best-to-Others (BO) et Others-to-Worst (OW)",
                  pd.DataFrame({f"BO : {labels[best]} → autres": a_BO,
                                f"OW : autres → {labels[worst]}": a_OW}, index=labels).T))
    df_Aub = pd.DataFrame(A_ub, columns=[f"w{j+1}" for j in range(n)] + ["ξ"], index=names)
    df_Aub["≤ b"] = b_ub
    steps.append(("2) Programme linéaire : min ξ — contraintes sous forme A·x ≤ b "
                  "(+ Σ wj = 1, wj ≥ 0, ξ ≥ 0)", df_Aub))
    steps.append(("3) Poids optimaux", pd.DataFrame({"Poids w": w}, index=labels)))
    ratios = []
    for j in range(n):
        ratios.append({"Critère": labels[j],
                       "a_Bj": a_BO[j], "w_B / w_j": w[best] / w[j] if w[j] > 0 else np.inf,
                       "a_jW": a_OW[j], "w_j / w_W": w[j] / w[worst] if w[worst] > 0 else np.inf})
    steps.append(("4) Contrôle : rapports obtenus vs jugements", pd.DataFrame(ratios).set_index("Critère")))
    steps.append(("Indicateur de cohérence ξ*",
                  f"ξ* = {xi:.4f} — plus ξ* est proche de 0, plus les jugements de l'expert sont cohérents."))
    return {"weights": w, "xi": xi, "warnings": warnings, "steps": steps,
            "n_comparisons": 2 * n - 3}


# ---------------------------------------------------------------------------
# DEMATEL
# ---------------------------------------------------------------------------
def dematel_weights(Z, labels: list[str], threshold: float | None = None) -> dict:
    """DEMATEL classique.
    Z : matrice d'influence directe (échelle 0 = aucune … 4 = très forte), diagonale nulle.
    X = Z / s  avec s = max(max somme lignes, max somme colonnes)
    T = X (I − X)^-1   (matrice de relation totale)
    D = somme des lignes (influence donnée), R = somme des colonnes (influence reçue)
    Proéminence D+R, relation D−R (>0 : cause, <0 : effet)
    Poids w_j = sqrt((D+R)² + (D−R)²) normalisés.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    steps = []
    steps.append(("1) Matrice d'influence directe Z", pd.DataFrame(Z, index=labels, columns=labels)))
    s = max(Z.sum(axis=1).max(), Z.sum(axis=0).max())
    if s <= 0:
        raise ValueError("Saisissez les influences entre critères : la matrice est encore nulle.")
    X = Z / s
    steps.append((f"2) Matrice normalisée X = Z / s  (s = {s:g})", pd.DataFrame(X, index=labels, columns=labels)))
    I = np.eye(n)
    T = X @ np.linalg.inv(I - X)
    steps.append(("3) Matrice de relation totale T = X (I − X)⁻¹", pd.DataFrame(T, index=labels, columns=labels)))
    D = T.sum(axis=1)
    R = T.sum(axis=0)
    w = np.sqrt((D + R) ** 2 + (D - R) ** 2)
    w = w / w.sum()
    df = pd.DataFrame({"D (donne)": D, "R (reçoit)": R, "D + R (proéminence)": D + R,
                       "D − R (relation)": D - R,
                       "Groupe": ["Cause" if v > 0 else "Effet" for v in D - R],
                       "Poids w": w}, index=labels)
    steps.append(("4) Indicateurs D, R, D+R, D−R et poids", df))
    if threshold is None:
        threshold = float(T.mean())
    links = [(labels[i], labels[j], T[i, j]) for i in range(n) for j in range(n) if i != j and T[i, j] > threshold]
    return {"weights": w, "T": T, "D": D, "R": R, "threshold": threshold, "links": links,
            "table": df, "steps": steps}


# ---------------------------------------------------------------------------
# Entropie
# ---------------------------------------------------------------------------
def entropy_weights(X, crit_labels: list[str], alt_labels: list[str], types: list[str],
                    pre_normalize: str = "none") -> dict:
    """Poids par l'entropie de Shannon.
    pre_normalize = "none"   : p_ij = x_ij / Σ x_ij  (données brutes, positives)
                  = "linear" : d_ij = x_ij/max (critère +) ou min/x_ij (critère −), puis p_ij
    E_j = −(1/ln m) Σ p_ij ln p_ij ;  w_j = (1 − E_j) / (n − Σ E_k)
    """
    X = np.asarray(X, dtype=float)
    check_decision_matrix(X)
    m, n = X.shape
    if m < 2:
        raise ValueError("L'entropie nécessite au moins 2 alternatives.")
    if np.any(X < 0):
        raise ValueError("Entropie : les valeurs doivent être positives ou nulles.")
    steps = []
    if pre_normalize == "linear":
        ben = is_benefit(types)
        D = np.empty_like(X)
        for j in range(n):
            if ben[j]:
                D[:, j] = X[:, j] / X[:, j].max()
            else:
                if np.any(X[:, j] == 0):
                    raise ValueError(f"Critère {crit_labels[j]} : valeur nulle, normalisation min/x impossible.")
                D[:, j] = X[:, j].min() / X[:, j]
        steps.append(("0) Matrice normalisée D (x/max pour +, min/x pour −)",
                      pd.DataFrame(D, index=alt_labels, columns=crit_labels)))
    else:
        D = X
    col = D.sum(axis=0)
    if np.any(col == 0):
        raise ValueError("Une colonne est entièrement nulle : entropie non définie.")
    P = D / col
    steps.append(("1) Matrice des proportions p_ij = d_ij / Σ_i d_ij", pd.DataFrame(P, index=alt_labels, columns=crit_labels)))
    k = 1.0 / np.log(m)
    with np.errstate(divide="ignore", invalid="ignore"):
        PlnP = np.where(P > 0, P * np.log(P), 0.0)
    steps.append(("2) Termes p_ij × ln(p_ij)  (0 × ln 0 = 0)", pd.DataFrame(PlnP, index=alt_labels, columns=crit_labels)))
    E = -k * PlnP.sum(axis=0)
    div = 1 - E
    denom = n - E.sum()
    if denom <= 1e-15:
        raise ValueError("Tous les critères ont une entropie de 1 (aucune dispersion) : poids indéfinis.")
    w = div / denom
    df = pd.DataFrame({"Entropie E_j": E, "Diversification 1 − E_j": div, "Poids w_j": w}, index=crit_labels)
    steps.append((f"3) Entropie, diversification et poids  (k = 1/ln m = {k:.4f})", df))
    return {"weights": w, "E": E, "steps": steps}


# ---------------------------------------------------------------------------
# CRITIC
# ---------------------------------------------------------------------------
def critic_weights(X, crit_labels: list[str], alt_labels: list[str], types: list[str]) -> dict:
    """CRITIC (Diakoulaki et al.)
    r_ij min-max selon le sens ; ρ_jk corrélation de Pearson ;
    σ_j écart-type (m − 1) ; C_j = σ_j Σ_k (1 − |ρ_jk|) ; w_j = C_j / Σ C.
    """
    X = np.asarray(X, dtype=float)
    check_decision_matrix(X)
    m, n = X.shape
    if m < 3:
        raise ValueError("CRITIC nécessite au moins 3 alternatives (corrélations).")
    ben = is_benefit(types)
    R = np.empty_like(X)
    for j in range(n):
        lo, hi = X[:, j].min(), X[:, j].max()
        if hi == lo:
            raise ValueError(f"Critère {crit_labels[j]} : toutes les valeurs sont égales (max = min).")
        R[:, j] = (X[:, j] - lo) / (hi - lo) if ben[j] else (hi - X[:, j]) / (hi - lo)
    steps = [("1) Matrice normalisée (min-max selon le sens du critère)",
              pd.DataFrame(R, index=alt_labels, columns=crit_labels))]
    rho = np.corrcoef(R, rowvar=False)
    steps.append(("2) Matrice de corrélation ρ_jk", pd.DataFrame(rho, index=crit_labels, columns=crit_labels)))
    sigma = R.std(axis=0, ddof=1)
    conflict = (1 - np.abs(rho)).sum(axis=1)
    C = sigma * conflict
    w = C / C.sum()
    df = pd.DataFrame({"Moyenne r̄_j": R.mean(axis=0), "Écart-type σ_j": sigma,
                       "Σ(1 − |ρ_jk|)": conflict, "C_j": C, "Poids w_j": w}, index=crit_labels)
    steps.append(("3) Écart-type, conflit, indice C_j et poids", df))
    return {"weights": w, "corr": rho, "steps": steps}
