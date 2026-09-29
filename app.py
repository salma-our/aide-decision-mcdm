"""
Aide à la décision multicritère (MCDM) — Application Streamlit
Cours : Aide à la décision — Partie 1 (EMI)

Lancer :  streamlit run app.py

Méthodes implémentées (Partie 1) :
  Pondération  : Saisie directe, AHP, BWM, DEMATEL (subjectives) ; Entropie, CRITIC (objectives)
  Classement   : WSM, WPM, WASPAS, TOPSIS, VIKOR, AHP (complet)
  Robustesse   : comparaison des classements (Spearman) et analyse de sensibilité des poids

Pour ajouter une méthode au fil du cours :
  1. écrire la fonction de calcul dans mcdm/weighting.py ou mcdm/ranking.py
     (elle renvoie {"weights"|"scores", "steps": [(titre, DataFrame|texte)], ...})
  2. pour un classement, l'ajouter au dictionnaire RANKING_METHODS (mcdm/ranking.py)
     et sa fiche dans METHOD_INFO ci-dessous ; pour une pondération, l'ajouter
     à WEIGHT_METHODS et écrire son petit bloc d'interface dans page_weighting().
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from mcdm import analysis, examples, ranking, weighting
from mcdm.utils import BENEFIT, COST, to_float_matrix

st.set_page_config(page_title="Aide à la décision multicritère", page_icon="⚖️", layout="wide")

ss = st.session_state
ss["_run"] = ss.get("_run", 0) + 1

TYPES = [BENEFIT, COST]
WEIGHT_METHODS = ["Saisie directe", "AHP", "BWM", "DEMATEL", "Entropie", "CRITIC"]
RANK_METHODS = ["WSM", "WPM", "WASPAS", "TOPSIS", "VIKOR", "AHP (complet)"]

METHOD_INFO = {
    "WSM": ("Somme pondérée — score le plus élevé = meilleur",
            [r"r_{ij}=\frac{x_{ij}}{x_j^+}\ (+)\qquad r_{ij}=\frac{x_j^+}{x_{ij}}\ (-)",
             r"Q_i^{(1)}=\sum_{j=1}^{n} w_j\, r_{ij}"]),
    "WPM": ("Produit pondéré — score le plus élevé = meilleur",
            [r"r_{ij}=\frac{x_{ij}}{x_j^+}\ (+)\qquad r_{ij}=\frac{x_j^+}{x_{ij}}\ (-)",
             r"Q_i^{(2)}=\prod_{j=1}^{n} r_{ij}^{\,w_j}"]),
    "WASPAS": ("Combinaison WSM + WPM (Zavadskas et al., 2012) — score le plus élevé = meilleur",
               [r"Q_i=\lambda\, Q_i^{(1)}+(1-\lambda)\, Q_i^{(2)},\quad \lambda\in[0,1]"]),
    "TOPSIS": ("Proximité à la solution idéale (Hwang & Yoon, 1981) — RC* le plus élevé = meilleur",
               [r"r_{ij}=\frac{x_{ij}}{\sqrt{\sum_i x_{ij}^2}},\qquad v_{ij}=w_j\, r_{ij}",
                r"S_i^+=\sqrt{\sum_j (v_j^*-v_{ij})^2},\qquad S_i^-=\sqrt{\sum_j (v_j'-v_{ij})^2}",
                r"RC_i^*=\frac{S_i^-}{S_i^-+S_i^+}"]),
    "VIKOR": ("Classement de compromis — Q le plus faible = meilleur",
              [r"S_i=\sum_j w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-},\qquad R_i=\max_j\left[w_j\frac{x_j^*-x_{ij}}{x_j^*-x_j^-}\right]",
               r"Q_i=v\frac{S_i-S^*}{S^--S^*}+(1-v)\frac{R_i-R^*}{R^--R^*}",
               r"C1:\ Q(A'')-Q(A')\ge \frac{1}{m-1}\qquad C2:\ A'\ \text{meilleur aussi selon } S \text{ et } R"]),
}


# ============================================================================
# Outils d'interface
# ============================================================================
def _same_labels(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    return list(a.index) == list(b.index) and list(a.columns) == list(b.columns)


def _align(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
    """Recopie les valeurs de `old` dans `new` par position (bloc supérieur gauche)."""
    out = new.copy()
    if old is None:
        return out
    r = min(old.shape[0], new.shape[0])
    c = min(old.shape[1], new.shape[1])
    if r and c:
        out.iloc[:r, :c] = old.iloc[:r, :c].values
    return out


def editor(name: str, df_init: pd.DataFrame, **kw) -> pd.DataFrame:
    """st.data_editor persistant : conserve les saisies quand on change de page / méthode,
    et se réaligne automatiquement quand les libellés (critères, alternatives) changent."""
    store = ss.setdefault("_ed", {})
    ent = store.get(name)
    run = ss["_run"]
    if ent is None:
        ent = {"cur": df_init.copy(), "ver": 0}
        rebuild = True
    else:
        rebuild = ent["last_run"] != run - 1 or not _same_labels(ent["cur"], df_init)
        if rebuild:
            ent["cur"] = _align(ent["cur"], df_init)
            ent["ver"] += 1
    if rebuild:
        ent["base"] = ent["cur"].copy()
    ent["last_run"] = run
    out = st.data_editor(ent["base"], key=f"_ed_{name}_{ent['ver']}", **kw)
    ent["cur"] = out
    store[name] = ent
    return out


def reset_editors(*names):
    store = ss.setdefault("_ed", {})
    if not names:
        store.clear()
    for n in names:
        store.pop(n, None)


def show_df(df: pd.DataFrame, precision: int = 4, **kw):
    try:
        st.dataframe(df.style.format(precision=precision, na_rep="—"), **kw)
    except Exception:  # noqa: BLE001
        st.dataframe(df, **kw)


def show_steps(steps):
    for title, obj in steps:
        st.markdown(f"**{title}**")
        if isinstance(obj, pd.DataFrame):
            show_df(obj)
        else:
            st.markdown(str(obj))


def pairwise_default(labels) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame([["1"] * n for _ in range(n)], index=labels, columns=labels)


def pairwise_to_float(df: pd.DataFrame, auto_reciprocal: bool) -> np.ndarray:
    A = np.array(to_float_matrix(df).to_numpy(), dtype=float, copy=True)
    n = A.shape[0]
    np.fill_diagonal(A, 1.0)
    if auto_reciprocal:
        for i in range(n):
            for j in range(i):
                if A[j, i] <= 0:
                    raise ValueError("Les valeurs de comparaison doivent être > 0.")
                A[i, j] = 1.0 / A[j, i]
    return A


def unique_labels(labels, prefix):
    out, seen = [], set()
    for k, lab in enumerate(labels):
        lab = str(lab).strip() if lab is not None and str(lab).strip() not in ("", "nan", "None") else f"{prefix}{k+1}"
        base, c = lab, 2
        while lab in seen:
            lab = f"{base} ({c})"
            c += 1
        seen.add(lab)
        out.append(lab)
    return out


def bar_weights(labels, w, title="Poids des critères"):
    fig = px.bar(x=list(labels), y=list(w), text=[f"{v:.3f}" for v in w],
                 labels={"x": "Critère", "y": "Poids"}, title=title)
    fig.update_traces(textposition="outside")
    fig.update_layout(height=340, margin=dict(t=50, b=10), yaxis_range=[0, max(w) * 1.25 if len(w) else 1])
    st.plotly_chart(fig)


def bar_scores(labels, scores, score_name, higher_is_better=True):
    order = np.argsort(-np.asarray(scores) if higher_is_better else np.asarray(scores), kind="stable")
    lab = [labels[i] for i in order]
    sc = [float(scores[i]) for i in order]
    colors = ["#2e7d32" if k == 0 else "#90a4ae" for k in range(len(sc))]
    fig = go.Figure(go.Bar(x=lab, y=sc, marker_color=colors, text=[f"{v:.4f}" for v in sc], textposition="outside"))
    fig.update_layout(title=f"{score_name} ({'plus élevé' if higher_is_better else 'plus faible'} = meilleur)",
                      height=360, margin=dict(t=50, b=10), yaxis_title=score_name)
    st.plotly_chart(fig)


def download_csv(df: pd.DataFrame, filename: str, label="⬇️ Télécharger (CSV)"):
    st.download_button(label, df.to_csv(sep=";", decimal=",").encode("utf-8-sig"),
                       file_name=filename, mime="text/csv")


# ============================================================================
# État du problème
# ============================================================================
def load_problem(alts, crits, types, X, weights, source="Saisie directe", extra=None):
    ss["alts"] = list(alts)
    ss["crits"] = list(crits)
    ss["types"] = list(types)
    ss["weights"] = list(map(float, weights))
    ss["weights_source"] = source
    ss["X"] = pd.DataFrame(np.asarray(X, dtype=float), index=ss["alts"], columns=ss["crits"])
    for k in ["ahp_crit_init", "ahp_alt_init", "bwm_init", "dematel_init"]:
        ss.pop(k, None)
    if extra:
        ss.update(extra)
    reset_editors()


def load_example(name: str):
    if name in examples.EXAMPLES:
        e = examples.EXAMPLES[name]
        load_problem(e["alternatives"], e["criteria"], e["types"], e["matrix"], e["weights"])
        ss["suggested_method"] = e["method"]
    elif name.startswith("AHP"):
        e = examples.AHP_EXAMPLE
        crits, alts = e["criteria"], e["alternatives"]
        crit_df = pd.DataFrame(e["criteria_matrix"], index=crits, columns=crits)
        alt_dfs = [pd.DataFrame(M, index=alts, columns=alts) for M in e["alt_matrices"]]
        w = weighting.ahp_weights(to_float_matrix(crit_df).to_numpy(), crits)["weights"]
        load_problem(alts, crits, [BENEFIT] * len(crits), np.full((len(alts), len(crits)), np.nan), w,
                     source="AHP", extra={"ahp_crit_init": crit_df, "ahp_alt_init": alt_dfs})
        ss["suggested_method"] = "AHP"
    elif name.startswith("BWM"):
        e = examples.BWM_EXAMPLE
        crits = e["criteria"]
        bwm_df = pd.DataFrame([e["BO"], e["OW"]], index=["BO (Best → autres)", "OW (autres → Worst)"],
                              columns=crits).astype(float)
        load_problem(["F1", "F2", "F3"], crits, [BENEFIT] * len(crits), np.full((3, len(crits)), np.nan),
                     [1 / len(crits)] * len(crits),
                     extra={"bwm_init": bwm_df, "bwm_best": 0, "bwm_worst": len(crits) - 1})
        ss["suggested_method"] = "BWM"


EXAMPLE_NAMES = list(examples.EXAMPLES) + ["AHP — Achat d'une voiture (exercice)", "BWM — Choix d'un fournisseur"]

if "alts" not in ss:
    load_example(EXAMPLE_NAMES[0])


def current_problem():
    """Retourne (X numpy, w normalisés, types, crits, alts) ou lève ValueError."""
    X = ss["X"].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("La matrice de décision contient des cases vides : complétez-la dans la page « 1. Données ».")
    w = np.asarray(ss["weights"], dtype=float)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Les poids doivent être positifs et de somme > 0.")
    return X, w / w.sum(), ss["types"], ss["crits"], ss["alts"]


# ============================================================================
# Barre latérale
# ============================================================================
st.sidebar.title("⚖️ Aide à la décision")
st.sidebar.caption("Analyse multicritère (MCDM) — Partie 1")
page = st.sidebar.radio("Navigation", ["🏠 Accueil", "1️⃣ Données du problème", "2️⃣ Pondération des critères",
                                       "3️⃣ Classement des alternatives", "4️⃣ Comparaison & robustesse"])
st.sidebar.divider()
st.sidebar.markdown("**📚 Exercices du cours**")
ex_choice = st.sidebar.selectbox("Exercice", EXAMPLE_NAMES, label_visibility="collapsed")
if st.sidebar.button("Charger l'exercice"):
    load_example(ex_choice)
    st.sidebar.success("Exercice chargé.")
if st.sidebar.button("🆕 Nouveau problème vide (3 × 3)"):
    load_problem(["A1", "A2", "A3"], ["C1", "C2", "C3"], [BENEFIT] * 3, np.full((3, 3), np.nan), [1 / 3] * 3)
st.sidebar.divider()
st.sidebar.markdown(f"**Problème courant** : {len(ss['alts'])} alternatives × {len(ss['crits'])} critères")
st.sidebar.markdown(f"**Poids** : {ss.get('weights_source', 'Saisie directe')}")
_w = np.asarray(ss["weights"], float)
_w = _w / _w.sum() if _w.sum() > 0 else _w
st.sidebar.dataframe(pd.DataFrame({"Type": ss["types"], "Poids": np.round(_w, 4)}, index=ss["crits"]))


# ============================================================================
# Pages
# ============================================================================
def page_home():
    st.title("⚖️ Aide à la décision multicritère (MCDM)")
    st.markdown(
        "Cette plateforme accompagne le cours **Aide à la décision**. Vous saisissez le problème "
        "(alternatives, critères, matrice de décision), vous choisissez une méthode, et l'application "
        "calcule le résultat **en détaillant chaque étape** comme dans le cours."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.info("**1. Données**\n\nAlternatives, critères (+/−), matrice de décision, import/export.")
    c2.info("**2. Pondération**\n\nAHP, BWM, DEMATEL, Entropie, CRITIC ou saisie directe.")
    c3.info("**3. Classement**\n\nWSM, WPM, WASPAS, TOPSIS, VIKOR, AHP complet.")
    c4.info("**4. Robustesse**\n\nComparaison des méthodes (Spearman) et sensibilité des poids.")

    st.subheader("Méthodes disponibles (Partie 1)")
    st.dataframe(pd.DataFrame([
        ["Pondération — subjective", "AHP", "Comparaisons par paires (échelle de Saaty 1–9), λmax, CI, CR < 0,1"],
        ["Pondération — subjective", "BWM", "Best / Worst, 2n − 3 comparaisons, programme linéaire, ξ*"],
        ["Pondération — subjective", "DEMATEL", "Relations d'influence entre critères (cause / effet)"],
        ["Pondération — objective", "Entropie", "Dispersion de l'information : E_j, 1 − E_j"],
        ["Pondération — objective", "CRITIC", "Écart-type × conflit (corrélations)"],
        ["Classement — additive", "WSM / WPM / WASPAS", "Somme, produit pondérés et leur combinaison (λ)"],
        ["Classement — additive", "TOPSIS", "Distance aux solutions idéales positive et négative"],
        ["Classement — additive", "VIKOR", "Solution de compromis (S, R, Q, conditions C1 et C2)"],
        ["Classement — additive", "AHP complet", "Priorités locales × poids des critères"],
        ["Robustesse", "Spearman / Sensibilité", "Comparer les classements, faire varier les poids"],
    ], columns=["Étape", "Méthode", "Principe"]), hide_index=True)

    st.subheader("Quelle méthode choisir ?")
    st.markdown(
        "- **Données exactes** → méthodes *crisp* (cette partie) ; jugements imprécis → méthodes floues (parties suivantes).\n"
        "- **Effort demandé au décideur** : AHP exige n(n−1)/2 comparaisons, BWM seulement 2n−3, "
        "CRITIC et Entropie aucune.\n"
        "- **Compensation acceptée** → méthodes additives (WSM, TOPSIS, VIKOR…).\n"
        "- **Bonnes pratiques** : appliquer plusieurs méthodes, comparer les classements (Spearman) et "
        "mener une analyse de sensibilité sur les poids (page 4)."
    )
    st.caption("💡 Chargez un exercice du cours depuis la barre latérale pour tester rapidement.")


def page_data():
    st.title("1️⃣ Données du problème")
    st.caption("Saisissez les alternatives, les critères (sens : Max = critère positif, Min = critère négatif), "
               "leurs poids et la matrice de décision. Les valeurs décimales acceptent le point.")

    c1, c2 = st.columns(2)
    m = int(c1.number_input("Nombre d'alternatives (m)", 2, 50, len(ss["alts"])))
    n = int(c2.number_input("Nombre de critères (n)", 2, 30, len(ss["crits"])))
    if m != len(ss["alts"]) or n != len(ss["crits"]):
        alts = (ss["alts"] + [f"A{k+1}" for k in range(len(ss["alts"]), m)])[:m]
        crits = (ss["crits"] + [f"C{k+1}" for k in range(len(ss["crits"]), n)])[:n]
        types = (ss["types"] + [BENEFIT] * n)[:n]
        wmean = float(np.mean(ss["weights"])) if ss["weights"] else 1.0
        weights = (list(ss["weights"]) + [wmean] * n)[:n]
        X = _align(ss["X"], pd.DataFrame(np.nan, index=alts, columns=crits))
        ss.update(alts=alts, crits=crits, types=types, weights=weights, X=X)
        reset_editors("alts", "crit")

    left, right = st.columns([1, 2])
    with left:
        st.subheader("Alternatives")
        df_a = editor("alts", pd.DataFrame({"Alternative": ss["alts"]}), hide_index=True, num_rows="fixed")
        alts = unique_labels(df_a["Alternative"].tolist(), "A")
    with right:
        st.subheader("Critères")
        df_c = editor("crit", pd.DataFrame({"Critère": ss["crits"], "Type": ss["types"],
                                            "Poids": np.asarray(ss["weights"], float)}),
                      hide_index=True, num_rows="fixed",
                      column_config={
                          "Type": st.column_config.SelectboxColumn("Type", options=TYPES, required=True),
                          "Poids": st.column_config.NumberColumn("Poids", min_value=0.0, format="%.4f"),
                      })
        crits = unique_labels(df_c["Critère"].tolist(), "C")
        types = [t if t in TYPES else BENEFIT for t in df_c["Type"].tolist()]
        weights = pd.to_numeric(df_c["Poids"], errors="coerce").fillna(0).clip(lower=0).tolist()
        tot = sum(weights)
        if abs(tot - 1) > 1e-6:
            st.warning(f"Somme des poids = {tot:.4f} ≠ 1 : les poids seront normalisés automatiquement.")
        if list(weights) != list(ss["weights"]):
            ss["weights_source"] = "Saisie directe"
    ss.update(alts=alts, crits=crits, types=types, weights=weights)

    st.subheader("Matrice de décision")
    X_init = pd.DataFrame(ss["X"].to_numpy(), index=alts, columns=crits)
    X = editor("X", X_init, num_rows="fixed",
               column_config={c: st.column_config.NumberColumn(c, format="%g") for c in crits})
    X = X.apply(pd.to_numeric, errors="coerce")
    ss["X"] = X
    if X.isna().any().any():
        st.warning("Certaines cases sont vides : complétez la matrice avant de lancer une méthode de classement "
                   "(sauf AHP complet, qui n'utilise pas la matrice).")
    else:
        st.success("Matrice complète ✅")

    with st.expander("📥 Importer / 📤 Exporter"):
        st.markdown(
            "Format du fichier (CSV `;` ou Excel) : la **première colonne** contient les alternatives, la "
            "**première ligne** les critères. Deux lignes facultatives nommées **Type** (Max / Min) et "
            "**Poids** peuvent être ajoutées."
        )
        up = st.file_uploader("Fichier CSV ou Excel", type=["csv", "xlsx", "xls"])
        if up is not None and st.button("Importer ce fichier"):
            try:
                if up.name.lower().endswith(".csv"):
                    raw = up.getvalue().decode("utf-8-sig")
                    sep = ";" if raw.count(";") >= raw.count(",") else ","
                    df = pd.read_csv(io.StringIO(raw), sep=sep, index_col=0, dtype=str)
                else:
                    df = pd.read_excel(up, index_col=0, dtype=str)
                df.index = df.index.map(str)
                low = {i.strip().lower(): i for i in df.index}
                tcol = low.get("type")
                pcol = low.get("poids") or low.get("weights") or low.get("poids w")
                types_in = [BENEFIT] * df.shape[1]
                if tcol:
                    types_in = [COST if str(v).strip().lower().startswith(("min", "-", "−", "cost", "coût")) else BENEFIT
                                for v in df.loc[tcol]]
                w_in = [1 / df.shape[1]] * df.shape[1]
                if pcol:
                    w_in = to_float_matrix(df.loc[[pcol]]).to_numpy()[0].tolist()
                data = df.drop(index=[i for i in [tcol, pcol] if i])
                Xin = to_float_matrix(data)
                load_problem([str(a) for a in data.index], [str(c) for c in data.columns], types_in, Xin.to_numpy(), w_in)
                st.success("Fichier importé ✅")
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Import impossible : {e}")

        export = X.copy().astype(object)
        export.loc["Type"] = ["Max" if t == BENEFIT else "Min" for t in types]
        export.loc["Poids"] = weights
        download_csv(export, "probleme_mcdm.csv", "⬇️ Exporter le problème (CSV)")


# ----------------------------------------------------------------------------
def apply_weights_button(w, source):
    w = np.asarray(w, float)
    if st.button(f"✅ Utiliser ces poids ({source}) pour le classement", type="primary"):
        ss["weights"] = list(map(float, w / w.sum()))
        ss["weights_source"] = source
        reset_editors("crit")
        st.success("Poids enregistrés : ils sont maintenant utilisés dans les pages 3 et 4.")


def page_weighting():
    st.title("2️⃣ Pondération des critères")
    crits, alts, types = ss["crits"], ss["alts"], ss["types"]
    n = len(crits)
    default = ss.get("suggested_method") if ss.get("suggested_method") in WEIGHT_METHODS else "Saisie directe"
    method = st.selectbox("Méthode de pondération", WEIGHT_METHODS, index=WEIGHT_METHODS.index(default),
                          help="Subjectives : AHP, BWM, DEMATEL — Objectives : Entropie, CRITIC")

    # --------------------------------------------------------------- Saisie directe
    if method == "Saisie directe":
        st.info("Les poids se saisissent dans la page **1. Données** (colonne « Poids » du tableau des critères).")
        w = np.asarray(ss["weights"], float)
        if w.sum() > 0:
            bar_weights(crits, w / w.sum())

    # --------------------------------------------------------------- AHP
    elif method == "AHP":
        st.markdown(f"Comparez les critères deux à deux (échelle de Saaty). "
                    f"Nombre de comparaisons : **n(n−1)/2 = {n*(n-1)//2}**. Les fractions sont acceptées (ex. `1/7`).")
        with st.expander("📏 Échelle de Saaty"):
            st.dataframe(weighting.SAATY_SCALE, hide_index=True)
        c1, c2 = st.columns(2)
        auto = c1.toggle("Compléter automatiquement le triangle inférieur (a_ji = 1/a_ij)", value=True)
        calc = c2.radio("Calcul du vecteur propre", ["Méthode approximative (cours)", "Vecteur propre exact"],
                        horizontal=True)
        init = ss.get("ahp_crit_init")
        if init is None or list(init.index) != crits:
            init = pairwise_default(crits)
        st.markdown("**Matrice de comparaison des critères** (saisir la ligne i vs la colonne j)")
        dfA = editor("ahp_crit", init)
        try:
            A = pairwise_to_float(dfA, auto)
            if not auto:
                for msg in weighting.ahp_check_reciprocal(A):
                    st.warning(msg)
            res = weighting.ahp_weights(A, crits, "eigen" if calc.startswith("Vecteur") else "approx")
            k1, k2, k3 = st.columns(3)
            k1.metric("λmax", f"{res['lambda_max']:.4f}")
            k2.metric("CI", f"{res['CI']:.4f}")
            k3.metric("CR", f"{res['CR']:.4f}", "cohérente" if res["consistent"] else "incohérente",
                      delta_color="normal" if res["consistent"] else "inverse")
            if not res["consistent"]:
                st.error("CR ≥ 0,1 : les jugements sont incohérents, révisez la matrice de comparaison.")
            if n > 10:
                st.caption("n > 10 : la valeur de RI est indicative (le tableau du cours s'arrête à n = 10).")
            bar_weights(crits, res["weights"], "Poids AHP")
            with st.expander("🔎 Détail des calculs", expanded=True):
                show_steps(res["steps"])
            apply_weights_button(res["weights"], "AHP")
        except ValueError as e:
            st.error(str(e))

    # --------------------------------------------------------------- BWM
    elif method == "BWM":
        st.markdown(f"Choisissez le critère le plus important (**Best**) et le moins important (**Worst**), "
                    f"puis notez de 1 à 9. Nombre de comparaisons : **2n − 3 = {2*n-3}** (contre {n*(n-1)//2} pour AHP).")
        c1, c2 = st.columns(2)
        b = c1.selectbox("Best (le plus important)", range(n), format_func=lambda i: crits[i],
                         index=min(ss.get("bwm_best", 0), n - 1))
        wst = c2.selectbox("Worst (le moins important)", range(n), format_func=lambda i: crits[i],
                           index=min(ss.get("bwm_worst", n - 1), n - 1))
        ss["bwm_best"], ss["bwm_worst"] = b, wst
        init = ss.get("bwm_init")
        if init is None or list(init.columns) != crits:
            init = pd.DataFrame([[1.0] * n, [1.0] * n], index=["BO (Best → autres)", "OW (autres → Worst)"],
                                columns=crits)
        st.markdown(f"**BO** : combien de fois *{crits[b]}* est plus important que chaque critère ?  \n"
                    f"**OW** : combien de fois chaque critère est plus important que *{crits[wst]}* ?")
        dfb = editor("bwm", init,
                     column_config={c: st.column_config.NumberColumn(c, min_value=1, max_value=9, step=1)
                                    for c in crits})
        try:
            vals = dfb.apply(pd.to_numeric, errors="coerce")
            if vals.isna().any().any():
                raise ValueError("Complétez toutes les notes (1 à 9).")
            res = weighting.bwm_weights(b, wst, vals.iloc[0].to_numpy(), vals.iloc[1].to_numpy(), crits)
            for msg in res["warnings"]:
                st.warning(msg)
            st.metric("ξ* (indicateur de cohérence, proche de 0 = cohérent)", f"{res['xi']:.4f}")
            bar_weights(crits, res["weights"], "Poids BWM")
            with st.expander("🔎 Détail des calculs (programme linéaire)", expanded=True):
                show_steps(res["steps"])
            apply_weights_button(res["weights"], "BWM")
        except ValueError as e:
            st.error(str(e))

    # --------------------------------------------------------------- DEMATEL
    elif method == "DEMATEL":
        st.markdown("Indiquez l'**influence directe** de chaque critère (ligne) sur chaque autre critère (colonne) : "
                    "0 = aucune, 1 = faible, 2 = moyenne, 3 = forte, 4 = très forte. La diagonale est ignorée (0).")
        init = ss.get("dematel_init")
        if init is None or list(init.index) != crits:
            init = pd.DataFrame(0.0, index=crits, columns=crits)
        dfz = editor("dematel", init,
                     column_config={c: st.column_config.NumberColumn(c, min_value=0, max_value=4, step=1)
                                    for c in crits})
        try:
            Z = np.array(dfz.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(), dtype=float, copy=True)
            np.fill_diagonal(Z, 0)
            res = weighting.dematel_weights(Z, crits)
            c1, c2 = st.columns(2)
            with c1:
                bar_weights(crits, res["weights"], "Poids DEMATEL")
            with c2:
                t = res["table"]
                fig = px.scatter(t, x="D + R (proéminence)", y="D − R (relation)", text=t.index,
                                 color="Groupe", title="Diagramme cause–effet",
                                 color_discrete_map={"Cause": "#c62828", "Effet": "#1565c0"})
                fig.add_hline(y=0, line_dash="dash", line_color="grey")
                fig.update_traces(textposition="top center", marker_size=12)
                fig.update_layout(height=340, margin=dict(t=50, b=10))
                st.plotly_chart(fig)
            with st.expander("🔎 Détail des calculs", expanded=True):
                show_steps(res["steps"])
                st.markdown(f"**Relations significatives** (t_ij > seuil = moyenne de T = {res['threshold']:.4f}) :")
                st.markdown("\n".join(f"- {a} → {b_} ({v:.3f})" for a, b_, v in res["links"]) or "_aucune_")
            apply_weights_button(res["weights"], "DEMATEL")
        except (ValueError, np.linalg.LinAlgError) as e:
            st.error(str(e))

    # --------------------------------------------------------------- Entropie
    elif method == "Entropie":
        st.latex(r"p_{ij}=\frac{d_{ij}}{\sum_i d_{ij}},\quad E_j=-\frac{1}{\ln m}\sum_i p_{ij}\ln p_{ij},\quad "
                 r"w_j=\frac{1-E_j}{n-\sum_k E_k}")
        pre = st.radio("Données utilisées pour p_ij",
                       ["Matrice brute", "Matrice normalisée (x/max pour +, min/x pour −)"], horizontal=True)
        try:
            X = ss["X"].to_numpy(float)
            if np.isnan(X).any():
                raise ValueError("Complétez la matrice de décision (page 1).")
            res = weighting.entropy_weights(X, crits, alts, types, "linear" if pre.startswith("Matrice normalisée") else "none")
            bar_weights(crits, res["weights"], "Poids par l'entropie")
            with st.expander("🔎 Détail des calculs", expanded=True):
                show_steps(res["steps"])
            apply_weights_button(res["weights"], "Entropie")
        except ValueError as e:
            st.error(str(e))

    # --------------------------------------------------------------- CRITIC
    elif method == "CRITIC":
        st.latex(r"r_{ij}=\frac{x_{ij}-\min x_j}{\max x_j-\min x_j}\ (+),\quad "
                 r"r_{ij}=\frac{\max x_j-x_{ij}}{\max x_j-\min x_j}\ (-)")
        st.latex(r"C_j=\sigma_j\sum_{k=1}^{n}\left(1-|\rho_{jk}|\right),\qquad w_j=\frac{C_j}{\sum_j C_j}")
        try:
            X = ss["X"].to_numpy(float)
            if np.isnan(X).any():
                raise ValueError("Complétez la matrice de décision (page 1).")
            res = weighting.critic_weights(X, crits, alts, types)
            c1, c2 = st.columns(2)
            with c1:
                bar_weights(crits, res["weights"], "Poids CRITIC")
            with c2:
                fig = px.imshow(res["corr"], x=crits, y=crits, text_auto=".2f", zmin=-1, zmax=1,
                                color_continuous_scale="RdBu", title="Corrélations ρ_jk")
                fig.update_layout(height=340, margin=dict(t=50, b=10))
                st.plotly_chart(fig)
            with st.expander("🔎 Détail des calculs", expanded=True):
                show_steps(res["steps"])
            apply_weights_button(res["weights"], "CRITIC")
        except ValueError as e:
            st.error(str(e))


# ----------------------------------------------------------------------------
def method_params(name: str, key_prefix: str) -> dict:
    if name == "WASPAS":
        return {"lam": st.slider("λ (poids de WSM)", 0.0, 1.0, 0.5, 0.05, key=f"{key_prefix}_lam")}
    if name == "VIKOR":
        c1, c2 = st.columns(2)
        v = c1.slider("v (stratégie de la majorité)", 0.0, 1.0, 0.5, 0.05, key=f"{key_prefix}_v")
        strict = c2.toggle("C2 stricte : meilleur selon S **et** R (cours)", value=True, key=f"{key_prefix}_strict",
                           help="Désactivé : meilleur selon S et/ou R (Opricovic)")
        return {"v": v, "strict": strict}
    return {}


def page_ranking():
    st.title("3️⃣ Classement des alternatives")
    default = ss.get("suggested_method")
    if default == "AHP":
        default = "AHP (complet)"
    idx = RANK_METHODS.index(default) if default in RANK_METHODS else 3
    method = st.selectbox("Méthode de classement", RANK_METHODS, index=idx)
    st.caption(f"Poids utilisés : **{ss.get('weights_source', 'Saisie directe')}** — modifiables en page 1 ou 2.")

    if method == "AHP (complet)":
        page_ahp_full()
        return

    desc, formulas = METHOD_INFO[method]
    with st.expander(f"📘 Rappel : {method} — {desc}"):
        for f in formulas:
            st.latex(f)
    params = method_params(method, "rank")
    try:
        X, w, types, crits, alts = current_problem()
        with st.expander("Données utilisées"):
            d = pd.DataFrame(X, index=alts, columns=crits)
            d.loc["Type"] = ["+" if t == BENEFIT else "−" for t in types]
            d.loc["Poids"] = w
            show_df(d)
        res = ranking.RANKING_METHODS[method](X, w, types, crits, alts, **params)
    except ValueError as e:
        st.error(str(e))
        return

    best = res["result"].index[0]
    if method == "VIKOR":
        comp = res["compromise"]
        st.success(f"🏆 Solution de compromis : **{', '.join(comp)}**" +
                   ("" if len(comp) == 1 else " (ensemble de compromis)"))
    else:
        st.success(f"🏆 Meilleure alternative selon {method} : **{best}**")

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown("**Classement final**")
        show_df(res["result"])
        download_csv(res["result"], f"resultat_{method}.csv")
    with c2:
        name = {"WSM": "Q¹", "WPM": "Q²", "WASPAS": "Q", "TOPSIS": "RC*", "VIKOR": "Q"}[method]
        bar_scores(alts, res["scores"], name, res["higher_is_better"])
    with st.expander("🔎 Détail des calculs, étape par étape", expanded=True):
        show_steps(res["steps"])


def page_ahp_full():
    crits, alts = ss["crits"], ss["alts"]
    m = len(alts)
    st.markdown("**AHP complet** : pour chaque critère, comparez les alternatives deux à deux ; les priorités "
                "locales sont ensuite combinées avec les poids des critères (priorité globale = Σ w_j · w_ij).")
    w = np.asarray(ss["weights"], float)
    w = w / w.sum()
    if ss.get("weights_source") != "AHP":
        st.info("Les poids actuels ne proviennent pas d'AHP. Pour suivre la démarche du cours, calculez-les "
                "d'abord par AHP (page 2) puis cliquez sur « Utiliser ces poids ».")
    bar_weights(crits, w, f"Poids des critères ({ss.get('weights_source')})")
    auto = st.toggle("Compléter automatiquement le triangle inférieur (a_ji = 1/a_ij)", value=True, key="ahpfull_auto")
    inits = ss.get("ahp_alt_init") or []
    tabs = st.tabs([f"Critère : {c}" for c in crits])
    local = np.zeros((m, len(crits)))
    ok = True
    cr_rows = []
    for j, (tab, c) in enumerate(zip(tabs, crits)):
        with tab:
            init = inits[j] if j < len(inits) and list(inits[j].index) == alts else pairwise_default(alts)
            dfA = editor(f"ahp_alt_{j}", init)
            try:
                A = pairwise_to_float(dfA, auto)
                if not auto:
                    for msg in weighting.ahp_check_reciprocal(A):
                        st.warning(msg)
                r = weighting.ahp_weights(A, alts)
                local[:, j] = r["weights"]
                cr_rows.append([c, r["lambda_max"], r["CI"], r["CR"], "Oui ✅" if r["consistent"] or m <= 2 else "Non ❌"])
                with st.expander("Détail"):
                    show_steps(r["steps"])
            except ValueError as e:
                st.error(str(e))
                ok = False
    if not ok:
        return
    st.markdown("**Cohérence des matrices de comparaison des alternatives**")
    show_df(pd.DataFrame(cr_rows, columns=["Critère", "λmax", "CI", "CR", "Cohérente ?"]).set_index("Critère"))
    if m <= 2:
        st.caption("Avec 2 alternatives, une matrice réciproque est toujours parfaitement cohérente (CR = 0).")
    scores = local @ w
    final = pd.DataFrame(local, index=alts, columns=crits)
    final["Priorité globale"] = scores
    final["Rang"] = pd.Series(scores, index=alts).rank(ascending=False, method="min").astype(int)
    final = final.sort_values("Rang")
    wrow = pd.DataFrame([list(w) + [np.nan, np.nan]], index=["Poids des critères"], columns=final.columns)
    st.success(f"🏆 Meilleure alternative selon AHP : **{final.index[0]}**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Priorités locales et priorités globales**")
        show_df(pd.concat([wrow, final]))
        download_csv(final, "resultat_AHP.csv")
    with c2:
        bar_scores(alts, scores, "Priorité globale", True)


# ----------------------------------------------------------------------------
def page_compare():
    st.title("4️⃣ Comparaison des méthodes & robustesse")
    try:
        X, w, types, crits, alts = current_problem()
    except ValueError as e:
        st.error(str(e))
        return
    methods = st.multiselect("Méthodes à comparer", list(ranking.RANKING_METHODS), default=list(ranking.RANKING_METHODS))
    if len(methods) < 1:
        st.info("Choisissez au moins une méthode.")
        return
    with st.expander("Paramètres (WASPAS, VIKOR)"):
        params = {"WASPAS": method_params("WASPAS", "cmp"), "VIKOR": method_params("VIKOR", "cmp")}
    try:
        results = analysis.run_all(X, w, types, crits, alts, methods, params)
    except ValueError as e:
        st.error(str(e))
        return

    tab1, tab2 = st.tabs(["📊 Comparaison des classements", "🎚️ Analyse de sensibilité"])
    with tab1:
        rt = analysis.ranks_table(results, alts)
        scores = pd.DataFrame({k: r["scores"] for k, r in results.items()}, index=alts)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Rangs obtenus** (1 = meilleur)")
            st.dataframe(rt)
            st.caption("VIKOR : rang selon Q (plus petit = meilleur).")
        with c2:
            st.markdown("**Scores**")
            show_df(scores)
        long = rt.reset_index(names="Alternative").melt(id_vars="Alternative", var_name="Méthode", value_name="Rang")
        fig = px.line(long, x="Méthode", y="Rang", color="Alternative", markers=True, title="Rangs par méthode")
        fig.update_yaxes(autorange="reversed", dtick=1)
        fig.update_layout(height=380)
        st.plotly_chart(fig)
        if len(methods) >= 2:
            st.markdown("**Corrélation de rang de Spearman** entre méthodes")
            st.latex(r"\rho = 1-\frac{6\sum_i d_i^2}{m(m^2-1)}")
            sp = analysis.spearman_matrix(rt)
            fig = px.imshow(sp, text_auto=".3f", zmin=-1, zmax=1, color_continuous_scale="RdYlGn")
            fig.update_layout(height=380, margin=dict(t=20))
            st.plotly_chart(fig)
            st.caption("ρ proche de 1 : les deux méthodes donnent des classements très semblables.")
        # alternative la plus souvent première
        firsts = (rt == 1).sum(axis=1).sort_values(ascending=False)
        st.info(f"Alternative classée 1ʳᵉ le plus souvent : **{firsts.index[0]}** ({firsts.iloc[0]}/{len(methods)} méthodes)")
        download_csv(rt, "comparaison_rangs.csv")

    with tab2:
        st.markdown("On fait varier le poids d'**un critère** de 0 à 1 ; les autres poids sont réajustés "
                    "proportionnellement pour que la somme reste égale à 1.")
        c1, c2 = st.columns(2)
        meth = c1.selectbox("Méthode", methods, key="sens_m")
        j = c2.selectbox("Critère dont on fait varier le poids", range(len(crits)), format_func=lambda i: crits[i])
        grid = np.round(np.linspace(0, 1, 41), 3)
        df_s, df_r = analysis.sensitivity(X, w, types, crits, alts, meth, j, grid, params.get(meth, {}))
        for df, title, rev in [(df_s, "Score", False), (df_r, "Rang", True)]:
            long = df.reset_index().melt(id_vars=df.index.name, var_name="Alternative", value_name=title)
            fig = px.line(long, x=df.index.name, y=title, color="Alternative",
                          title=f"{title} en fonction du poids de {crits[j]} ({meth})",
                          line_shape="hv" if rev else "linear")
            fig.add_vline(x=float(w[j]), line_dash="dash", line_color="black",
                          annotation_text=f"poids actuel = {w[j]:.3f}")
            if rev:
                fig.update_yaxes(autorange="reversed", dtick=1)
            fig.update_layout(height=380)
            st.plotly_chart(fig)
        # changements du meilleur
        best = df_r.apply(lambda row: ", ".join(row[row == 1].index) if row.notna().all() else "—", axis=1)
        changes = best[best != best.shift()]
        st.markdown("**Évolution de l'alternative classée 1ʳᵉ**")
        st.dataframe(pd.DataFrame({"À partir du poids": changes.index, "Meilleure alternative": changes.values}),
                     hide_index=True)


# ============================================================================
PAGES = {
    "🏠 Accueil": page_home,
    "1️⃣ Données du problème": page_data,
    "2️⃣ Pondération des critères": page_weighting,
    "3️⃣ Classement des alternatives": page_ranking,
    "4️⃣ Comparaison & robustesse": page_compare,
}
PAGES[page]()
