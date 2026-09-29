"""Bibliothèque de méthodes d'aide à la décision multicritère (MCDM).

Organisation :
    weighting.py : méthodes de pondération des critères (AHP, BWM, DEMATEL, Entropie, CRITIC)
    ranking.py   : méthodes de classement (WSM, WPM, WASPAS, TOPSIS, VIKOR) + registre RANKING_METHODS
    analysis.py  : comparaison des méthodes (Spearman) et analyse de sensibilité
    examples.py  : emplacement réservé aux jeux de données d'exemple
"""
from . import analysis, examples, ranking, utils, weighting  # noqa: F401
