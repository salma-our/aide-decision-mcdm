"""Exemples et exercices du cours (Partie 1), chargeables en un clic dans l'application."""
from .utils import BENEFIT, COST

B, C = BENEFIT, COST

EXAMPLES = {
    "TOPSIS — Choix d'une voiture (4 marques)": {
        "alternatives": ["M1", "M2", "M3", "M4"],
        "criteria": ["Style", "Fiabilité", "Économie carburant", "Coût"],
        "types": [B, B, B, C],
        "matrix": [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]],
        "weights": [0.2, 0.1, 0.4, 0.3],
        "method": "TOPSIS",
    },
    "WSM / WPM / WASPAS — Choix d'une machine (5 critères)": {
        "alternatives": ["A1", "A2", "A3", "A4"],
        "criteria": ["C1", "C2", "C3", "C4", "C5"],
        "types": [C, C, B, C, B],
        "matrix": [[0.035, 847, 0.335, 1.760, 0.590],
                   [0.027, 834, 0.335, 1.680, 0.665],
                   [0.037, 808, 0.590, 2.400, 0.500],
                   [0.028, 821, 0.500, 1.590, 0.410]],
        "weights": [0.331, 0.181, 0.369, 0.072, 0.047],
        "method": "WASPAS",
    },
    "CRITIC — Machines de découpage": {
        "alternatives": ["A1", "A2", "A3", "A4"],
        "criteria": ["C1 Épaisseur max", "C2 Largeur coupe min", "C3 Qualité", "C4 Coût maintenance"],
        "types": [C, C, B, C],
        "matrix": [[30, 0.100, 1, 20], [100, 0.700, 1, 40], [50, 1, 2, 10], [300, 2, 3, 35]],
        "weights": [0.25, 0.25, 0.25, 0.25],
        "method": "CRITIC",
    },
    "VIKOR — Achat d'une machine (2 critères)": {
        "alternatives": ["A1", "A2", "A3"],
        "criteria": ["C1", "C2"],
        "types": [C, B],
        "matrix": [[1, 3000], [2, 3750], [5, 4500]],
        "weights": [0.5, 0.5],
        "method": "VIKOR",
    },
}

# Exercice AHP : achat d'une voiture
AHP_EXAMPLE = {
    "criteria": ["Coût", "Confort", "Sécurité"],
    "alternatives": ["Voiture 1", "Voiture 2"],
    "criteria_matrix": [["1", "7", "3"], ["1/7", "1", "1/3"], ["1/3", "3", "1"]],
    "alt_matrices": [
        [["1", "7"], ["1/7", "1"]],     # Coût
        [["1", "1/5"], ["5", "1"]],     # Confort
        [["1", "1/9"], ["9", "1"]],     # Sécurité
    ],
}

# Exemple BWM : choix d'un fournisseur
BWM_EXAMPLE = {
    "criteria": ["C1 Qualité", "C2 Prix", "C3 Délai", "C4 RSE"],
    "best": 0,
    "worst": 3,
    "BO": [1, 3, 4, 8],
    "OW": [8, 5, 3, 1],
}
