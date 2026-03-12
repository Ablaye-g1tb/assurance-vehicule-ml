# Scoring Souscription Assurance Véhicule — Régression Logistique
Prédire la probabilité qu'un client souscrive à une assurance véhicule, afin d'optimiser le ciblage commercial et réduire les contacts inutiles

# Dataset
- Source : Kaggle
- Taille : 381 109 clients, 10 variables
- Cible : Souscription (0 / 1)
- Déséquilibre : 87.8% classe 0 / 12.2% classe 1

# Démarche complète
1. Exploratory Data Analysis (EDA)

- Analyse de la distribution des variables continues et catégorielles
- Identification des variables à haute cardinalité (Code_Region : 53 valeurs, Canal_Vente : 155 valeurs)
- Analyse du déséquilibre des classes
- Croisements features -> cible :
  - Dommages_Vehicule : signal ×47 (23.8% vs 0.5%)
  - Assure_Avant : déjà assuré -> 0.1% de souscription
  - Age_Vehicule : relation linéaire (4.4% -> 17.4% -> 29.4%)
- Matrice de corrélation (variables continues uniquement)

2. Preprocessing

- Encodage ordinal : Age_Vehicule (< 1 an = 0, 1-2 ans = 1, > 2 ans = 2)
- Encodage binaire : Sexe, Dommages_Vehicule
- Code_Region et Canal_Vente -> passthrough (haute cardinalité, pas de scaling)
- Suppression de l'identifiant client

3. Modélisation

- Séparation X/y + Train/Test split (80/20, stratifié)
- Normalisation via Pipeline :

- RobustScaler -> Prime_Annuelle (distribution asymétrique)
- StandardScaler -> Age, Anciennete_Client


Pipeline sklearn pour éviter tout data leakage

4. Évaluation & Optimisation

- Modèle baseline-> AUC-ROC : 0.834, Recall : 0.98
- Validation croisée (StratifiedKFold, 5 folds) -> σ = 0.0014 (très stable)
- Analyse des coefficients -> Assure_Avant (-4.2), Dommages_Vehicule (+2.0)
- Feature Selection (RFECV) -> 10 -> 9 features (Code_Region supprimée)
- GridSearchCV → modèle insensible à C (plafond atteint)
- Optimisation du seuil de décision -> seuil optimal : 0.67

5. Résultats finaux
- AUC-ROC : 0.834 apres optimisation 0.834
- Recall classe 1: 0.98 -> 0.74
- Precision classe 1: 0.25 -> 0.31
- F1 classe 1: 0.40 -> 0.43
- Nb features 10 -> 9
