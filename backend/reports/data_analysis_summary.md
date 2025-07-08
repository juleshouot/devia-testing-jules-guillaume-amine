# Rapport d'Analyse des Pandémies
*Généré le 2025-05-27 10:35*

## 1. Statistiques Globales

### COVID
- **Cas totaux**: 520,833,195
- **Décès totaux**: 6,287,654
- **Pic quotidien de cas**: 3,829,248
- **Pic quotidien de décès**: 16,834
- **Taux de létalité global**: 1.21%

### Monkeypox
- **Cas totaux**: 261,726
- **Décès totaux**: 418
- **Pic quotidien de cas**: 5,406
- **Pic quotidien de décès**: 36
- **Taux de létalité global**: 0.16%

## 2. Analyse du Taux de Reproduction (Rt)

Le taux de reproduction effectif (Rt) est un indicateur clé qui mesure combien de personnes sont infectées par un individu contagieux.

### COVID
- **Rt moyen**: 1.29
- **Rt médian**: 0.96
- **Rt minimum**: 0.00
- **Rt maximum**: 10.00
- **Pourcentage du temps avec Rt > 1**: 45.4%

### Monkeypox
- **Rt moyen**: 0.77
- **Rt médian**: 0.00
- **Rt minimum**: 0.00
- **Rt maximum**: 10.00
- **Pourcentage du temps avec Rt > 1**: 21.3%

## 3. Analyse du Taux de Mortalité

Le taux de mortalité (Case Fatality Rate) mesure la proportion de cas confirmés qui aboutissent à un décès.

### COVID
- **CFR moyen**: 2.89%
- **CFR médian**: 0.65%
- **CFR minimum**: 0.00%
- **CFR maximum**: 100.00%

### Monkeypox
- **CFR moyen**: 0.39%
- **CFR médian**: 0.00%
- **CFR minimum**: 0.00%
- **CFR maximum**: 100.00%

## 4. Modèles Prédictifs

Performances des modèles entraînés:

### transmission - Random Forest
- **R²**: 0.9978680710833163
- **RMSE**: 0.08164673520676612
- **MAE**: 0.0055493817601768485
- **CV RMSE**: 0.16854708418525638

### mortality - GradientBoosting
- **R²**: nan
- **RMSE**: nan
- **MAE**: nan
- **CV RMSE**: nan

### transmission - Linear Regression
- **R²**: 0.035042428922618996
- **RMSE**: 1842.785617086059
- **MAE**: 170.73077864094827
- **CV RMSE**: 1731.8989354777411

### mortality - Random Forest
- **R²**: 0.9984848382144575
- **RMSE**: 0.002724994636315382
- **MAE**: 9.623012605816439e-05
- **CV RMSE**: 0.007231456000615563

### geographical - Ridge
- **R²**: nan
- **RMSE**: nan
- **MAE**: nan
- **CV RMSE**: nan

### geographical_spread - Random_Forest
- **R²**: -0.01674399821428585
- **RMSE**: 1.3256452412148416
- **MAE**: nan
- **CV RMSE**: nan

### unknown - unknown
- **R²**: nan
- **RMSE**: nan
- **MAE**: nan
- **CV RMSE**: nan

### transmission - GradientBoosting
- **R²**: nan
- **RMSE**: nan
- **MAE**: nan
- **CV RMSE**: nan

### geographical_spread - Linear Regression
- **R²**: 1.0
- **RMSE**: 1.3567149463006938e-15
- **MAE**: 8.881784197001252e-16
- **CV RMSE**: 3.640223280579608e-13

## 5. Visualisations Générées

Les visualisations suivantes ont été générées et sont disponibles dans le répertoire `/reports/figures/`:
- Tendances des nouveaux cas
- Tendances des cas cumulés
- Distribution des valeurs Rt
- Évolution temporelle de Rt
- Distribution du taux de mortalité
- Évolution temporelle du taux de mortalité
- Propagation géographique hebdomadaire
- Propagation géographique cumulative
- Comparaison des métriques des modèles

## 6. Conclusions et Recommandations

- La surveillance du Rt reste cruciale pour anticiper les vagues épidémiques
- Le taux de mortalité varie considérablement au cours du temps et entre les régions
- L'analyse de la propagation géographique permet d'identifier les schémas de transmission

---

*Ce rapport a été généré automatiquement. Les analyses sont basées sur les données disponibles et peuvent nécessiter une révision par des experts.*
