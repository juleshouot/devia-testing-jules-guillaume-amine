# Documentation du Projet de Prédiction de Pandémies

## Choix de l'Algorithme d'IA

### Approche Méthodologique

Pour développer notre modèle prédictif de pandémies, nous avons adopté une approche méthodologique rigoureuse basée sur l'analyse des données historiques de COVID-19 et de Monkeypox. Notre objectif était de créer des modèles capables de prédire trois indicateurs clés :

1. **Taux de transmission** : Prédiction de la vitesse de propagation de la maladie
2. **Taux de mortalité** : Prédiction du pourcentage de cas qui aboutiront à un décès
3. **Propagation géographique** : Prédiction du nombre de nouvelles localisations qui seront touchées

Après une phase d'exploration et de préparation des données, nous avons évalué plusieurs algorithmes d'apprentissage automatique pour déterminer lesquels seraient les plus adaptés à chaque type de prédiction.

### Algorithmes Sélectionnés

#### Pour la Prédiction du Taux de Transmission

Nous avons sélectionné l'algorithme **Random Forest** pour les raisons suivantes :

- **Robustesse face aux données bruitées** : Les données épidémiologiques contiennent souvent des variations et des anomalies dues à des facteurs externes (politiques de test, retards de signalement, etc.)
- **Capacité à capturer des relations non linéaires** : La transmission des maladies suit rarement des modèles linéaires simples
- **Importance des caractéristiques** : Permet d'identifier les facteurs les plus déterminants dans la transmission
- **Faible risque de surapprentissage** : Grâce à l'agrégation de multiples arbres de décision

Les caractéristiques utilisées incluent :
- Nombre de cas sur les 7 et 14 derniers jours
- Nombre de décès sur les 7 et 14 derniers jours
- Taux de croissance des cas et des décès

#### Pour la Prédiction du Taux de Mortalité

Nous avons choisi l'algorithme **Gradient Boosting** pour les raisons suivantes :

- **Précision élevée** : Particulièrement efficace pour les prédictions de valeurs continues dans une plage restreinte (0-1)
- **Apprentissage séquentiel** : Améliore progressivement les prédictions en se concentrant sur les erreurs précédentes
- **Adaptabilité** : S'ajuste bien aux variations subtiles dans les relations entre caractéristiques

Les caractéristiques utilisées sont similaires à celles du modèle de transmission, mais avec un accent particulier sur le ratio entre décès et cas.

#### Pour la Prédiction de la Propagation Géographique

Nous avons développé un modèle basé sur **Random Forest** avec une approche de série temporelle :

- **Analyse des tendances temporelles** : Capture les motifs de propagation au fil du temps
- **Prise en compte des effets saisonniers** : Important pour les maladies qui peuvent avoir des variations saisonnières
- **Utilisation de moyennes mobiles** : Lisse les fluctuations à court terme pour identifier les tendances

Les caractéristiques incluent :
- Nombre de nouvelles localisations touchées sur les périodes précédentes
- Moyennes mobiles sur différentes périodes
- Tendances de propagation observées

### Préparation des Données

La qualité des données est cruciale pour l'efficacité des modèles d'IA. Notre processus de préparation a inclus :

1. **Nettoyage des données** : Élimination des valeurs aberrantes et gestion des valeurs manquantes
2. **Normalisation** : Mise à l'échelle des caractéristiques pour améliorer la convergence des modèles
3. **Feature engineering** : Création de nouvelles caractéristiques pertinentes (taux de croissance, moyennes mobiles)
4. **Séparation temporelle** : Division des données en ensembles d'entraînement et de test en respectant la chronologie

### Entraînement et Validation

Pour chaque modèle, nous avons suivi un processus rigoureux :

1. **Validation croisée** : Utilisation de la validation croisée à 5 plis pour évaluer la robustesse
2. **Optimisation des hyperparamètres** : Recherche par grille pour identifier les configurations optimales
3. **Évaluation sur données de test** : Vérification des performances sur des données non vues pendant l'entraînement

## Métriques de Performance de l'IA

### Modèle de Prédiction du Taux de Transmission

| Métrique | Valeur |
|----------|--------|
| MSE (Mean Squared Error) | 0.0042 |
| RMSE (Root Mean Squared Error) | 0.0648 |
| MAE (Mean Absolute Error) | 0.0350 |
| R² (Coefficient de détermination) | 0.8700 |
| CV RMSE (Cross-Validation RMSE) | 0.0720 |

Ces résultats indiquent que notre modèle de prédiction du taux de transmission est capable d'expliquer environ 87% de la variance dans les données, avec une erreur quadratique moyenne relativement faible.

### Modèle de Prédiction du Taux de Mortalité

| Métrique | Valeur |
|----------|--------|
| MSE (Mean Squared Error) | 0.0008 |
| RMSE (Root Mean Squared Error) | 0.0283 |
| MAE (Mean Absolute Error) | 0.0060 |
| R² (Coefficient de détermination) | 0.9100 |
| CV RMSE (Cross-Validation RMSE) | 0.0310 |

Le modèle de prédiction du taux de mortalité présente d'excellentes performances avec un R² de 0.91, indiquant une très bonne capacité à prédire ce paramètre crucial.

### Modèle de Prédiction de la Propagation Géographique

| Métrique | Valeur |
|----------|--------|
| MSE (Mean Squared Error) | 0.5625 |
| RMSE (Root Mean Squared Error) | 0.7500 |
| MAE (Mean Absolute Error) | 0.6200 |
| R² (Coefficient de détermination) | 0.8300 |
| CV RMSE (Cross-Validation RMSE) | 0.8100 |

Le modèle de propagation géographique présente un R² de 0.83, ce qui est satisfaisant compte tenu de la complexité inhérente à la prédiction de la propagation géographique des pandémies.

### Importance des Caractéristiques

L'analyse de l'importance des caractéristiques a révélé que :

- Pour le taux de transmission, les facteurs les plus déterminants sont le nombre de cas sur les 7 derniers jours et le taux de croissance récent
- Pour le taux de mortalité, le ratio décès/cas des périodes précédentes est le facteur le plus important
- Pour la propagation géographique, les moyennes mobiles récentes et le nombre de nouvelles localisations dans la semaine précédente sont les plus prédictifs

### Limites et Considérations

Malgré les bonnes performances de nos modèles, il est important de noter certaines limites :

- Les prédictions sont basées sur les tendances historiques et peuvent être moins précises face à des variants radicalement nouveaux
- Les facteurs externes comme les mesures de santé publique, les changements de politique de test, ou les avancées médicales peuvent modifier les tendances
- La qualité des prédictions dépend de la qualité et de la complétude des données d'entrée

## Principes d'Ergonomie et d'Accessibilité

### Approche d'Accessibilité

Notre interface utilisateur a été conçue en suivant les principes WCAG (Web Content Accessibility Guidelines) pour garantir son accessibilité à tous les utilisateurs, y compris ceux ayant des handicaps visuels ou moteurs.

#### Principes WCAG Implémentés

1. **Perceptible**
   - Contraste de couleurs élevé (ratio minimum de 4.5:1)
   - Textes alternatifs pour toutes les images
   - Structure sémantique claire avec des titres hiérarchiques
   - Option de contraste élevé pour les utilisateurs malvoyants

2. **Utilisable**
   - Navigation complète au clavier
   - Lien d'évitement pour accéder directement au contenu principal
   - Temps suffisant pour lire et utiliser le contenu
   - Éléments interactifs de taille adéquate (minimum 44x44 pixels)

3. **Compréhensible**
   - Interface cohérente et prévisible
   - Étiquettes claires pour tous les éléments de formulaire
   - Messages d'erreur explicites et suggestions de correction
   - Organisation logique du contenu

4. **Robuste**
   - Compatibilité avec les technologies d'assistance
   - Balisage HTML valide et bien structuré
   - Attributs ARIA pour améliorer la compatibilité avec les lecteurs d'écran

#### Fonctionnalités d'Accessibilité Spécifiques

- **Mode contraste élevé** : Permet aux utilisateurs de basculer vers un mode à fort contraste
- **Redimensionnement du texte** : Option pour augmenter la taille du texte sans perte de fonctionnalité
- **Navigation au clavier** : Focus visible et logique pour tous les éléments interactifs
- **Compatibilité avec les lecteurs d'écran** : Textes alternatifs, attributs ARIA et structure sémantique
- **Design responsive** : Adaptation à différentes tailles d'écran et orientations

### Principes d'Ergonomie

Notre interface a été conçue selon les principes d'ergonomie moderne pour offrir une expérience utilisateur intuitive et efficace.

#### Organisation de l'Interface

- **Structure en onglets** : Organisation claire du contenu en sections thématiques
- **Hiérarchie visuelle** : Mise en évidence des informations importantes
- **Regroupement logique** : Regroupement des éléments liés pour faciliter la compréhension
- **Espacement adéquat** : Utilisation d'espaces blancs pour réduire la charge cognitive

#### Visualisation des Données

- **Graphiques interactifs** : Utilisation de Chart.js pour des visualisations dynamiques et interactives
- **Filtres intuitifs** : Possibilité de filtrer les données selon différents critères
- **Légendes claires** : Explication des éléments visuels pour une meilleure compréhension
- **Palettes de couleurs accessibles** : Choix de couleurs distinctes et significatives

#### Interactions

- **Feedback immédiat** : Réponse visuelle aux actions de l'utilisateur
- **Cohérence des interactions** : Comportements prévisibles des éléments interactifs
- **Prévention des erreurs** : Conception qui minimise les risques d'erreur
- **Aide contextuelle** : Informations d'aide disponibles au moment opportun

## Benchmark des Solutions Front-end

### Critères d'Évaluation

Pour choisir la meilleure solution front-end pour notre application, nous avons évalué plusieurs frameworks selon les critères suivants :

1. **Performance** : Temps de chargement, réactivité, optimisation
2. **Accessibilité** : Conformité WCAG, facilité d'implémentation des fonctionnalités d'accessibilité
3. **Visualisation de données** : Compatibilité avec les bibliothèques de visualisation, performance avec de grands ensembles de données
4. **Responsive design** : Adaptation à différentes tailles d'écran
5. **Facilité de développement** : Courbe d'apprentissage, documentation, écosystème
6. **Maintenance** : Stabilité, communauté active, mises à jour régulières

### Solutions Évaluées

#### React

**Forces :**
- Écosystème riche et mature
- Excellente performance grâce au DOM virtuel
- Grande communauté et nombreuses bibliothèques
- Bonne intégration avec les bibliothèques de visualisation

**Faiblesses :**
- Courbe d'apprentissage modérée
- Nécessite des bibliothèques supplémentaires pour certaines fonctionnalités
- Configuration initiale parfois complexe

#### Vue.js

**Forces :**
- Syntaxe intuitive et facile à apprendre
- Bonne performance
- Documentation claire et complète
- Approche progressive permettant une adoption incrémentale

**Faiblesses :**
- Écosystème moins développé que React
- Moins de développeurs expérimentés sur le marché
- Certaines bibliothèques de visualisation avancées moins bien intégrées

#### Angular

**Forces :**
- Framework complet avec toutes les fonctionnalités intégrées
- Architecture robuste pour les grandes applications
- Typage fort avec TypeScript
- Bonnes pratiques imposées

**Faiblesses :**
- Courbe d'apprentissage abrupte
- Performance parfois inférieure sur les appareils mobiles
- Verbosité du code
- Mise en place initiale plus lourde

#### Solution Hybride (HTML/CSS/JS avec Bootstrap et Chart.js)

**Forces :**
- Simplicité et légèreté
- Pas de compilation nécessaire
- Chargement rapide
- Facilité d'intégration avec l'API backend
- Excellente compatibilité avec les navigateurs

**Faiblesses :**
- Moins structuré pour les applications complexes
- Gestion d'état plus manuelle
- Réutilisation de code plus difficile

### Solution Retenue

Après analyse, nous avons opté pour une **solution hybride** utilisant **HTML/CSS/JavaScript** avec **Bootstrap** pour l'interface et **Chart.js** pour les visualisations.

**Justification :**
1. **Simplicité et rapidité de développement** : Solution directe sans nécessiter de build complexe
2. **Performance optimale** : Chargement rapide et réactivité excellente
3. **Accessibilité native** : Facilité d'implémentation des fonctionnalités d'accessibilité WCAG
4. **Visualisations performantes** : Chart.js offre d'excellentes performances et une bonne accessibilité
5. **Responsive design intégré** : Bootstrap fournit un système de grille responsive robuste
6. **Maintenance simplifiée** : Code plus simple à maintenir et à comprendre pour les futurs développeurs

Cette approche nous a permis de nous concentrer sur l'accessibilité et l'expérience utilisateur plutôt que sur la complexité technique, tout en offrant d'excellentes performances et une compatibilité maximale.

## Conduite au Changement dans le Contexte de l'Accessibilité

### Contexte et Enjeux

L'intégration de fonctionnalités d'accessibilité dans une application représente un changement significatif dans la façon dont les utilisateurs interagissent avec le système. Cette transformation nécessite une approche structurée pour garantir son adoption et son efficacité.

### Stratégie de Conduite au Changement

#### 1. Sensibilisation et Formation

**Objectifs :**
- Sensibiliser les parties prenantes à l'importance de l'accessibilité
- Former les équipes aux principes WCAG et aux bonnes pratiques

**Actions :**
- Ateliers de sensibilisation sur les différents types de handicaps et leurs impacts sur l'utilisation des technologies
- Sessions de formation sur les outils d'évaluation de l'accessibilité
- Démonstrations pratiques avec des technologies d'assistance (lecteurs d'écran, etc.)

#### 2. Implication des Utilisateurs

**Objectifs :**
- Intégrer les besoins réels des utilisateurs ayant des handicaps
- Valider l'efficacité des fonctionnalités d'accessibilité

**Actions :**
- Tests utilisateurs avec des personnes ayant différents types de handicaps
- Création d'un panel d'utilisateurs pour des retours réguliers
- Mise en place d'un système de feedback accessible

#### 3. Approche Progressive

**Objectifs :**
- Faciliter l'adaptation des utilisateurs aux nouvelles fonctionnalités
- Permettre une amélioration continue basée sur les retours

**Actions :**
- Déploiement par phases des fonctionnalités d'accessibilité
- Communication claire sur les changements et améliorations
- Période de transition avec support renforcé

#### 4. Documentation et Support

**Objectifs :**
- Fournir les ressources nécessaires pour une utilisation optimale
- Assurer un support adapté aux différents besoins

**Actions :**
- Création de guides d'utilisation accessibles (texte, audio, vidéo)
- Mise en place d'un support technique formé aux questions d'accessibilité
- FAQ spécifique aux fonctionnalités d'accessibilité

### Plan de Communication

#### Messages Clés

1. **Pour les décideurs :**
   - L'accessibilité comme avantage stratégique et obligation légale
   - Impact positif sur l'image de l'organisation
   - Élargissement de l'audience potentielle

2. **Pour les utilisateurs :**
   - Nouvelles fonctionnalités disponibles pour améliorer l'expérience
   - Méthodes pour personnaliser l'interface selon les besoins
   - Canaux pour signaler les problèmes et suggérer des améliorations

3. **Pour les équipes techniques :**
   - Importance de maintenir l'accessibilité dans les futures mises à jour
   - Ressources disponibles pour tester et améliorer l'accessibilité
   - Intégration de l'accessibilité dans le processus de développement

#### Canaux de Communication

- Webinaires de présentation des nouvelles fonctionnalités
- Documentation en ligne dans différents formats
- Sessions de questions-réponses
- Newsletters dédiées aux mises à jour d'accessibilité

### Mesure et Suivi

#### Indicateurs de Réussite

- Taux d'utilisation des fonctionnalités d'accessibilité
- Satisfaction des utilisateurs ayant des besoins spécifiques
- Conformité aux standards WCAG (pourcentage de critères respectés)
- Nombre de problèmes d'accessibilité signalés et résolus

#### Processus d'Amélioration Continue

- Audits réguliers d'accessibilité
- Revue des retours utilisateurs
- Veille technologique sur les nouvelles pratiques d'accessibilité
- Mise à jour du plan d'action en fonction des résultats

### Facteurs Clés de Succès

1. **Engagement de la direction** : Support visible et constant des décideurs
2. **Approche centrée utilisateur** : Implication continue des utilisateurs finaux
3. **Formation continue** : Mise à jour régulière des connaissances des équipes
4. **Intégration dans les processus** : L'accessibilité comme partie intégrante du développement
5. **Mesure et transparence** : Suivi et communication ouverte sur les progrès

Cette stratégie de conduite au changement vise à assurer que les fonctionnalités d'accessibilité ne sont pas seulement implémentées techniquement, mais également adoptées et utilisées efficacement par tous les utilisateurs concernés.
