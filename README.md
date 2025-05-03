# Projet de Prédiction de Pandémies - OMS

Ce projet a été développé pour l'Organisation Mondiale de la Santé (OMS) afin de fournir des prédictions précises sur l'évolution des pandémies, en utilisant des techniques avancées d'intelligence artificielle et d'apprentissage automatique.

## Table des matières

- [Aperçu du projet](#aperçu-du-projet)
- [Fonctionnalités](#fonctionnalités)
- [Architecture du système](#architecture-du-système)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Utilisation](#utilisation)
  - [Lancement de l'application](#lancement-de-lapplication)
  - [Entraînement des modèles](#entraînement-des-modèles)
  - [Utilisation de l'API](#utilisation-de-lapi)
  - [Interface utilisateur](#interface-utilisateur)
- [Documentation](#documentation)
- [Licence](#licence)

## Aperçu du projet

Le projet de Prédiction de Pandémies est une application complète qui permet de :

- Analyser les données historiques des pandémies (COVID-19, Monkeypox)
- Prédire les taux de transmission et de mortalité futurs
- Anticiper la propagation géographique des pandémies
- Visualiser les données et prédictions via une interface utilisateur accessible

Le système utilise des modèles d'apprentissage automatique (Random Forest, Gradient Boosting) pour générer des prédictions précises basées sur les tendances historiques.

## Fonctionnalités

- **Prédiction du taux de transmission** : Estimation de la vitesse de propagation de la maladie
- **Prédiction du taux de mortalité** : Estimation du pourcentage de cas qui aboutiront à un décès
- **Prédiction de la propagation géographique** : Estimation du nombre de nouvelles localisations qui seront touchées
- **Génération de prévisions** : Projections sur plusieurs semaines pour anticiper l'évolution des pandémies
- **Visualisation interactive** : Graphiques et tableaux de bord pour explorer les données et les prédictions
- **Interface accessible** : Conformité WCAG avec fonctionnalités d'accessibilité (contraste élevé, texte agrandi, navigation au clavier)
- **API RESTful** : Endpoints pour intégrer les prédictions dans d'autres systèmes

## Architecture du système

Le projet est organisé selon une architecture moderne et modulaire :

```
pandemics_project/
├── backend/                # Backend Django avec API REST
│   ├── pandemics_app/      # Application principale
│   │   ├── ml/             # Modèles d'IA et scripts d'entraînement
│   │   ├── migrations/     # Migrations de base de données
│   │   ├── models.py       # Modèles de données
│   │   ├── serializers.py  # Sérialiseurs pour l'API
│   │   ├── views.py        # Vues et endpoints API
│   │   ├── urls.py         # Configuration des URLs
│   │   └── tests.py        # Tests unitaires et d'intégration
│   └── backend/            # Configuration du projet Django
├── frontend/               # Interface utilisateur
│   ├── index.html          # Page principale
│   └── accessibility_tests.py # Tests d'accessibilité
├── data/                   # Données pour l'entraînement et les tests
│   ├── covid.csv           # Données COVID-19
│   └── monkeypox.csv       # Données Monkeypox
├── dags/                   # DAGs Airflow pour l'ETL
│   └── airflow_etl.py      # Pipeline ETL automatisé
├── docker-compose.yaml     # Configuration Docker Compose
└── documentation_technique.md # Documentation technique détaillée
```

## Prérequis

- Docker et Docker Compose
- Python 3.8 ou supérieur (pour le développement local)
- Navigateur web moderne (Chrome, Firefox, Safari, Edge)

## Installation

1. Clonez le dépôt :
   ```bash
   git clone <url-du-depot>
   cd pandemics_project
   ```

2. Lancez l'application avec Docker Compose :
   ```bash
   docker-compose up -d
   ```

   Cette commande va :
   - Construire les images Docker nécessaires
   - Créer et démarrer les conteneurs pour le backend, la base de données et Airflow
   - Configurer les volumes et les réseaux

3. Attendez que tous les services soient opérationnels (cela peut prendre quelques minutes lors du premier lancement).

## Utilisation

### Lancement de l'application

Une fois les conteneurs démarrés, vous pouvez accéder à :

- **Interface utilisateur** : http://localhost:8080
- **API REST** : http://localhost:8000/api/
- **Interface Airflow** : http://localhost:8081 (utilisateur : airflow, mot de passe : airflow)

### Entraînement des modèles

Pour entraîner ou ré-entraîner les modèles d'IA avec de nouvelles données :

1. Assurez-vous que les données sont au format approprié dans le dossier `data/`

2. Exécutez le script d'entraînement :
   ```bash
   docker-compose exec backend python manage.py shell -c "from pandemics_app.ml.model_training import train_models; train_models()"
   ```

   Ou, si vous travaillez en local sans Docker :
   ```bash
   cd backend
   python manage.py shell -c "from pandemics_app.ml.model_training import train_models; train_models()"
   ```

3. Les modèles entraînés seront sauvegardés dans le dossier `backend/pandemics_app/ml/models/`

### Utilisation de l'API

L'API REST offre plusieurs endpoints pour interagir avec le système :

#### Endpoints principaux

- `GET /api/latest-data/` : Récupère les dernières données pour toutes les localisations et virus
- `POST /api/predict/transmission/` : Prédit le taux de transmission
- `POST /api/predict/mortality/` : Prédit le taux de mortalité
- `POST /api/predict/geographical-spread/` : Prédit la propagation géographique
- `POST /api/predict/combined/` : Génère des prédictions combinées
- `POST /api/predict/forecast/` : Génère des prévisions sur plusieurs semaines
- `GET /api/model-metrics-summary/` : Récupère les métriques de performance des modèles

#### Exemple de requête pour la prédiction du taux de transmission

```bash
curl -X POST http://localhost:8000/api/predict/transmission/ \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": 1,
    "virus_id": 1,
    "cases_7day_lag7": 1000,
    "cases_7day_lag14": 800,
    "deaths_7day_lag7": 20,
    "deaths_7day_lag14": 15,
    "cases_growth": 0.25,
    "deaths_growth": 0.33
  }'
```

### Interface utilisateur

L'interface utilisateur offre plusieurs fonctionnalités :

1. **Tableau de bord** : Vue d'ensemble des données actuelles avec filtres par virus, localisation et période

2. **Prédictions IA** : Visualisation des prédictions pour les 4 prochaines semaines
   - Taux de transmission et mortalité par localisation et virus
   - Propagation géographique par virus
   - Performance des modèles d'IA

3. **Explorateur de données** : Outil de visualisation personnalisée pour explorer les données historiques

4. **Fonctionnalités d'accessibilité** :
   - Mode contraste élevé (bouton avec icône de demi-cercle)
   - Texte agrandi (bouton avec icône de police)
   - Navigation au clavier (utilisez Tab pour naviguer)
   - Lien d'évitement (appuyez sur Tab dès le chargement de la page)

## Documentation

Pour une documentation plus détaillée, consultez les ressources suivantes :

- [Documentation technique](documentation_technique.md) : Détails sur les algorithmes d'IA, les métriques de performance, et les principes d'accessibilité
- [API Documentation](http://localhost:8000/api/docs/) : Documentation interactive de l'API (disponible lorsque le serveur est en cours d'exécution)

## Licence

Ce projet est développé pour l'Organisation Mondiale de la Santé (OMS) et est soumis aux conditions de licence spécifiées par l'OMS.
