# Documentation Technique

## Pipeline CI/CD

## Configuration Docker

## Diagrammes UML



### Pipeline CI/CD

Le pipeline CI/CD est implémenté à l'aide de GitHub Actions. Le workflow est défini dans le fichier `.github/workflows/main.yml`.

Ce workflow est déclenché à chaque push sur la branche `main` et à chaque pull request ciblant la branche `main`.

Les étapes principales du pipeline sont les suivantes :

1.  **Checkout du code** : Récupère le code source du dépôt.
2.  **Configuration de l'environnement** : Met en place l'environnement Python et Node.js nécessaire.
3.  **Installation des dépendances** : Installe les dépendances du backend (Django) et du frontend (React).
4.  **Tests unitaires et d'intégration** : Exécute les tests définis pour le backend et le frontend.
5.  **Analyse de la qualité du code** : Utilise des outils comme Flake8 pour Python et ESLint pour JavaScript pour analyser la qualité du code.
6.  **Construction des images Docker** : Construit les images Docker pour le backend et les autres services définis dans `docker-compose.yaml`.
7.  **Déploiement (manuel ou automatique)** : Une fois les étapes précédentes réussies, le déploiement peut être déclenché manuellement ou automatiquement vers les environnements de staging ou de production, en fonction des profils Docker Compose (`us`, `fr`, `ch`).




### Configuration Docker

Le projet utilise Docker et Docker Compose pour la conteneurisation des services. Les fichiers de configuration principaux sont :

-   `Dockerfile` (à la racine) : Utilisé pour construire l'image de base d'Airflow, incluant les dépendances nécessaires.
-   `backend/Dockerfile` : Utilisé pour construire l'image du service Django (backend).
-   `docker-compose.yaml` : Définit les services de base pour l'environnement de développement (Django, PostgreSQL, Airflow Webserver, Airflow Scheduler).
-   `docker-compose.prod.yaml` : Définit les services pour les déploiements spécifiques aux pays (US, FR, CH), en étendant les services de `docker-compose.yaml` et en appliquant des profils.
-   `common-services.yaml` : Contient des définitions de services communes, comme les healthchecks, qui peuvent être étendues par d'autres fichiers Docker Compose.
-   `.env` : Fichier pour les variables d'environnement sensibles (clés secrètes Django et Airflow, paramètres de base de données, pays de déploiement).

**Services définis :**

-   **django** : Le backend de l'application, basé sur Django REST Framework. Il utilise Gunicorn pour servir l'application en production.
-   **postgres** : La base de données PostgreSQL pour Django et Airflow.
-   **airflow-webserver** : L'interface utilisateur d'Apache Airflow.
-   **airflow-scheduler** : Le planificateur de tâches d'Apache Airflow.
-   **metabase** : Un outil de Business Intelligence (BI) pour l'analyse des données.

**Adaptations par pays :**

Le projet est configuré pour des déploiements spécifiques à différents pays (US, France, Suisse) via des profils Docker Compose et des variables d'environnement :

-   **US (États-Unis)** : Déploiement complet de l'API technique (Django) et de Metabase.
-   **FR (France)** : L'API technique (Django backend) n'est pas déployée. La localisation du frontend est adaptée.
-   **CH (Suisse)** : Metabase et l'API technique ne sont pas déployés. Le frontend est multilingue.

Ces adaptations sont gérées par le fichier `docker-compose.prod.yaml` qui utilise des profils (`us`, `fr`, `ch`) pour activer ou désactiver des services et des variables d'environnement (`DEPLOY_COUNTRY`) pour modifier le comportement de l'application Django.




### Diagrammes UML

Des diagrammes UML (Unified Modeling Language) peuvent être utilisés pour visualiser l'architecture et le comportement du système. Les diagrammes pertinents pour ce projet incluent :

-   **Diagramme de déploiement** : Illustre l'architecture physique du système, montrant où les différents composants logiciels sont déployés sur le matériel.
-   **Diagramme de composants** : Décrit la structure logique du système en montrant les composants logiciels et leurs dépendances.

(Les diagrammes UML seront ajoutés ici.)





### Outils de gestion de projet Agile

Pour la gestion de projet Agile, les outils suivants peuvent être mis en place :

-   **Jira/Trello/Azure DevOps** : Pour la gestion des backlogs, des sprints, des tâches et le suivi de l'avancement.
-   **Confluence/Wiki** : Pour la documentation du projet, les spécifications, les décisions techniques et les comptes-rendus de réunions.
-   **Git (GitHub/GitLab/Bitbucket)** : Pour le contrôle de version du code source, avec des fonctionnalités de gestion des pull requests et de revue de code.

Pour ce projet, nous avons déjà configuré un workflow CI/CD avec GitHub Actions, ce qui s'intègre bien avec GitHub pour la gestion du code. Pour le suivi des tâches, un simple fichier `todo.md` a été utilisé, mais pour un projet plus grand, un outil dédié serait préférable.


