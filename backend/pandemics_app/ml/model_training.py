import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

# Assurer que les répertoires existent
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def get_engine():
    POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "guigui")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "pandemies")
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        connect_args={"options": "-c search_path=pandemics"}
    )


def save_model_metrics_to_db(model_name, model_type, metrics):
    """Sauvegarde les métriques dans la base de données Django - VERSION CORRIGÉE"""
    try:
        import django
        import sys
        
        # 🔧 CORRECTION 1: Ajouter le répertoire racine Django au PYTHONPATH
        django_root = '/django_api'
        if django_root not in sys.path:
            sys.path.insert(0, django_root)
            print(f"✅ Ajout de {django_root} au PYTHONPATH")
        
        # 🔧 CORRECTION 2: Utiliser le bon nom de settings module
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pandemics_project.settings')
        django.setup()
        print("✅ Django setup réussi")
        
        # 🔧 CORRECTION 3: Utiliser le bon nom de l'app
        from pandemics_app.models import ModelMetrics
        
        # Récupérer les métriques pour ce modèle
        model_metrics = metrics.get(model_name, {})
        
        # Créer ou mettre à jour l'entrée
        metric_entry, created = ModelMetrics.objects.update_or_create(
            model_type=model_type,
            model_name=model_name,
            defaults={
                'mse': float(model_metrics.get('mse', 0)),
                'rmse': float(model_metrics.get('rmse', 0)),
                'mae': float(model_metrics.get('mae', 0)),
                'r2_score': float(model_metrics.get('r2_score', 0)),
                'cv_rmse': float(model_metrics.get('cv_rmse', 0))
            }
        )
        
        action = "créée" if created else "mise à jour"
        print(f"✅ Métriques {action} dans la DB pour {model_type} - {model_name}")
        print(f"   📊 RMSE: {model_metrics.get('rmse', 0):.4f}, R²: {model_metrics.get('r2_score', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde des métriques en DB: {e}")
        import traceback
        traceback.print_exc()
        print("📝 Les métriques restent disponibles dans les fichiers joblib")
        return False


def load_prepared_data():
    """Charge les données de base et effectue des jointures"""
    engine = get_engine()
    query = text("""
                 SELECT w.*, l.name AS location, v.name AS virus
                 FROM worldmeter w
                          JOIN location l ON w.location_id = l.id
                          JOIN virus v ON w.virus_id = v.id
                 ORDER BY w.date ASC
                 """)
    df = pd.read_sql(query, engine)
    print("Colonnes dans le DataFrame :", df.columns.tolist())

    # Convertir date en datetime si ce n'est pas déjà fait
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    return df[df['virus'] == 'COVID'], df[df['virus'] == 'Monkeypox']


def prepare_train_test_data_temporal(df, target_col, feature_cols, test_size=0.2, random_state=42):
    """
    Prépare les données d'entraînement et de test avec une séparation temporelle stricte
    """
    # Tri chronologique pour validation temporelle
    df = df.sort_values('date')

    # Déterminer la date de séparation train/test
    split_date = df['date'].iloc[int(len(df) * (1 - test_size))]
    print(f"Date de séparation train/test: {split_date}")

    # Diviser en ensembles d'entraînement et de test
    train_data = df[df['date'] < split_date]
    test_data = df[df['date'] >= split_date]

    # Sélectionner les colonnes, gérer les valeurs manquantes et l'infini
    X_train = train_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_data[target_col]
    X_test = test_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_data[target_col]

    # Standardisation des features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Données préparées : {X_train.shape[0]} train, {X_test.shape[0]} test")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols


def save_model(model, scaler, feature_cols, model_name, model_type, metrics):
    """Sauvegarde le modèle, le scaler, les métadonnées ET les métriques en base de données"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    safe_name = model_name.replace(" ", "_")
    
    # Sauvegarde des fichiers (comme avant)
    joblib.dump(model, os.path.join(MODEL_DIR, f"{model_type}_{safe_name}_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, f"{model_type}_{safe_name}_scaler.joblib"))

    # Enrichir les métadonnées avec les métriques
    metadata_to_save = {
        'feature_cols': feature_cols,
        'model_type': model_type,
        'model_name': model_name,
    }
    if metrics and model_name in metrics:
        metadata_to_save.update(metrics[model_name])

    joblib.dump(metadata_to_save, os.path.join(MODEL_DIR, f"{model_type}_{safe_name}_metadata.joblib"))
    
    # 🔧 NOUVEAU : Sauvegarder aussi dans la base de données Django
    save_model_metrics_to_db(model_name, model_type, metrics)
    
    print(f"✅ Modèle {model_type} sauvegardé ({model_name}) avec métriques en fichier ET en base.")


def train_and_evaluate_models(X_train, X_test, y_train, y_test, model_type):
    """Entraîne et évalue plusieurs modèles"""
    print(f"\n=== Entraînement des modèles pour {model_type} ===")

    # Définition des modèles et des hyperparamètres à tester
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42)
    }

    param_grids = {
        'Ridge Regression': {'alpha': [0.1, 1.0, 10.0]},
        'Random Forest': {
            'n_estimators': [50, 100],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        },
        'Gradient Boosting': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5]
        },
        'XGBoost': {
            'n_estimators': [50, 100],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8],
            'colsample_bytree': [0.8]
        }
    }

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Évaluation de chaque modèle
    best_models_dict, model_metrics_dict = {}, {}
    for name, model in models.items():
        print(f"\n→ Entraînement de : {name}")

        if name in param_grids:
            print("  Lancement de GridSearchCV...")
            grid = GridSearchCV(
                model,
                param_grids[name],
                cv=tscv,  # Validation croisée temporelle
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=2
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            print(f"  Meilleurs paramètres : {grid.best_params_}")
        else:
            best_model = model.fit(X_train, y_train)

        # Prédictions et métriques
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Validation croisée temporelle
        cv_scores = cross_val_score(
            best_model,
            X_train, y_train,
            cv=tscv,
            scoring='neg_mean_squared_error'
        )
        cv_rmse_val = np.sqrt(-cv_scores.mean())

        print(f"  RMSE: {rmse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f} | CV RMSE: {cv_rmse_val:.4f}")

        best_models_dict[name] = best_model
        model_metrics_dict[name] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'cv_rmse': cv_rmse_val
        }

    # Sélection du meilleur modèle selon RMSE
    best_model_name = min(model_metrics_dict, key=lambda x: model_metrics_dict[x]['rmse'])
    print(
        f"\n✅ Meilleur modèle pour {model_type} : {best_model_name} (RMSE = {model_metrics_dict[best_model_name]['rmse']:.4f})")
    return best_models_dict, model_metrics_dict, best_model_name


def train_transmission_model():
    """
    Entraîne un modèle amélioré pour prédire le taux de reproduction effectif (Rt)
    """
    print("\n=== Modèle : Taux de transmission (Rt) ===")
    covid_df, monkeypox_df = load_prepared_data()

    # Traiter séparément chaque virus pour calculer Rt
    dfs = []
    for df, virus_name in [(covid_df, 'COVID'), (monkeypox_df, 'Monkeypox')]:
        print(f"Préparation des données pour {virus_name}...")

        # Grouper par location pour les calculs spécifiques à chaque région
        grouped = df.groupby('location')
        all_dfs = []

        for location, group_df in grouped:
            temp_df = group_df.copy().sort_values('date')

            # Calcul des moyennes mobiles sur 7 jours (plus stables)
            temp_df['cases_ma7'] = temp_df['new_cases'].rolling(window=7, min_periods=1).mean()
            temp_df['deaths_ma7'] = temp_df['new_deaths'].rolling(window=7, min_periods=1).mean()

            # Décalage pour calculer Rt (taux de reproduction)
            incubation_period = 7  # Estimation moyenne
            temp_df['previous_cases_ma7'] = temp_df['cases_ma7'].shift(incubation_period)

            # Calcul de Rt (limité pour éviter des valeurs extrêmes)
            temp_df['rt'] = (temp_df['cases_ma7'] / temp_df['previous_cases_ma7'].replace(0, 0.1)).clip(0, 10)

            # Features supplémentaires
            temp_df['cases_growth'] = temp_df['cases_ma7'].pct_change(7).fillna(0).clip(-1, 2)
            temp_df['deaths_growth'] = temp_df['deaths_ma7'].pct_change(7).fillna(0).clip(-1, 2)
            temp_df['days_since_start'] = (temp_df['date'] - temp_df['date'].min()).dt.days
            temp_df['cases_acceleration'] = temp_df['cases_growth'].diff().fillna(0)

            # Features cycliques
            temp_df['day_sin'] = np.sin(2 * np.pi * temp_df['date'].dt.dayofweek / 7)
            temp_df['day_cos'] = np.cos(2 * np.pi * temp_df['date'].dt.dayofweek / 7)
            temp_df['month_sin'] = np.sin(2 * np.pi * temp_df['date'].dt.month / 12)
            temp_df['month_cos'] = np.cos(2 * np.pi * temp_df['date'].dt.month / 12)

            all_dfs.append(temp_df)

        # Combiner tous les dataframes
        combined_df = pd.concat(all_dfs)
        dfs.append(combined_df)

    # Combiner les données des deux virus
    df_final = pd.concat(dfs)

    # Nettoyage final
    df_final = df_final.dropna(subset=['rt'])
    df_final = df_final.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Sauvegarde des données préparées
    rt_df = df_final[['date', 'location', 'virus', 'rt', 'cases_ma7', 'deaths_ma7']]
    rt_df.to_csv(os.path.join(DATA_DIR, 'transmission_rt_data.csv'), index=False)
    print(f"✅ Données Rt sauvegardées: {os.path.join(DATA_DIR, 'transmission_rt_data.csv')}")

    # Définir les features et la cible
    target_col = 'rt'
    feature_cols = [
        'cases_ma7', 'deaths_ma7', 'previous_cases_ma7',
        'cases_growth', 'deaths_growth', 'cases_acceleration',
        'days_since_start', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
        'total_cases', 'total_deaths',
        'new_cases_per_million', 'total_cases_per_million',
        'new_deaths_per_million', 'total_deaths_per_million'
    ]

    # Préparation des données avec validation temporelle
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_data_temporal(
        df_final, target_col, feature_cols
    )

    # Entraînement et évaluation des modèles
    models, metrics, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, 'transmission'
    )

    # Sauvegarde du meilleur modèle
    save_model(models[best_name], scaler, feature_cols, best_name, 'transmission', metrics)

    return metrics


def train_mortality_model():
    """
    Entraîne un modèle amélioré pour prédire le taux de mortalité
    """
    print("\n=== Modèle : Taux de mortalité ===")
    covid_df, monkeypox_df = load_prepared_data()

    # Traiter séparément chaque virus
    dfs = []
    for df, virus_name in [(covid_df, 'COVID'), (monkeypox_df, 'Monkeypox')]:
        print(f"Préparation des données pour {virus_name}...")

        # Grouper par location
        grouped = df.groupby('location')
        all_dfs = []

        for location, group_df in grouped:
            temp_df = group_df.copy().sort_values('date')

            # Remplacement des valeurs nulles
            temp_df['new_cases'] = temp_df['new_cases'].fillna(0)
            temp_df['new_deaths'] = temp_df['new_deaths'].fillna(0)

            # Calcul des sommes glissantes
            temp_df['cases_7day'] = temp_df['new_cases'].rolling(7).sum().fillna(0)
            temp_df['deaths_7day'] = temp_df['new_deaths'].rolling(7).sum().fillna(0)

            # Décalages pour tenir compte du délai entre infection et décès
            lag_days = 14  # Délai moyen entre cas et décès
            temp_df['cases_7day_lag'] = temp_df['cases_7day'].shift(lag_days)

            # Calcul du taux de mortalité (CFR - Case Fatality Rate)
            # Limiter entre 0 et 1 (0-100%)
            temp_df['mortality_ratio'] = (temp_df['deaths_7day'] / temp_df['cases_7day_lag'].replace(0, 1)).clip(0, 1)

            # Features supplémentaires
            temp_df['days_since_start'] = (temp_df['date'] - temp_df['date'].min()).dt.days
            temp_df['treatment_improvement'] = np.exp(
                -0.001 * temp_df['days_since_start'])  # Proxy pour l'amélioration des traitements
            temp_df['healthcare_pressure'] = (temp_df['cases_7day'] / temp_df['cases_7day'].max()).fillna(
                0)  # Proxy pour la pression sur le système de santé

            all_dfs.append(temp_df)

        combined_df = pd.concat(all_dfs)
        dfs.append(combined_df)

    # Combiner les données des deux virus
    df_final = pd.concat(dfs)

    # Nettoyage final
    df_final = df_final.dropna(subset=['mortality_ratio'])

    # Sauvegarde des données préparées
    mortality_df = df_final[['date', 'location', 'virus', 'mortality_ratio', 'cases_7day', 'deaths_7day']]
    mortality_df.to_csv(os.path.join(DATA_DIR, 'mortality_ratio_data.csv'), index=False)
    print(f"✅ Données de mortalité sauvegardées: {os.path.join(DATA_DIR, 'mortality_ratio_data.csv')}")

    # Définir les features et la cible
    target_col = 'mortality_ratio'
    feature_cols = [
        'cases_7day', 'deaths_7day', 'cases_7day_lag',
        'days_since_start', 'treatment_improvement', 'healthcare_pressure',
        'total_cases', 'total_deaths', 'new_cases', 'new_deaths',
        'new_cases_per_million', 'total_cases_per_million',
        'new_deaths_per_million', 'total_deaths_per_million'
    ]

    # Préparation des données avec validation temporelle
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_data_temporal(
        df_final, target_col, feature_cols
    )

    # Entraînement et évaluation des modèles
    models, metrics, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, 'mortality'
    )

    # Sauvegarde du meilleur modèle
    save_model(models[best_name], scaler, feature_cols, best_name, 'mortality', metrics)

    return metrics


def train_geographical_spread_model():
    """
    Entraîne un modèle amélioré pour prédire la propagation géographique
    """
    print("\n=== Modèle : Propagation géographique ===")
    engine = get_engine()

    # 1. Récupérer les premières dates de cas positifs par localisation et virus
    first_cases_query = text("""
                             SELECT l.name      AS location,
                                    v.name      AS virus,
                                    MIN(w.date) AS first_date
                             FROM worldmeter w
                                      JOIN location l ON w.location_id = l.id
                                      JOIN virus v ON w.virus_id = v.id
                             WHERE w.new_cases > 0
                             GROUP BY l.name, v.name
                             ORDER BY first_date
                             """)

    first_cases = pd.read_sql(first_cases_query, engine)

    # Convertir en datetime
    first_cases['first_date'] = pd.to_datetime(first_cases['first_date'])

    # 2. Ajouter des composantes temporelles
    first_cases['year'] = first_cases['first_date'].dt.year
    first_cases['week'] = first_cases['first_date'].dt.isocalendar().week
    first_cases['yearweek'] = first_cases['year'].astype(str) + '-' + first_cases['week'].astype(str).str.zfill(2)

    # 3. Compter par semaine le nombre de nouvelles localisations touchées
    spread_data = first_cases.groupby(['virus', 'yearweek']).size().reset_index(name='new_locations')

    # Convertir yearweek en date (milieu de la semaine) pour le tri temporel
    def yearweek_to_date(yearweek):
        year, week = yearweek.split('-')
        # Créer une date à partir de l'année et de la semaine (jour 3 = mercredi)
        return pd.to_datetime(f"{year}-W{week}-3", format="%Y-W%W-%w")

    spread_data['date'] = spread_data['yearweek'].apply(yearweek_to_date)
    spread_data = spread_data.sort_values(['virus', 'date'])

    # 4. Ajouter des features plus robustes - EXPLICITEMENT pour chaque virus
    final_dfs = []

    for virus in spread_data['virus'].unique():
        print(f"Traitement des données pour le virus: {virus}")
        virus_data = spread_data[spread_data['virus'] == virus].copy()

        # Créer des lags (valeurs précédentes)
        for i in range(1, 5):
            virus_data[f'lag{i}'] = virus_data['new_locations'].shift(i)

        # Moyennes mobiles
        virus_data['ma2'] = virus_data['new_locations'].rolling(2).mean()
        virus_data['ma3'] = virus_data['new_locations'].rolling(3).mean()

        # Tendance (différence entre moyennes mobiles)
        virus_data['trend'] = virus_data['ma2'] - virus_data['ma3'].shift(1)

        # Features saisonnières
        virus_data['month'] = virus_data['date'].dt.month
        virus_data['month_sin'] = np.sin(2 * np.pi * virus_data['month'] / 12)
        virus_data['month_cos'] = np.cos(2 * np.pi * virus_data['month'] / 12)

        # Ajouter ce virus au résultat final
        final_dfs.append(virus_data)

    # Combiner tous les DataFrames
    final_spread_data = pd.concat(final_dfs)

    # Vérification des colonnes
    print("Colonnes dans le DataFrame final:", final_spread_data.columns.tolist())

    # 5. Nettoyer et préparer les données finales
    final_spread_data = final_spread_data.dropna()

    # Sauvegarde des données préparées
    final_spread_data.to_csv(os.path.join(DATA_DIR, 'geographical_spread_data.csv'), index=False)
    print(
        f"✅ Données de propagation géographique sauvegardées: {os.path.join(DATA_DIR, 'geographical_spread_data.csv')}")

    # 6. Définir les features et la cible
    target_col = 'new_locations'
    feature_cols = [
        'lag1', 'lag2', 'lag3', 'lag4',
        'ma2', 'ma3', 'trend',
        'month_sin', 'month_cos'
    ]

    # 7. Séparer en train/test avec validation temporelle
    X_train, X_test, y_train, y_test, scaler, feature_cols = prepare_train_test_data_temporal(
        final_spread_data, target_col, feature_cols
    )

    # 8. Entraînement et évaluation des modèles
    models, metrics, best_name = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, 'geographical_spread'
    )

    # 9. Sauvegarde du meilleur modèle
    save_model(models[best_name], scaler, feature_cols, best_name, 'geographical_spread', metrics)

    return metrics


def train_models():
    """Fonction principale pour entraîner tous les modèles"""
    print("=== Début de l'entraînement des modèles ===")

    # Création des répertoires nécessaires
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Entraînement de chaque modèle avec sauvegarde automatique en base de données
    try:
        print("\n🚀 Entraînement du modèle de transmission...")
        transmission_metrics = train_transmission_model()
        print("✅ Modèle de transmission terminé")
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du modèle de transmission: {e}")
        transmission_metrics = None

    try:
        print("\n🚀 Entraînement du modèle de mortalité...")
        mortality_metrics = train_mortality_model()
        print("✅ Modèle de mortalité terminé")
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du modèle de mortalité: {e}")
        mortality_metrics = None

    try:
        print("\n🚀 Entraînement du modèle de propagation géographique...")
        geographical_metrics = train_geographical_spread_model()
        print("✅ Modèle de propagation géographique terminé")
    except Exception as e:
        print(f"❌ Erreur lors de l'entraînement du modèle de propagation géographique: {e}")
        geographical_metrics = None

    print("\n🎉 Entraînement terminé avec sauvegarde automatique des métriques en base de données.")
    
    # Résumé final
    print("\n📊 RÉSUMÉ:")
    if transmission_metrics:
        print(f"   ✅ Transmission: {len(transmission_metrics)} modèles entraînés")
    if mortality_metrics:
        print(f"   ✅ Mortalité: {len(mortality_metrics)} modèles entraînés")
    if geographical_metrics:
        print(f"   ✅ Propagation: {len(geographical_metrics)} modèles entraînés")
    
    return transmission_metrics, mortality_metrics, geographical_metrics


if __name__ == "__main__":
    train_models()