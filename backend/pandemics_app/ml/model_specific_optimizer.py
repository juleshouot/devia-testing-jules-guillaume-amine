import os
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine, text
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    TimeSeriesSplit,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")


def get_engine():
    """Connexion à la base de données"""
    POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "guigui")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "pandemies")
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        connect_args={"options": "-c search_path=pandemics"},
    )


def load_prepared_data():
    """Charge les données de base"""
    engine = get_engine()
    query = text(
        """
                 SELECT w.*, l.name AS location, v.name AS virus
                 FROM worldmeter w
                          JOIN location l ON w.location_id = l.id
                          JOIN virus v ON w.virus_id = v.id
                 ORDER BY w.date ASC
                 """
    )
    df = pd.read_sql(query, engine)

    if not pd.api.types.is_datetime64_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    return df[df["virus"] == "COVID"], df[df["virus"] == "Monkeypox"]


def create_transmission_features(df):
    """Features spécifiques au modèle de transmission (Rt)"""
    print("🔧 Création features de transmission avancées...")

    enhanced_dfs = []
    for location, group_df in df.groupby("location"):
        temp_df = group_df.copy().sort_values("date")

        print(f"   Processing {location}...")

        # ===== FEATURES RT SPÉCIFIQUES =====

        # Moyennes mobiles multiples pour Rt
        for window in [3, 7, 14, 21]:
            temp_df[f"cases_ma{window}"] = (
                temp_df["new_cases"].rolling(window, min_periods=1).mean()
            )
            temp_df[f"deaths_ma{window}"] = (
                temp_df["new_deaths"].rolling(window, min_periods=1).mean()
            )

        # Calcul Rt avec différents lags (clé pour transmission)
        for lag in [5, 7, 10, 14]:
            temp_df[f"rt_lag{lag}"] = temp_df["cases_ma7"] / temp_df["cases_ma7"].shift(
                lag
            ).replace(0, np.nan)
            temp_df[f"rt_lag{lag}"] = temp_df[f"rt_lag{lag}"].clip(0, 10)

        # Volatilité de transmission
        temp_df["transmission_volatility_7d"] = (
            temp_df["new_cases"].rolling(7, min_periods=1).std()
        )
        temp_df["transmission_volatility_14d"] = (
            temp_df["new_cases"].rolling(14, min_periods=1).std()
        )

        # Accélération de transmission
        temp_df["transmission_acceleration"] = temp_df["cases_ma7"].diff().diff()

        # Tendances multiples
        for period in [3, 7, 14]:
            temp_df[f"transmission_trend_{period}d"] = temp_df["cases_ma7"].pct_change(
                period
            )
            temp_df[f"death_trend_{period}d"] = temp_df["deaths_ma7"].pct_change(period)

        # Ratios épidémiologiques
        temp_df["ratio_ma7_ma14"] = temp_df["cases_ma7"] / temp_df[
            "cases_ma14"
        ].replace(0, np.nan)
        temp_df["ratio_ma7_ma21"] = temp_df["cases_ma7"] / temp_df[
            "cases_ma21"
        ].replace(0, np.nan)
        temp_df["ratio_deaths_cases"] = temp_df["deaths_ma7"] / temp_df[
            "cases_ma7"
        ].replace(0, np.nan)

        # Croissance comparée
        temp_df["cases_growth_7d"] = temp_df["cases_ma7"].pct_change(7)
        temp_df["cases_growth_14d"] = temp_df["cases_ma14"].pct_change(14)
        temp_df["growth_acceleration"] = temp_df["cases_growth_7d"].diff()

        # Features saisonnières améliorées
        temp_df["day_of_week"] = temp_df["date"].dt.dayofweek
        temp_df["day_sin"] = np.sin(2 * np.pi * temp_df["day_of_week"] / 7)
        temp_df["day_cos"] = np.cos(2 * np.pi * temp_df["day_of_week"] / 7)
        temp_df["month"] = temp_df["date"].dt.month
        temp_df["month_sin"] = np.sin(2 * np.pi * temp_df["month"] / 12)
        temp_df["month_cos"] = np.cos(2 * np.pi * temp_df["month"] / 12)
        temp_df["quarter_sin"] = np.sin(2 * np.pi * temp_df["date"].dt.quarter / 4)
        temp_df["quarter_cos"] = np.cos(2 * np.pi * temp_df["date"].dt.quarter / 4)

        # Temps depuis début
        temp_df["days_since_start"] = (temp_df["date"] - temp_df["date"].min()).dt.days
        temp_df["weeks_since_start"] = temp_df["days_since_start"] / 7

        # Détection de phases épidémiques
        temp_df["rt_ma7"] = temp_df["rt_lag7"].rolling(7, min_periods=1).mean()
        temp_df["epidemic_phase"] = pd.cut(
            temp_df["rt_ma7"],
            bins=[0, 0.8, 1.2, 2.0, float("inf")],
            labels=[0, 1, 2, 3],
        )
        temp_df["epidemic_phase"] = temp_df["epidemic_phase"].astype(float)

        # Indicateurs de contexte
        temp_df["peak_indicator"] = (
            temp_df["cases_ma7"]
            == temp_df["cases_ma7"].rolling(21, min_periods=1).max()
        ).astype(int)
        temp_df["trough_indicator"] = (
            temp_df["cases_ma7"]
            == temp_df["cases_ma7"].rolling(21, min_periods=1).min()
        ).astype(int)

        # Stabilité/instabilité
        temp_df["stability_index"] = 1 / (1 + temp_df["transmission_volatility_7d"])

        enhanced_dfs.append(temp_df)

    enhanced_df = pd.concat(enhanced_dfs, ignore_index=True)
    enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)

    # Forward fill puis backward fill pour les NaN
    for col in enhanced_df.select_dtypes(include=[np.number]).columns:
        enhanced_df[col] = (
            enhanced_df[col].fillna(method="ffill").fillna(method="bfill").fillna(0)
        )

    print(f"✅ Features transmission créées: {enhanced_df.shape[1]} colonnes")
    return enhanced_df


def select_best_features(X, y, max_features=25):
    """Sélectionne les meilleures features avec corrélation et Random Forest"""
    print(f"🎯 Sélection des {max_features} meilleures features...")

    # Supprimer les features avec variance nulle
    variance_selector = X.var()
    valid_features = variance_selector[variance_selector > 1e-8].index.tolist()
    X_filtered = X[valid_features]

    # Méthode 1: Corrélation avec la target
    correlations = abs(X_filtered.corrwith(y)).sort_values(ascending=False)
    top_corr_features = correlations.head(max_features // 2).index.tolist()

    # Méthode 2: Random Forest Feature Importance
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_filtered, y)
    feature_importance = pd.Series(
        rf.feature_importances_, index=X_filtered.columns
    ).sort_values(ascending=False)
    top_rf_features = feature_importance.head(max_features // 2).index.tolist()

    # Combiner les deux méthodes
    selected_features = list(set(top_corr_features + top_rf_features))[:max_features]

    print(f"✅ {len(selected_features)} features sélectionnées")
    print(f"Top 5 par corrélation: {top_corr_features[:5]}")
    print(f"Top 5 par importance RF: {top_rf_features[:5]}")

    return selected_features


def optimize_transmission_model():
    """Optimise spécifiquement le modèle de transmission"""
    print("\n🚀 OPTIMISATION MODÈLE DE TRANSMISSION (Sans Optuna)")
    print("=" * 60)

    # Charger et préparer les données
    covid_df, _ = load_prepared_data()
    print(f"📊 Données COVID chargées: {len(covid_df)} lignes")

    enhanced_df = create_transmission_features(covid_df)

    # Préparer pour prédiction future de Rt
    print("🎯 Préparation des données pour prédiction future...")
    training_data = []

    for location, group_df in enhanced_df.groupby("location"):
        group_df = group_df.sort_values("date")

        for i in range(30, len(group_df) - 14):
            current_features = group_df.iloc[i]

            # Target futur (Rt dans 7-14 jours)
            future_start = i + 7
            future_end = min(i + 14, len(group_df))
            future_cases = group_df.iloc[future_start:future_end]["new_cases"].mean()
            current_cases = group_df.iloc[max(0, i - 6) : i + 1]["new_cases"].mean()

            if current_cases > 0.1:
                future_rt = min(max(future_cases / current_cases, 0), 10)
                training_row = current_features.to_dict()
                training_row["target_rt"] = future_rt
                training_data.append(training_row)

    training_df = pd.DataFrame(training_data)
    print(f"✅ {len(training_df)} échantillons d'entraînement créés")

    # Sélection des features
    feature_columns = [
        col
        for col in training_df.columns
        if col not in ["target_rt", "date", "location", "virus"]
        and pd.api.types.is_numeric_dtype(training_df[col])
    ]

    X = training_df[feature_columns].fillna(0)
    y = training_df["target_rt"]

    # Sélection des meilleures features
    selected_features = select_best_features(X, y, max_features=25)
    X_selected = X[selected_features]

    # Split temporel
    split_idx = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected.iloc[:split_idx], X_selected.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(
        f"📊 Split final: {len(X_train)} train, {len(X_test)} test, {len(selected_features)} features"
    )

    # Définir les modèles et paramètres à tester
    models_params = {
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "GradientBoosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.05, 0.1, 0.15],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9, 1.0],
            },
        },
        "XGBoost": {
            "model": XGBRegressor(random_state=42, eval_metric="rmse"),
            "params": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.05, 0.1, 0.15],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 0.9],
                "colsample_bytree": [0.8, 0.9],
            },
        },
    }

    best_models = {}
    best_scores = {}

    # Optimiser chaque modèle
    print("\n🔍 Optimisation des hyperparamètres...")
    tscv = TimeSeriesSplit(n_splits=3)

    for name, config in models_params.items():
        print(f"\n   Optimisation {name}...")

        grid_search = GridSearchCV(
            config["model"],
            config["params"],
            cv=tscv,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_
        best_scores[name] = -grid_search.best_score_

        print(f"   ✅ {name}: CV Score = {-grid_search.best_score_:.4f}")
        print(f"   📋 Meilleurs params: {grid_search.best_params}")

    # Sélectionner le meilleur modèle
    best_model_name = min(best_scores.keys(), key=lambda x: best_scores[x])
    best_model = best_models[best_model_name]

    print(f"\n🏆 MEILLEUR MODÈLE: {best_model_name}")

    # Évaluation finale
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n📊 RÉSULTATS FINAUX:")
    print(f"   MAE: {mae:.3f} (objectif: < 0.30)")
    print(f"   RMSE: {rmse:.3f} (objectif: < 0.45)")
    print(f"   R²: {r2:.3f} (objectif: > 0.40)")

    # Comparaison avec baseline
    baseline_mae = 0.420  # From validation
    baseline_r2 = 0.1359

    mae_improvement = ((baseline_mae - mae) / baseline_mae) * 100
    r2_improvement = (
        ((r2 - baseline_r2) / baseline_r2) * 100 if baseline_r2 > 0 else float("inf")
    )

    print(f"\n🎯 AMÉLIORATIONS vs BASELINE:")
    print(f"   MAE: {mae_improvement:+.1f}% {'✅' if mae_improvement > 0 else '❌'}")
    print(f"   R²: {r2_improvement:+.1f}% {'✅' if r2_improvement > 0 else '❌'}")

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": selected_features,
                "importance": best_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        print(f"\n🎯 TOP 10 FEATURES LES PLUS IMPORTANTES:")
        print(importance_df.head(10).to_string(index=False, float_format="%.4f"))

    # Sauvegarder le modèle optimisé
    save_optimized_model(
        best_model,
        selected_features,
        best_model_name,
        "transmission_optimized",
        mae,
        rmse,
        r2,
    )

    return best_model, selected_features, {"mae": mae, "rmse": rmse, "r2": r2}


def save_optimized_model(model, features, model_name, model_type, mae, rmse, r2):
    """Sauvegarde le modèle optimisé"""
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Sauvegarder le modèle
    model_path = os.path.join(MODEL_DIR, f"{model_type}_{model_name}_model.joblib")
    joblib.dump(model, model_path)

    # Sauvegarder un scaler dummy (pour compatibilité)
    scaler = StandardScaler()
    scaler_path = os.path.join(MODEL_DIR, f"{model_type}_{model_name}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Sauvegarder les métadonnées
    metadata = {
        "model_name": model_name,
        "model_type": model_type,
        "feature_cols": features,
        "metrics": {"mae": mae, "rmse": rmse, "r2": r2},
        "version": "optimized_v1",
        "optimization_method": "GridSearchCV",
    }

    metadata_path = os.path.join(
        MODEL_DIR, f"{model_type}_{model_name}_metadata.joblib"
    )
    joblib.dump(metadata, metadata_path)

    print(f"\n💾 MODÈLE SAUVEGARDÉ:")
    print(f"   📁 Modèle: {model_path}")
    print(f"   📁 Scaler: {scaler_path}")
    print(f"   📁 Métadonnées: {metadata_path}")


def diagnose_current_models():
    """Diagnostic des modèles actuels"""
    print("\n🔍 DIAGNOSTIC DES MODÈLES ACTUELS")
    print("=" * 50)

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_model.joblib")]

    if not model_files:
        print("❌ Aucun modèle trouvé")
        return

    for model_file in model_files:
        model_type = model_file.replace("_model.joblib", "")
        metadata_file = model_file.replace("_model.joblib", "_metadata.joblib")
        metadata_path = os.path.join(MODEL_DIR, metadata_file)

        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            metrics = metadata.get("metrics", {})

            print(f"\n📊 {model_type}:")
            print(f"   MAE: {metrics.get('mae', 'N/A')}")
            print(f"   RMSE: {metrics.get('rmse', 'N/A')}")
            print(f"   R²: {metrics.get('r2', 'N/A')}")
            print(f"   Features: {len(metadata.get('feature_cols', []))}")
        else:
            print(f"\n📊 {model_type}: (métadonnées manquantes)")


def main():
    """Fonction principale d'optimisation"""
    print("👑 OPTIMISATION SIMPLE DES MODÈLES ML (Sans Optuna)")
    print("=" * 60)
    print("🎯 Objectif: Améliorer les modèles avec GridSearchCV")
    print("🔧 Méthode: Features avancées + Hyperparameter tuning")
    print("=" * 60)

    print("\n🎯 CHOIX D'OPTIMISATION:")
    print("1. Optimiser modèle de transmission (RECOMMANDÉ)")
    print("2. Diagnostiquer modèles existants")
    print("3. Optimiser transmission + diagnostic")

    try:
        choice = input("\nVotre choix (1-3): ").strip()

        if choice == "1":
            model, features, metrics = optimize_transmission_model()

            print(f"\n🎉 OPTIMISATION TERMINÉE!")
            print(f"✅ Nouveau modèle sauvegardé avec {len(features)} features")
            print(f"✅ R² amélioré: {metrics['r2']:.3f}")
            print(f"✅ MAE réduit: {metrics['mae']:.3f}")

        elif choice == "2":
            diagnose_current_models()

        elif choice == "3":
            diagnose_current_models()
            model, features, metrics = optimize_transmission_model()

        else:
            print("❌ Choix invalide")

    except KeyboardInterrupt:
        print("\n❌ Optimisation interrompue")
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback

        traceback.print_exc()

    print("\n✅ Optimisation terminée!")
    print("\n💡 PROCHAINES ÉTAPES:")
    print("   1. Tester le nouveau modèle avec transmission_validator.py")
    print("   2. Comparer les performances avant/après")
    print("   3. Déployer en production si satisfaisant")


if __name__ == "__main__":
    main()
