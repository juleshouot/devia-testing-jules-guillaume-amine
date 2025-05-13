import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.exceptions import NotFittedError

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# --- Connexion à la BDD ---
def get_engine():
    return create_engine(
        "postgresql+psycopg2://user:guigui@postgres:5432/pandemies",
        connect_args={"options": "-c search_path=pandemics"}
    )


def fetch_location_features_from_db(location_name="France", virus_name="COVID"):
    """
    Récupère les dernières données de la table worldmeter pour une location et un virus donnés.
    Prépare les features nécessaires pour les prédictions de transmission et de mortalité.
    """
    engine = get_engine()
    query = text("""
                 SELECT w.*, l.name AS location, v.name AS virus
                 FROM worldmeter w
                          JOIN location l ON w.location_id = l.id
                          JOIN virus v ON w.virus_id = v.id
                 WHERE l.name = :location
                   AND v.name = :virus
                 ORDER BY w.date DESC LIMIT 14
                 """)
    df = pd.read_sql(query, engine, params={"location": location_name, "virus": virus_name})

    if df.empty or len(df) < 14:
        raise ValueError("Pas assez de données pour générer les features.")

    df = df.sort_values("date")

    # Rolling sums
    df["cases_7day"] = df["new_cases"].rolling(7).sum()
    df["deaths_7day"] = df["new_deaths"].rolling(7).sum()
    df["cases_7day_lag7"] = df["cases_7day"].shift(7)
    df["deaths_7day_lag7"] = df["deaths_7day"].shift(7)

    # Taux de croissance sur 7 jours (en pourcentage entre 0 et 1)
    df["target_growth_rate"] = df["cases_7day"].pct_change(7)
    # Limiter les valeurs extrêmes pour éviter les prédictions aberrantes
    df["target_growth_rate"] = df["target_growth_rate"].clip(-1, 2)

    latest = df.iloc[-1]

    features = {
        "total_cases": latest["total_cases"],
        "total_deaths": latest["total_deaths"],
        "new_cases": latest["new_cases"],
        "new_deaths": latest["new_deaths"],
        "new_cases_smoothed": latest["new_cases_smoothed"],
        "new_deaths_smoothed": latest["new_deaths_smoothed"],
        "new_cases_per_million": latest["new_cases_per_million"],
        "total_cases_per_million": latest["total_cases_per_million"],
        "new_cases_smoothed_per_million": latest["new_cases_smoothed_per_million"],
        "new_deaths_per_million": latest["new_deaths_per_million"],
        "total_deaths_per_million": latest["total_deaths_per_million"],
        "new_deaths_smoothed_per_million": latest["new_deaths_smoothed_per_million"],
        "target_growth_rate": latest["target_growth_rate"],
        "cases_7day": latest["cases_7day"],
        "deaths_7day": latest["deaths_7day"],
        "cases_7day_lag7": latest["cases_7day_lag7"],
        "deaths_7day_lag7": latest["deaths_7day_lag7"],
    }

    return {k: (0 if pd.isna(v) or np.isinf(v) else v) for k, v in features.items()}


# --- Récupération dynamique des données géographiques ---
def fetch_geographical_features(virus_name="COVID"):
    """
    Récupère les données géographiques pour un virus donné.
    Retourne des valeurs par défaut s'il n'y a pas assez de données.
    """
    try:
        engine = get_engine()
        # Requête modifiée pour obtenir l'historique complet sans DISTINCT ON
        query = text("""
                     SELECT l.name      as location,
                            v.name      as virus,
                            MIN(w.date) AS first_date,
                            EXTRACT(YEAR FROM MIN(w.date)) AS year,
                            EXTRACT(WEEK FROM MIN(w.date)) AS week
                     FROM worldmeter w
                         JOIN location l
                     ON w.location_id = l.id
                         JOIN virus v ON w.virus_id = v.id
                     WHERE v.name = :virus AND w.new_cases > 0
                     GROUP BY l.name, v.name
                     ORDER BY first_date
                     """)
        df = pd.read_sql(query, engine, params={"virus": virus_name})

        # Si le DataFrame est vide, retourner des valeurs par défaut
        if df.empty:
            print(
                "⚠️ Aucune donnée trouvée pour calculer les features géographiques. Utilisation de valeurs par défaut.")
            return {
                "lag1": 1.0,
                "lag2": 1.0,
                "lag3": 1.0,
                "lag4": 1.0,
                "ma2": 1.0,
                "ma3": 1.0
            }

        # Format de l'année-semaine amélioré
        df['yearweek'] = df['year'].astype(int).astype(str) + '-' + df['week'].astype(int).astype(str).str.zfill(2)

        # Agrégation par semaine pour compter le nombre de nouvelles localisations
        spread = df.groupby('yearweek').size().reset_index(name='new_locations')

        # Vérifier si nous avons suffisamment de données après le groupby
        if len(spread) < 5:
            print(
                f"⚠️ Données insuffisantes ({len(spread)} semaines). Au moins 5 semaines nécessaires. Utilisation de valeurs par défaut.")
            return {
                "lag1": 1.0,
                "lag2": 1.0,
                "lag3": 1.0,
                "lag4": 1.0,
                "ma2": 1.0,
                "ma3": 1.0
            }

        # Calcul des lags et moyennes mobiles
        for i in range(1, 5):
            spread[f'lag{i}'] = spread['new_locations'].shift(i)
        spread['ma2'] = spread['new_locations'].rolling(2).mean()
        spread['ma3'] = spread['new_locations'].rolling(3).mean()

        # Avant de supprimer les NaN, vérifions s'il nous restera des données
        if len(spread.dropna()) == 0:
            print("⚠️ Toutes les données géographiques sont NA après calcul. Utilisation de valeurs par défaut.")
            return {
                "lag1": 1.0,
                "lag2": 1.0,
                "lag3": 1.0,
                "lag4": 1.0,
                "ma2": 1.0,
                "ma3": 1.0
            }

        spread.dropna(inplace=True)
        latest = spread.iloc[-1]
        return {
            "lag1": latest["lag1"],
            "lag2": latest["lag2"],
            "lag3": latest["lag3"],
            "lag4": latest["lag4"],
            "ma2": latest["ma2"],
            "ma3": latest["ma3"]
        }
    except Exception as e:
        print(f"⚠️ Erreur lors de la récupération des features géographiques: {e}")
        # En cas d'erreur, utiliser des valeurs par défaut
        return {
            "lag1": 1.0,
            "lag2": 1.0,
            "lag3": 1.0,
            "lag4": 1.0,
            "ma2": 1.0,
            "ma3": 1.0
        }


class PandemicPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self._load_models()

    def _prepare_features(self, data, model_type: str) -> np.ndarray:
        if model_type not in self.model_metadata:
            raise ValueError(f"Métadonnées non disponibles pour le type de modèle: {model_type}")

        expected_cols = self.model_metadata[model_type].get('feature_cols', [])
        if not expected_cols:
            raise ValueError(f"Aucune colonne de feature définie pour le modèle: {model_type}")

        print(f"\n[DEBUG] Colonnes attendues pour {model_type}:")
        print(expected_cols)

        if isinstance(data, dict):
            print(f"[DEBUG] Colonnes reçues: {list(data.keys())}")
            data = pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            print(f"[DEBUG] Colonnes reçues: {list(data.columns)}")
        else:
            raise TypeError("Les données doivent être un dict ou un DataFrame.")

        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Données manquantes pour les colonnes: {missing_cols}")

        X = data[expected_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        scaler = self.scalers.get(model_type)
        if not scaler:
            raise ValueError(f"Aucun scaler trouvé pour le modèle {model_type}")

        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du scaling : {e}")

        return X_scaled

    def _load_models(self):
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Le répertoire des modèles n'existe pas: {MODEL_DIR}")

        for model_type in ['transmission', 'mortality', 'geographical_spread']:
            try:
                model_files = [f for f in os.listdir(MODEL_DIR)
                               if f.startswith(model_type) and f.endswith("_model.joblib")]
                if not model_files:
                    print(f"⚠️ Aucun modèle trouvé pour {model_type}")
                    continue

                base_name = model_files[0].replace("_model.joblib", "")
                self.models[model_type] = joblib.load(os.path.join(MODEL_DIR, f"{base_name}_model.joblib"))
                self.scalers[model_type] = joblib.load(os.path.join(MODEL_DIR, f"{base_name}_scaler.joblib"))
                self.model_metadata[model_type] = joblib.load(os.path.join(MODEL_DIR, f"{base_name}_metadata.joblib"))

                print(f"✅ Modèle chargé pour {model_type}: {base_name}_model.joblib")
                print(f"[INFO] Features attendues pour {model_type}:")
                print(self.model_metadata[model_type].get('feature_cols', []))

            except Exception as e:
                print(f"⚠️ Erreur lors du chargement du modèle {model_type}: {e}")

    def predict_transmission_rate(self, data: dict) -> float:
        """Prédit le taux de croissance des cas (entre -1 et 2 soit -100% à +200%)"""
        features = self._prepare_features(data, 'transmission')
        prediction = float(self.models['transmission'].predict(features)[0])
        # Limiter les valeurs extrêmes
        return min(max(prediction, -1.0), 2.0)

    def predict_mortality_rate(self, data: dict) -> float:
        """Prédit le taux de mortalité (entre 0 et 1 soit 0% à 100%)"""
        features = self._prepare_features(data, 'mortality')
        prediction = float(self.models['mortality'].predict(features)[0])
        # Limiter entre 0 et 1
        return min(max(prediction, 0.0), 1.0)

    def predict_geographical_spread(self, data: dict) -> float:
        """Prédit le nombre de nouvelles zones géographiques attendues"""
        features = self._prepare_features(data, 'geographical_spread')
        prediction = float(self.models['geographical_spread'].predict(features)[0])
        # Éviter les valeurs négatives pour un compte
        return max(prediction, 0.0)


if __name__ == "__main__":
    predictor = PandemicPredictor()
    results = {}

    try:
        print("\n=== RÉCUPÉRATION DES DONNÉES ===")
        location_data = None
        geo_data = None

        try:
            location_data = fetch_location_features_from_db("France", "COVID")
            print("✅ Données de location récupérées avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des données de location: {e}")

        try:
            geo_data = fetch_geographical_features()
            print("✅ Données géographiques récupérées avec succès")
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des données géographiques: {e}")

        print("\n=== PRÉDICTIONS ===")
        if geo_data:
            try:
                spread = predictor.predict_geographical_spread(geo_data)
                print(f"✅ Nouvelles zones prévues: {spread:.2f}")
                results["spread"] = spread
            except Exception as e:
                print(f"❌ Erreur lors de la prédiction géographique: {e}")

        if location_data:
            try:
                # Calcul du taux de croissance hebdomadaire (%)
                transmission = predictor.predict_transmission_rate(location_data)
                print(f"✅ Taux de croissance prédit: {transmission:.4f} ({transmission * 100:.2f}%)")
                results["transmission"] = transmission
            except Exception as e:
                print(f"❌ Erreur lors de la prédiction de transmission: {e}")

            try:
                mortality = predictor.predict_mortality_rate(location_data)
                print(f"✅ Taux de mortalité prédit: {mortality:.4f} ({mortality * 100:.2f}%)")
                results["mortality"] = mortality
            except Exception as e:
                print(f"❌ Erreur lors de la prédiction de mortalité: {e}")

        print("\n--- RÉSUMÉ DES PRÉDICTIONS ---")
        if "transmission" in results:
            print(f"Taux de croissance prédit : {results['transmission']:.4f} ({results['transmission'] * 100:.2f}%)")
        else:
            print("Taux de croissance : Non disponible")

        if "mortality" in results:
            print(f"Taux de mortalité prédit  : {results['mortality']:.4f} ({results['mortality'] * 100:.2f}%)")
        else:
            print("Taux de mortalité prédit : Non disponible")

        if "spread" in results:
            print(f"Nouvelles zones prévues   : {results['spread']:.2f}")
        else:
            print("Nouvelles zones prévues : Non disponible")

    except Exception as e:
        print(f"\n❌ Erreur générale lors de l'exécution: {e}")