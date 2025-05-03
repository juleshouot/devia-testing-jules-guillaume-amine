import os
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class PandemicPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self._load_models()

    def _prepare_features(self, data, model_type: str) -> np.ndarray:
        """
        Prépare les données pour la prédiction (format, colonnes, scaling).
        """
        if model_type not in self.model_metadata:
            raise ValueError(f"Métadonnées non disponibles pour le type de modèle: {model_type}")

        expected_cols = self.model_metadata[model_type].get('feature_cols', [])
        if not expected_cols:
            raise ValueError(f"Aucune colonne de feature définie pour le modèle: {model_type}")

        # Convertir en DataFrame si nécessaire
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            raise TypeError("Les données doivent être un dict ou un DataFrame.")

        # Vérifier les colonnes manquantes
        missing_cols = [col for col in expected_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Données manquantes pour les colonnes: {missing_cols}")

        # Filtrer et nettoyer les données
        X = data[expected_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Appliquer le scaler
        scaler = self.scalers.get(model_type)
        if not scaler:
            raise ValueError(f"Aucun scaler trouvé pour le modèle {model_type}")

        try:
            X_scaled = scaler.transform(X)
        except Exception as e:
            raise RuntimeError(f"Erreur lors du scaling : {e}")

        return X_scaled

    def _load_models(self):
        """
        Charge les modèles, scalers et métadonnées depuis le dossier des modèles.
        """
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
                model_path = os.path.join(MODEL_DIR, f"{base_name}_model.joblib")
                scaler_path = os.path.join(MODEL_DIR, f"{base_name}_scaler.joblib")
                metadata_path = os.path.join(MODEL_DIR, f"{base_name}_metadata.joblib")

                self.models[model_type] = joblib.load(model_path)
                self.scalers[model_type] = joblib.load(scaler_path)
                self.model_metadata[model_type] = joblib.load(metadata_path)

                print(f"✅ Modèle chargé pour {model_type}: {os.path.basename(model_path)}")

            except Exception as e:
                print(f"⚠️ Erreur lors du chargement du modèle {model_type}: {e}")

    def predict_transmission_rate(self, data: dict) -> float:
        features = self._prepare_features(data, 'transmission')
        prediction = self.models['transmission'].predict(features)[0]
        return float(prediction)

    def predict_mortality_rate(self, data: dict) -> float:
        features = self._prepare_features(data, 'mortality')
        prediction = self.models['mortality'].predict(features)[0]
        return float(prediction)

    def predict_geographical_spread(self, data: dict) -> float:
        features = self._prepare_features(data, 'geographical_spread')
        prediction = self.models['geographical_spread'].predict(features)[0]
        return float(prediction)


if __name__ == "__main__":
    predictor = PandemicPredictor()

    # Exemple de données de test à adapter selon tes features exactes
    location_data = {
        "new_cases_smoothed_per_million": 120,
        "new_deaths_smoothed_per_million": 0.5,
        "total_cases_per_million": 150000,
        "total_deaths_per_million": 2300,
        "new_cases_per_million": 80,
        "new_deaths_per_million": 0.4,
        "new_cases": 5000,
        "new_deaths": 25,
        "total_cases": 300000,
        "total_deaths": 9000,
        "new_cases_smoothed": 4700,
        "new_deaths_smoothed": 22,
    }

    geographical_data = {
        'new_locations_lag1': 10,
        'new_locations_lag2': 8,
        'new_locations_lag3': 6,
        'new_locations_lag4': 7,
        'new_locations_ma2': 9,
        'new_locations_ma3': 8.5
    }

    try:
        transmission_rate = predictor.predict_transmission_rate(location_data)
        mortality_rate = predictor.predict_mortality_rate(location_data)
        spread = predictor.predict_geographical_spread(geographical_data)

        print("\n--- Prédictions ---")
        print(f"Taux de transmission prédit : {transmission_rate:.4f}")
        print(f"Taux de mortalité prédit    : {mortality_rate:.4f}")
        print(f"Nouvelles zones prévues     : {spread:.2f}")
    except Exception as e:
        print(f"❌ Erreur lors de la prédiction : {e}")
