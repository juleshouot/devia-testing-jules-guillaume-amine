import os
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.exceptions import NotFittedError
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


# --- Connexion à la BDD ---
def get_engine():
    return create_engine(
        "postgresql+psycopg2://user:guigui@postgres:5432/pandemies",
        connect_args={"options": "-c search_path=pandemics"},
    )


def fetch_location_features_from_db(
    location_name="France", virus_name="COVID", days=30
):
    """
    Récupère les dernières données de la table worldmeter pour une location et un virus donnés.
    Prépare les features nécessaires pour les prédictions de transmission et de mortalité.

    Args:
        location_name: Nom de la location (pays/région)
        virus_name: Nom du virus
        days: Nombre de jours à récupérer (par défaut 30, mais nous avons besoin d'au moins 21 pour les calculs)

    Returns:
        Un dictionnaire avec toutes les features nécessaires
    """
    try:
        engine = get_engine()
        query = text(
            """
                     SELECT w.*, l.name AS location, v.name AS virus
                     FROM worldmeter w
                              JOIN location l ON w.location_id = l.id
                              JOIN virus v ON w.virus_id = v.id
                     WHERE l.name = :location
                       AND v.name = :virus
                     ORDER BY w.date DESC LIMIT :days
                     """
        )
        df = pd.read_sql(
            query,
            engine,
            params={"location": location_name, "virus": virus_name, "days": days},
        )

        if df.empty:
            print(f"Aucune donnée trouvée pour {virus_name} à {location_name}")
            return None

        if (
            len(df) < 21
        ):  # Nous avons besoin d'au moins 21 jours de données pour les calculs
            print(
                f"Données insuffisantes pour {virus_name} à {location_name}. Minimum requis: 21 jours, trouvé: {len(df)} jours"
            )
            return None

        # Tri par date et conversion
        df = df.sort_values("date")
        if not pd.api.types.is_datetime64_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # --- PRÉPARATION POUR MODÈLE RT (TRANSMISSION) ---

        # Moyennes mobiles sur 7 jours pour lisser les fluctuations
        df["cases_ma7"] = df["new_cases"].rolling(window=7, min_periods=1).mean()
        df["deaths_ma7"] = df["new_deaths"].rolling(window=7, min_periods=1).mean()

        # Décalage pour calculer Rt
        incubation_period = 7
        df["previous_cases_ma7"] = df["cases_ma7"].shift(incubation_period)

        # Taux de croissance et autres features
        df["cases_growth"] = df["cases_ma7"].pct_change(7).fillna(0).clip(-1, 2)
        df["deaths_growth"] = df["deaths_ma7"].pct_change(7).fillna(0).clip(-1, 2)
        df["cases_acceleration"] = df["cases_growth"].diff().fillna(0)

        # Features temporelles
        df["days_since_start"] = (df["date"] - df["date"].min()).dt.days
        df["day_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

        # --- PRÉPARATION POUR MODÈLE DE MORTALITÉ ---

        # Sommes glissantes sur 7 jours
        df["cases_7day"] = df["new_cases"].rolling(7).sum().fillna(0)
        df["deaths_7day"] = df["new_deaths"].rolling(7).sum().fillna(0)

        # Décalage pour tenir compte du délai entre infection et décès
        lag_days = 14
        df["cases_7day_lag"] = df["cases_7day"].shift(lag_days)

        # Features supplémentaires pour la mortalité
        df["treatment_improvement"] = np.exp(-0.001 * df["days_since_start"])
        df["healthcare_pressure"] = (df["cases_7day"] / df["cases_7day"].max()).fillna(
            0
        )

        # Dernière ligne pour prédiction (données les plus récentes)
        latest_data = df.iloc[-1].to_dict()

        # Remplacer NaN et Inf par 0
        for key, value in latest_data.items():
            try:
                if isinstance(value, (int, float, np.number)):
                    if pd.isna(value) or np.isinf(value):
                        latest_data[key] = 0
                elif pd.isna(value):
                    latest_data[key] = 0
            except:
                continue

        return latest_data
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données: {e}")
        return None


def fetch_global_features_for_geographical_spread(virus_name="COVID", weeks=10):
    """
    Récupère les données pour la prédiction de la propagation géographique

    Args:
        virus_name: Nom du virus
        weeks: Nombre de semaines à récupérer

    Returns:
        Un dictionnaire avec toutes les features nécessaires
    """
    try:
        engine = get_engine()

        # 1. Trouver les premières occurrences pour chaque location
        first_cases_query = text(
            """
                                 SELECT l.name      AS location,
                                        v.name      AS virus,
                                        MIN(w.date) AS first_date
                                 FROM worldmeter w
                                          JOIN location l ON w.location_id = l.id
                                          JOIN virus v ON w.virus_id = v.id
                                 WHERE w.new_cases > 0
                                   AND v.name = :virus
                                 GROUP BY l.name, v.name
                                 ORDER BY first_date DESC LIMIT :weeks
                                 """
        )

        first_cases = pd.read_sql(
            first_cases_query, engine, params={"virus": virus_name, "weeks": weeks * 2}
        )

        if first_cases.empty:
            print(f"Aucune donnée trouvée pour {virus_name}")
            return None

        # Convertir en datetime
        first_cases["first_date"] = pd.to_datetime(first_cases["first_date"])

        # 2. Grouper par semaine
        first_cases["year"] = first_cases["first_date"].dt.year
        first_cases["week"] = first_cases["first_date"].dt.isocalendar().week
        first_cases["yearweek"] = (
            first_cases["year"].astype(str)
            + "-"
            + first_cases["week"].astype(str).str.zfill(2)
        )

        # 3. Compter par semaine
        spread_data = (
            first_cases.groupby("yearweek").size().reset_index(name="new_locations")
        )

        # Convertir yearweek en date
        def yearweek_to_date(yearweek):
            year, week = yearweek.split("-")
            return pd.to_datetime(f"{year}-W{week}-3", format="%Y-W%W-%w")

        spread_data["date"] = spread_data["yearweek"].apply(yearweek_to_date)
        spread_data = spread_data.sort_values("date")

        # 4. Créer les features
        # Lags (valeurs précédentes)
        for i in range(1, 5):
            spread_data[f"lag{i}"] = spread_data["new_locations"].shift(i)

        # Moyennes mobiles
        spread_data["ma2"] = spread_data["new_locations"].rolling(2).mean()
        spread_data["ma3"] = spread_data["new_locations"].rolling(3).mean()

        # Tendance
        spread_data["trend"] = spread_data["ma2"] - spread_data["ma3"].shift(1)

        # Features saisonnières
        spread_data["month"] = spread_data["date"].dt.month
        spread_data["month_sin"] = np.sin(2 * np.pi * spread_data["month"] / 12)
        spread_data["month_cos"] = np.cos(2 * np.pi * spread_data["month"] / 12)

        # Dernière ligne pour prédiction
        latest_data = spread_data.iloc[-1].to_dict()

        # Remplacer NaN et Inf par 0
        for key, value in latest_data.items():
            try:
                if isinstance(value, (int, float, np.number)):
                    if pd.isna(value) or np.isinf(value):
                        latest_data[key] = 0
                elif pd.isna(value):
                    latest_data[key] = 0
            except:
                continue

        return latest_data
    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données: {e}")
        return None


class PandemicPredictor:
    """
    Classe pour prédire divers indicateurs pandémiques:
    - Taux de reproduction (Rt)
    - Taux de mortalité
    - Propagation géographique
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self.load_models()

    def load_models(self):
        """Charge tous les modèles disponibles"""
        if not os.path.exists(MODEL_DIR):
            raise FileNotFoundError(f"Répertoire de modèles non trouvé: {MODEL_DIR}")

        # Charger tous les modèles du répertoire
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith("_model.joblib")]

        for model_file in model_files:
            try:
                # Chemins des fichiers
                model_path = os.path.join(MODEL_DIR, model_file)
                metadata_path = model_file.replace("_model.joblib", "_metadata.joblib")
                metadata_path = os.path.join(MODEL_DIR, metadata_path)
                scaler_path = model_file.replace("_model.joblib", "_scaler.joblib")
                scaler_path = os.path.join(MODEL_DIR, scaler_path)

                # Charger le modèle
                model = joblib.load(model_path)

                # Charger les métadonnées si elles existent
                metadata = {}
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)

                # Charger le scaler si disponible
                scaler = None
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)

                # Extraire le type et le nom à partir des métadonnées si disponibles,
                # sinon à partir du nom de fichier
                if "model_type" in metadata and "model_name" in metadata:
                    model_type = metadata["model_type"]
                    model_name = metadata["model_name"]
                else:
                    # Format du nom de fichier: type_nom_model.joblib
                    parts = model_file.split("_model.joblib")[0].split("_")
                    model_type = parts[0]
                    model_name = " ".join(parts[1:]) if len(parts) > 1 else "default"

                # Stocker dans les dictionnaires
                self.models[(model_type, model_name)] = model
                self.scalers[(model_type, model_name)] = scaler
                self.metadata[(model_type, model_name)] = metadata

                print(f"✅ Modèle chargé: {model_type} - {model_name}")
            except Exception as e:
                print(f"❌ Erreur lors du chargement du modèle {model_file}: {e}")

    def get_best_model(self, model_type):
        """
        Récupère le meilleur modèle disponible pour un type donné

        Args:
            model_type: Type de modèle ('transmission', 'mortality', 'geographical_spread')

        Returns:
            tuple: (modèle, scaler, metadata)
        """
        # Filtrer les modèles par type
        models_of_type = [(k, v) for k, v in self.models.items() if k[0] == model_type]

        # Si aucun modèle n'est trouvé, essayer des variantes du nom
        if not models_of_type:
            # Essayer avec/sans underscore
            if "_" in model_type:
                alt_type = model_type.replace("_", " ")
            else:
                alt_type = model_type.replace(" ", "_")

            models_of_type = [
                (k, v) for k, v in self.models.items() if k[0] == alt_type
            ]

            # Essayer juste la première partie
            if not models_of_type and "-" in model_type:
                alt_type = model_type.split("-")[0].strip()
                models_of_type = [
                    (k, v) for k, v in self.models.items() if k[0].startswith(alt_type)
                ]

        if not models_of_type:
            # Afficher les types disponibles pour le débogage
            available_types = set(k[0] for k in self.models.keys())
            raise ValueError(
                f"Aucun modèle de type '{model_type}' trouvé. Types disponibles: {available_types}"
            )

        # Trouver le modèle avec la meilleure performance (RMSE le plus bas)
        best_key = min(
            models_of_type,
            key=lambda x: self.metadata.get(x[0], {}).get("rmse", float("inf")),
        )[0]

        return self.models[best_key], self.scalers[best_key], self.metadata[best_key]

    def predict_rt(self, location_name="France", virus_name="COVID"):
        """
        Prédit le taux de reproduction effectif (Rt) pour une localisation et un virus

        Args:
            location_name: Nom de la location
            virus_name: Nom du virus

        Returns:
            float: Le Rt prédit
        """
        try:
            # Récupérer le meilleur modèle de transmission
            model, scaler, metadata = self.get_best_model("transmission")

            # Récupérer les données et les features
            features_dict = fetch_location_features_from_db(location_name, virus_name)

            if features_dict is None:
                print(
                    f"❌ Impossible de récupérer les données pour {virus_name} à {location_name}"
                )
                return None

            feature_cols = metadata.get("feature_cols", [])

            # Extraire les valeurs des features
            features = np.array(
                [features_dict.get(col, 0) for col in feature_cols]
            ).reshape(1, -1)

            # Standardiser les features si un scaler est disponible
            if scaler:
                features = scaler.transform(features)

            # Prédire Rt
            rt_pred = model.predict(features)[0]

            # Limiter la valeur prédite (0-10)
            rt_pred = max(0, min(rt_pred, 10))

            return rt_pred

        except (NotFittedError, KeyError, ValueError) as e:
            print(f"❌ Erreur lors de la prédiction du Rt: {e}")
            return None

    def predict_mortality_ratio(self, location_name="France", virus_name="COVID"):
        """
        Prédit le taux de mortalité pour une localisation et un virus

        Args:
            location_name: Nom de la location
            virus_name: Nom du virus

        Returns:
            float: Le taux de mortalité prédit (0-1)
        """
        try:
            # Récupérer le meilleur modèle de mortalité
            model, scaler, metadata = self.get_best_model("mortality")

            # Récupérer les données et les features
            features_dict = fetch_location_features_from_db(location_name, virus_name)

            if features_dict is None:
                print(
                    f"❌ Impossible de récupérer les données pour {virus_name} à {location_name}"
                )
                return None

            feature_cols = metadata.get("feature_cols", [])

            # Extraire les valeurs des features
            features = np.array(
                [features_dict.get(col, 0) for col in feature_cols]
            ).reshape(1, -1)

            # Standardiser les features si un scaler est disponible
            if scaler:
                features = scaler.transform(features)

            # Prédire le taux de mortalité
            mortality_pred = model.predict(features)[0]

            # Limiter la valeur prédite (0-1)
            mortality_pred = max(0, min(mortality_pred, 1))

            return mortality_pred

        except (NotFittedError, KeyError, ValueError) as e:
            print(f"❌ Erreur lors de la prédiction du taux de mortalité: {e}")
            return None

    def predict_geographical_spread(self, virus_name="COVID"):
        """
        Prédit le nombre de nouvelles localisations touchées pour la semaine suivante

        Args:
            virus_name: Nom du virus

        Returns:
            int: Le nombre prédit de nouvelles localisations touchées
        """
        try:
            # Récupérer le meilleur modèle de propagation géographique
            model, scaler, metadata = self.get_best_model("geographical_spread")

            # Récupérer les données et les features
            features_dict = fetch_global_features_for_geographical_spread(virus_name)

            if features_dict is None:
                print(
                    f"❌ Impossible de récupérer les données globales pour {virus_name}"
                )
                return None

            feature_cols = metadata.get("feature_cols", [])

            # Extraire les valeurs des features
            features = np.array(
                [features_dict.get(col, 0) for col in feature_cols]
            ).reshape(1, -1)

            # Standardiser les features si un scaler est disponible
            if scaler:
                features = scaler.transform(features)

            # Prédire le nombre de nouvelles localisations
            spread_pred = model.predict(features)[0]

            # Arrondir à l'entier le plus proche (minimum 0)
            spread_pred = max(0, round(spread_pred))

            return spread_pred

        except (NotFittedError, KeyError, ValueError) as e:
            print(
                f"❌ Erreur lors de la prédiction de la propagation géographique: {e}"
            )
            return None


if __name__ == "__main__":
    # Test du prédicteur
    predictor = PandemicPredictor()

    try:
        # Récupérer toutes les localisations disponibles dans la base de données
        engine = get_engine()
        locations_query = text(
            """
                               SELECT DISTINCT l.name
                               FROM location l
                                        JOIN worldmeter w ON l.id = w.location_id
                               ORDER BY l.name
                               """
        )
        all_locations = pd.read_sql(locations_query, engine)["name"].tolist()

        # Récupérer tous les virus disponibles
        viruses_query = text(
            """
                             SELECT DISTINCT v.name
                             FROM virus v
                                      JOIN worldmeter w ON v.id = w.virus_id
                             ORDER BY v.name
                             """
        )
        all_viruses = pd.read_sql(viruses_query, engine)["name"].tolist()

        print(
            f"\n=== PRÉDICTIONS ÉPIDÉMIQUES POUR {len(all_viruses)} VIRUS ET {len(all_locations)} LOCALISATIONS ===\n"
        )

        # Résultats par virus
        for virus in all_viruses:
            print(f"\n--- VIRUS: {virus} ---\n")

            # Prédiction de propagation géographique globale
            try:
                spread_pred = predictor.predict_geographical_spread(virus)
                print(
                    f"📊 Propagation géographique prédite: {spread_pred} nouvelles localisations"
                )
            except Exception as e:
                print(
                    f"❌ Erreur lors de la prédiction de propagation pour {virus}: {e}"
                )

            print("\n| Location | Rt | Statut | Mortalité |")
            print("|----------|-----|--------|-----------|")

            # Compteur pour les statistiques
            success_count = 0
            decline_count = 0
            growth_count = 0

            # Prédictions par localisation
            for location in all_locations:
                try:
                    # Rt
                    rt_pred = predictor.predict_rt(location, virus)
                    if rt_pred is None:
                        continue

                    # Déterminer le statut épidémique
                    if rt_pred < 0.9:
                        status = "Fort déclin"
                        decline_count += 1
                    elif rt_pred < 1.0:
                        status = "Déclin"
                        decline_count += 1
                    elif rt_pred < 1.1:
                        status = "Faible croissance"
                        growth_count += 1
                    else:
                        status = "Forte croissance"
                        growth_count += 1

                    # Mortalité
                    mortality_pred = predictor.predict_mortality_ratio(location, virus)
                    if mortality_pred is None:
                        mortality_str = "N/A"
                    else:
                        mortality_str = f"{mortality_pred:.2%}"

                    # Afficher les résultats
                    print(
                        f"| {location} | {rt_pred:.2f} | {status} | {mortality_str} |"
                    )
                    success_count += 1

                except Exception:
                    # Si erreur, passer à la localisation suivante
                    continue

            # Résumé pour ce virus
            print(f"\n📈 Résumé pour {virus}:")
            print(
                f"  - Prédictions réussies: {success_count}/{len(all_locations)} localisations"
            )
            if success_count > 0:
                print(
                    f"  - Localisations en déclin: {decline_count} ({decline_count / success_count * 100:.1f}%)"
                )
                print(
                    f"  - Localisations en croissance: {growth_count} ({growth_count / success_count * 100:.1f}%)"
                )

        print("\n=== FIN DES PRÉDICTIONS ===\n")

    except Exception as e:
        print(f"❌ Erreur générale lors des prédictions: {e}")
