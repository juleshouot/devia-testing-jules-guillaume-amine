import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
STATIC_DIR = os.path.join(BASE_DIR, "static", "visualizations")


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


def load_mortality_model():
    """Charge le modèle de mortalité entrainé"""
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"❌ Répertoire de modèles non trouvé: {MODEL_DIR}")
            return None, None, None

        # Chercher les fichiers de modèle de mortalité
        model_files = [
            f
            for f in os.listdir(MODEL_DIR)
            if "mortality" in f and f.endswith("_model.joblib")
        ]

        if not model_files:
            print("❌ Aucun modèle de mortalité trouvé")
            return None, None, None

        model_file = model_files[0]
        print(f"✅ Chargement du modèle: {model_file}")

        # Charger le modèle
        model_path = os.path.join(MODEL_DIR, model_file)
        model = joblib.load(model_path)

        # Charger le scaler
        scaler_file = model_file.replace("_model.joblib", "_scaler.joblib")
        scaler_path = os.path.join(MODEL_DIR, scaler_file)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # Charger les métadonnées
        metadata_file = model_file.replace("_model.joblib", "_metadata.joblib")
        metadata_path = os.path.join(MODEL_DIR, metadata_file)
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}

        feature_cols = metadata.get(
            "feature_cols",
            [
                "cases_7day",
                "deaths_7day",
                "cases_7day_lag",
                "days_since_start",
                "treatment_improvement",
                "healthcare_pressure",
                "total_cases",
                "total_deaths",
                "new_cases",
                "new_deaths",
                "new_cases_per_million",
                "total_cases_per_million",
                "new_deaths_per_million",
                "total_deaths_per_million",
            ],
        )

        print(f"🔧 Features utilisées: {len(feature_cols)}")
        return model, scaler, feature_cols

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None, None


def get_available_locations():
    """Récupère les localisations avec suffisamment de données COVID"""
    try:
        engine = get_engine()
        query = text(
            """
                     SELECT l.name            as location,
                            COUNT(*)          as data_points,
                            SUM(w.new_cases)  as total_cases,
                            SUM(w.new_deaths) as total_deaths
                     FROM worldmeter w
                              JOIN location l ON w.location_id = l.id
                              JOIN virus v ON w.virus_id = v.id
                     WHERE v.name = 'COVID'
                       AND w.date BETWEEN '2020-03-01' AND '2022-05-01'
                     GROUP BY l.name
                     HAVING COUNT(*) >= 100          -- Au moins ~3 mois de données
                        AND SUM(w.new_cases) >= 1000 -- Seuil minimum de cas
                     ORDER BY SUM(w.new_cases) DESC
                     """
        )

        df = pd.read_sql(query, engine)
        print(f"📍 {len(df)} localisations trouvées avec suffisamment de données")
        print(
            df[["location", "data_points", "total_cases", "total_deaths"]]
            .head(10)
            .to_string(index=False)
        )

        return df["location"].tolist()

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des localisations: {e}")
        return []


def get_covid_data(location, start_date="2020-03-01", end_date="2022-05-01"):
    """Récupère les données COVID pour une localisation"""
    try:
        engine = get_engine()
        query = text(
            """
                     SELECT w.date,
                            w.new_cases,
                            w.new_deaths,
                            w.total_cases,
                            w.total_deaths,
                            w.new_cases_per_million,
                            w.total_cases_per_million,
                            w.new_deaths_per_million,
                            w.total_deaths_per_million
                     FROM worldmeter w
                              JOIN location l ON w.location_id = l.id
                              JOIN virus v ON w.virus_id = v.id
                     WHERE l.name = :location
                       AND v.name = 'COVID'
                       AND w.date BETWEEN :start_date AND :end_date
                     ORDER BY w.date ASC
                     """
        )

        df = pd.read_sql(
            query,
            engine,
            params={
                "location": location,
                "start_date": start_date,
                "end_date": end_date,
            },
        )

        if df.empty:
            print(f"❌ Aucune donnée trouvée pour {location}")
            return None

        df["date"] = pd.to_datetime(df["date"])
        print(f"📊 {len(df)} lignes de données récupérées pour {location}")

        return df

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données pour {location}: {e}")
        return None


def create_features_from_data(df, current_idx):
    """Crée les features pour prédiction (mêmes que l'entrainement)"""
    try:
        # Utiliser seulement les données jusqu'à l'index actuel
        data = df.iloc[: current_idx + 1].copy()

        # Gérer les valeurs manquantes
        data["new_cases"] = data["new_cases"].fillna(0)
        data["new_deaths"] = data["new_deaths"].fillna(0)

        # Sommes glissantes sur 7 jours
        data["cases_7day"] = data["new_cases"].rolling(7, min_periods=1).sum()
        data["deaths_7day"] = data["new_deaths"].rolling(7, min_periods=1).sum()

        # Décalage de 14 jours
        data["cases_7day_lag"] = data["cases_7day"].shift(14)

        # Features temporelles
        data["days_since_start"] = (data["date"] - data["date"].min()).dt.days
        data["treatment_improvement"] = np.exp(-0.001 * data["days_since_start"])
        data["healthcare_pressure"] = (
            data["cases_7day"] / data["cases_7day"].max()
        ).fillna(0)

        # Récupérer les dernières valeurs
        latest = data.iloc[-1]

        features = {
            "cases_7day": latest.get("cases_7day", 0),
            "deaths_7day": latest.get("deaths_7day", 0),
            "cases_7day_lag": latest.get("cases_7day_lag", 0),
            "days_since_start": latest.get("days_since_start", 0),
            "treatment_improvement": latest.get("treatment_improvement", 1),
            "healthcare_pressure": latest.get("healthcare_pressure", 0),
            "total_cases": latest.get("total_cases", 0),
            "total_deaths": latest.get("total_deaths", 0),
            "new_cases": latest.get("new_cases", 0),
            "new_deaths": latest.get("new_deaths", 0),
            "new_cases_per_million": latest.get("new_cases_per_million", 0),
            "total_cases_per_million": latest.get("total_cases_per_million", 0),
            "new_deaths_per_million": latest.get("new_deaths_per_million", 0),
            "total_deaths_per_million": latest.get("total_deaths_per_million", 0),
        }

        # Nettoyer les valeurs NaN et inf
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                features[key] = 0

        return features

    except Exception:
        return None


def calculate_actual_mortality(df, current_idx):
    """Calcule le taux de mortalité réel qui s'est produit"""
    try:
        # Cas de référence (14 jours avant le point actuel)
        if current_idx >= 14:
            ref_cases = df.iloc[current_idx - 14 : current_idx - 7]["new_cases"].sum()
        else:
            ref_cases = df.iloc[:current_idx]["new_cases"].tail(7).sum()

        # Décès dans les 14 jours suivants
        future_end_idx = min(current_idx + 14, len(df) - 1)
        future_deaths = df.iloc[current_idx:future_end_idx]["new_deaths"].sum()

        if ref_cases > 10:  # Seuil minimum
            return future_deaths / ref_cases
        return None

    except Exception:
        return None


def validate_single_location(location, model, scaler, feature_cols):
    """Valide le modèle pour une seule localisation"""
    print(f"\n🔍 Validation pour {location}...")

    # Récupérer les données
    df = get_covid_data(location)
    if df is None or len(df) < 50:
        print(f"❌ Données insuffisantes pour {location}")
        return None

    results = []

    # Parcourir le temps et faire des prédictions
    for i in range(30, len(df) - 14):  # Besoin d'historique et de données futures

        # Créer les features avec les données jusqu'au point actuel
        features_dict = create_features_from_data(df, i)
        if features_dict is None:
            continue

        # Faire la prédiction
        try:
            features_array = np.array(
                [features_dict.get(col, 0) for col in feature_cols]
            ).reshape(1, -1)
            if scaler is not None:
                features_array = scaler.transform(features_array)

            predicted_mortality = model.predict(features_array)[0]
            predicted_mortality = max(0, min(predicted_mortality, 1))  # Limiter à [0,1]

        except Exception:
            continue

        # Calculer le taux de mortalité réel
        actual_mortality = calculate_actual_mortality(df, i)

        if actual_mortality is not None and actual_mortality > 0:
            # Ajouter les données de contexte
            current_row = df.iloc[i]
            results.append(
                {
                    "date": current_row["date"],
                    "predicted": predicted_mortality,
                    "actual": actual_mortality,
                    "error": abs(predicted_mortality - actual_mortality),
                    "rel_error": abs(predicted_mortality - actual_mortality)
                    / max(actual_mortality, 0.001),
                    # Données de contexte
                    "total_cases": current_row.get("total_cases", 0),
                    "total_deaths": current_row.get("total_deaths", 0),
                    "new_cases_7d": df.iloc[max(0, i - 6) : i + 1]["new_cases"].sum(),
                    "new_deaths_7d": df.iloc[max(0, i - 6) : i + 1]["new_deaths"].sum(),
                    "cases_per_million": current_row.get("total_cases_per_million", 0),
                }
            )

    if len(results) < 10:
        print(f"❌ Pas assez de prédictions valides pour {location}")
        return None

    print(f"✅ {len(results)} prédictions générées")
    return results


def print_validation_metrics(results, location):
    """Affiche les métriques de validation"""
    df = pd.DataFrame(results)

    # Métriques de base
    mae = mean_absolute_error(df["actual"], df["predicted"])
    rmse = np.sqrt(mean_squared_error(df["actual"], df["predicted"]))
    r2 = r2_score(df["actual"], df["predicted"])

    print(f"\n📊 RÉSULTATS DE VALIDATION POUR {location.upper()}")
    print("=" * 50)
    print(f"Nombre de prédictions: {len(results)}")
    print(
        f"Période: {df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}"
    )

    print(f"\n🎯 MÉTRIQUES DE PRÉCISION:")
    print(f"Erreur Absolue Moyenne (MAE): {mae:.4f}")
    print(f"Erreur Quadratique Moyenne (RMSE): {rmse:.4f}")
    print(f"Score R²: {r2:.4f}")

    # Erreurs relatives
    median_rel_error = df["rel_error"].median()
    mean_rel_error = df["rel_error"].mean()

    print(f"\n📈 PERFORMANCE RELATIVE:")
    print(f"Erreur relative médiane: {median_rel_error:.1%}")
    print(f"Erreur relative moyenne: {mean_rel_error:.1%}")

    # Distribution de qualité
    excellent = (df["rel_error"] < 0.15).sum()
    good = ((df["rel_error"] >= 0.15) & (df["rel_error"] < 0.30)).sum()
    fair = ((df["rel_error"] >= 0.30) & (df["rel_error"] < 0.50)).sum()
    poor = (df["rel_error"] >= 0.50).sum()

    print(f"\n🏆 QUALITÉ DES PRÉDICTIONS:")
    print(f"Excellentes (<15% erreur): {excellent} ({excellent / len(df) * 100:.1f}%)")
    print(f"Bonnes (15-30% erreur): {good} ({good / len(df) * 100:.1f}%)")
    print(f"Correctes (30-50% erreur): {fair} ({fair / len(df) * 100:.1f}%)")
    print(f"Mauvaises (>50% erreur): {poor} ({poor / len(df) * 100:.1f}%)")

    # Afficher les détails des prédictions
    show_details = (
        input(f"\n📋 Afficher les détails des prédictions ? (o/n): ").strip().lower()
    )
    if show_details in ["o", "oui", "y", "yes"]:
        show_prediction_details(df, location)


def show_prediction_details(df, location):
    """Affiche les détails des prédictions avec dates et taux"""
    print(f"\n📋 DÉTAILS DES PRÉDICTIONS POUR {location.upper()}")
    print("=" * 80)

    # Options d'affichage
    print("Options d'affichage:")
    print("1. Toutes les prédictions")
    print("2. Les 20 premières prédictions")
    print("3. Les 20 dernières prédictions")
    print("4. Les meilleures prédictions (erreur < 15%)")
    print("5. Les pires prédictions (erreur > 50%)")
    print("6. Prédictions d'une période spécifique")

    try:
        choice = input("\nVotre choix (1-6): ").strip()

        if choice == "1":
            display_df = df
            title = "TOUTES LES PRÉDICTIONS"
        elif choice == "2":
            display_df = df.head(20)
            title = "20 PREMIÈRES PRÉDICTIONS"
        elif choice == "3":
            display_df = df.tail(20)
            title = "20 DERNIÈRES PRÉDICTIONS"
        elif choice == "4":
            display_df = df[df["rel_error"] < 0.15].head(20)
            title = "MEILLEURES PRÉDICTIONS (Erreur < 15%)"
        elif choice == "5":
            display_df = df[df["rel_error"] > 0.50].head(20)
            title = "PIRES PRÉDICTIONS (Erreur > 50%)"
        elif choice == "6":
            year = input("Année (ex: 2020, 2021, 2022): ").strip()
            month = input("Mois (optionnel, ex: 03, 12): ").strip()

            filtered_df = df[df["date"].dt.year == int(year)]
            if month:
                filtered_df = filtered_df[filtered_df["date"].dt.month == int(month)]

            display_df = filtered_df
            title = f"PRÉDICTIONS {year}" + (f"-{month}" if month else "")
        else:
            display_df = df.head(20)
            title = "20 PREMIÈRES PRÉDICTIONS (par défaut)"

        if len(display_df) == 0:
            print("❌ Aucune prédiction trouvée pour cette sélection")
            return

        print(f"\n📊 {title}")
        print("=" * 120)
        print(
            f"{'Date':<12} {'Prédit':<8} {'Réel':<8} {'Erreur':<8} {'Erreur%':<8} {'Qualité':<12} {'Total Cas':<10} {'Total Morts':<11} {'Cas 7j':<8} {'Morts 7j':<9}"
        )
        print("-" * 120)

        for _, row in display_df.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            predicted_pct = f"{row['predicted'] * 100:.2f}%"
            actual_pct = f"{row['actual'] * 100:.2f}%"
            error_abs = f"{row['error'] * 100:.2f}%"
            error_rel = f"{row['rel_error'] * 100:.1f}%"

            # Données de contexte
            total_cases = f"{int(row.get('total_cases', 0)):,}"
            total_deaths = f"{int(row.get('total_deaths', 0)):,}"
            cases_7d = f"{int(row.get('new_cases_7d', 0)):,}"
            deaths_7d = f"{int(row.get('new_deaths_7d', 0)):,}"

            # Déterminer la qualité
            if row["rel_error"] < 0.15:
                quality = "Excellente"
            elif row["rel_error"] < 0.30:
                quality = "Bonne"
            elif row["rel_error"] < 0.50:
                quality = "Correcte"
            else:
                quality = "Mauvaise"

            print(
                f"{date_str:<12} {predicted_pct:<8} {actual_pct:<8} {error_abs:<8} {error_rel:<8} {quality:<12} {total_cases:<10} {total_deaths:<11} {cases_7d:<8} {deaths_7d:<9}"
            )

        # Statistiques de la sélection
        print("\n📈 STATISTIQUES DE LA SÉLECTION:")
        print(f"Nombre de prédictions: {len(display_df)}")
        print(f"Erreur moyenne: {display_df['rel_error'].mean() * 100:.1f}%")
        print(f"Erreur médiane: {display_df['rel_error'].median() * 100:.1f}%")
        print(
            f"Meilleure prédiction: {display_df['rel_error'].min() * 100:.1f}% d'erreur"
        )
        print(f"Pire prédiction: {display_df['rel_error'].max() * 100:.1f}% d'erreur")

        # Contexte épidémiologique
        total_cases_range = f"{int(display_df['total_cases'].min()):,} → {int(display_df['total_cases'].max()):,}"
        total_deaths_range = f"{int(display_df['total_deaths'].min()):,} → {int(display_df['total_deaths'].max()):,}"
        print(f"\nContexte épidémiologique:")
        print(f"Plage de cas totaux: {total_cases_range}")
        print(f"Plage de décès totaux: {total_deaths_range}")

        avg_cases_7d = int(display_df["new_cases_7d"].mean())
        avg_deaths_7d = int(display_df["new_deaths_7d"].mean())
        print(f"Moyenne cas/semaine: {avg_cases_7d:,}")
        print(f"Moyenne décès/semaine: {avg_deaths_7d:,}")

        # Ajouter analyses contextuelles
        print(f"\n🔍 ANALYSES CONTEXTUELLES:")

        # Périodes avec le plus de cas
        high_cases_period = display_df.nlargest(3, "total_cases")
        print(f"\nPériodes avec le plus de cas:")
        for _, row in high_cases_period.iterrows():
            print(
                f"  {row['date'].strftime('%Y-%m-%d')}: {int(row['total_cases']):,} cas totaux, erreur: {row['rel_error'] * 100:.1f}%"
            )

        # Périodes avec le plus de décès
        high_deaths_period = display_df.nlargest(3, "total_deaths")
        print(f"\nPériodes avec le plus de décès:")
        for _, row in high_deaths_period.iterrows():
            print(
                f"  {row['date'].strftime('%Y-%m-%d')}: {int(row['total_deaths']):,} décès totaux, erreur: {row['rel_error'] * 100:.1f}%"
            )

        # Corrélation entre volume de cas et précision
        if len(display_df) > 10:
            import numpy as np

            correlation_cases = np.corrcoef(
                display_df["total_cases"], display_df["rel_error"]
            )[0, 1]
            correlation_deaths = np.corrcoef(
                display_df["total_deaths"], display_df["rel_error"]
            )[0, 1]
            print(f"\nCorrélation erreur vs contexte:")
            print(f"  Erreur vs Total cas: {correlation_cases:.3f}")
            print(f"  Erreur vs Total décès: {correlation_deaths:.3f}")

            if abs(correlation_cases) > 0.3:
                if correlation_cases > 0:
                    print(f"  → Plus de cas = prédictions moins précises")
                else:
                    print(f"  → Plus de cas = prédictions plus précises")

        # Évolution temporelle
        if len(display_df) > 5:
            first_half = display_df.head(len(display_df) // 2)
            second_half = display_df.tail(len(display_df) // 2)

            first_error = first_half["rel_error"].mean()
            second_error = second_half["rel_error"].mean()

            print(f"\nÉvolution temporelle:")
            print(f"  Première moitié: {first_error * 100:.1f}% d'erreur moyenne")
            print(f"  Seconde moitié: {second_error * 100:.1f}% d'erreur moyenne")

            if first_error > second_error:
                improvement = ((first_error - second_error) / first_error) * 100
                print(f"  → Amélioration de {improvement:.1f}% dans le temps")
            else:
                degradation = ((second_error - first_error) / first_error) * 100
                print(f"  → Dégradation de {degradation:.1f}% dans le temps")

        # Option pour sauvegarder
        save_csv = (
            input(f"\n💾 Sauvegarder ces données en CSV ? (o/n): ").strip().lower()
        )
        if save_csv in ["o", "oui", "y", "yes"]:
            save_prediction_details_csv(display_df, location, title)

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Affichage annulé")


def save_prediction_details_csv(df, location, title):
    """Sauvegarde les détails des prédictions en CSV"""
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)

        # Préparer les données pour CSV
        csv_df = df.copy()
        csv_df["date"] = csv_df["date"].dt.strftime("%Y-%m-%d")
        csv_df["predicted_pct"] = (csv_df["predicted"] * 100).round(3)
        csv_df["actual_pct"] = (csv_df["actual"] * 100).round(3)
        csv_df["error_pct"] = (csv_df["error"] * 100).round(3)
        csv_df["rel_error_pct"] = (csv_df["rel_error"] * 100).round(1)

        # Ajouter colonne qualité
        csv_df["quality"] = csv_df["rel_error"].apply(
            lambda x: (
                "Excellente"
                if x < 0.15
                else "Bonne" if x < 0.30 else "Correcte" if x < 0.50 else "Mauvaise"
            )
        )

        # Ajouter colonnes de contexte
        csv_df["total_cases"] = csv_df.get("total_cases", 0).astype(int)
        csv_df["total_deaths"] = csv_df.get("total_deaths", 0).astype(int)
        csv_df["new_cases_7d"] = csv_df.get("new_cases_7d", 0).astype(int)
        csv_df["new_deaths_7d"] = csv_df.get("new_deaths_7d", 0).astype(int)
        csv_df["cases_per_million"] = csv_df.get("cases_per_million", 0).round(1)

        # Sélectionner et renommer les colonnes
        final_df = csv_df[
            [
                "date",
                "predicted_pct",
                "actual_pct",
                "error_pct",
                "rel_error_pct",
                "quality",
                "total_cases",
                "total_deaths",
                "new_cases_7d",
                "new_deaths_7d",
                "cases_per_million",
            ]
        ]
        final_df.columns = [
            "Date",
            "Taux_Prédit_%",
            "Taux_Réel_%",
            "Erreur_Absolue_%",
            "Erreur_Relative_%",
            "Qualité",
            "Total_Cas",
            "Total_Décès",
            "Nouveaux_Cas_7j",
            "Nouveaux_Décès_7j",
            "Cas_Par_Million",
        ]

        # Nom du fichier
        safe_title = (
            title.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("%", "pct")
        )
        safe_location = location.lower().replace(" ", "_")
        filename = f"predictions_{safe_location}_{safe_title}.csv"
        filepath = os.path.join(STATIC_DIR, filename)

        # Sauvegarder
        final_df.to_csv(filepath, index=False, encoding="utf-8")
        print(f"💾 Données sauvegardées: {filepath}")

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde CSV: {e}")


def save_validation_plots(results, location):
    """Sauvegarde les graphiques de validation"""
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)

        df = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Validation du Modèle de Mortalité: {location}", fontsize=16)

        # 1. Prédit vs Réel
        axes[0, 0].scatter(df["actual"] * 100, df["predicted"] * 100, alpha=0.6, s=30)
        max_val = max(df["actual"].max() * 100, df["predicted"].max() * 100)
        axes[0, 0].plot(
            [0, max_val], [0, max_val], "r--", linewidth=2, label="Prédiction Parfaite"
        )
        axes[0, 0].set_xlabel("Taux de Mortalité Réel (%)")
        axes[0, 0].set_ylabel("Taux de Mortalité Prédit (%)")
        axes[0, 0].set_title("Prédit vs Réel")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Erreur dans le temps
        axes[0, 1].plot(
            df["date"], df["rel_error"] * 100, marker="o", markersize=3, alpha=0.7
        )
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Erreur Relative (%)")
        axes[0, 1].set_title("Évolution de l'Erreur")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution des erreurs
        axes[1, 0].hist(df["rel_error"] * 100, bins=30, alpha=0.7, edgecolor="black")
        axes[1, 0].axvline(
            df["rel_error"].median() * 100,
            color="red",
            linestyle="--",
            label=f'Médiane: {df["rel_error"].median() * 100:.1f}%',
        )
        axes[1, 0].set_xlabel("Erreur Relative (%)")
        axes[1, 0].set_ylabel("Fréquence")
        axes[1, 0].set_title("Distribution des Erreurs")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Prédictions dans le temps
        axes[1, 1].plot(
            df["date"], df["actual"] * 100, label="Réel", alpha=0.8, linewidth=2
        )
        axes[1, 1].plot(
            df["date"], df["predicted"] * 100, label="Prédit", alpha=0.8, linewidth=2
        )
        axes[1, 1].set_xlabel("Date")
        axes[1, 1].set_ylabel("Taux de Mortalité (%)")
        axes[1, 1].set_title("Prédictions vs Réalité dans le Temps")
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder
        plot_path = os.path.join(
            STATIC_DIR, f'validation_mortalite_{location.lower().replace(" ", "_")}.png'
        )
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📈 Graphique sauvegardé: {plot_path}")

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du graphique: {e}")


def validate_multiple_locations(
    locations, model, scaler, feature_cols, max_locations=10
):
    """Valide le modèle sur plusieurs localisations"""
    print(f"\n🌍 VALIDATION SUR PLUSIEURS LOCALISATIONS")
    print("=" * 60)

    all_results = []
    location_summaries = []

    for i, location in enumerate(locations[:max_locations]):
        print(f"\n({i + 1}/{min(len(locations), max_locations)}) {location}")

        results = validate_single_location(location, model, scaler, feature_cols)

        if results:
            all_results.extend(results)

            # Métriques pour cette localisation
            df = pd.DataFrame(results)
            mae = mean_absolute_error(df["actual"], df["predicted"])
            r2 = r2_score(df["actual"], df["predicted"])
            median_error = df["rel_error"].median()

            location_summaries.append(
                {
                    "Localisation": location,
                    "Prédictions": len(results),
                    "MAE": mae,
                    "R²": r2,
                    "Erreur_Médiane_%": median_error * 100,
                    "Qualité": (
                        "Excellente"
                        if median_error < 0.15
                        else (
                            "Bonne"
                            if median_error < 0.30
                            else "Correcte" if median_error < 0.50 else "Mauvaise"
                        )
                    ),
                }
            )

            print(f"  ✅ MAE: {mae:.4f}, R²: {r2:.3f}")

            # Sauvegarder le graphique
            save_validation_plots(results, location)

    # Résultats globaux
    if all_results:
        print(f"\n📊 RÉSULTATS GLOBAUX")
        print("=" * 60)

        overall_df = pd.DataFrame(all_results)
        overall_mae = mean_absolute_error(overall_df["actual"], overall_df["predicted"])
        overall_r2 = r2_score(overall_df["actual"], overall_df["predicted"])

        print(f"Total des prédictions: {len(all_results)}")
        print(f"Localisations testées: {len(location_summaries)}")
        print(f"MAE globale: {overall_mae:.4f}")
        print(f"R² global: {overall_r2:.4f}")

        # Tableau de comparaison
        if location_summaries:
            summary_df = pd.DataFrame(location_summaries)
            print(f"\n📋 COMPARAISON PAR LOCALISATION:")
            print(summary_df.to_string(index=False, float_format="%.3f"))

            # Top performers
            top_locations = summary_df.nlargest(5, "R²")
            print(f"\n🏆 MEILLEURES PERFORMANCES:")
            for _, row in top_locations.iterrows():
                print(
                    f"  {row['Localisation']}: R² = {row['R²']:.3f}, Erreur = {row['Erreur_Médiane_%']:.1f}%"
                )


def main():
    """Fonction principale de validation"""
    print("🦠 VALIDATION DU MODÈLE DE MORTALITÉ COVID-19")
    print("=" * 60)
    print("📅 Période d'analyse: Mars 2020 - Mai 2022")
    print("🎯 Objectif: Tester la précision des prédictions sur données historiques")
    print("=" * 60)

    # Charger le modèle
    model, scaler, feature_cols = load_mortality_model()
    if model is None:
        print("❌ Impossible de charger le modèle. Validation arrêtée.")
        return

    # Récupérer les localisations disponibles
    locations = get_available_locations()
    if not locations:
        print("❌ Aucune localisation trouvée avec suffisamment de données.")
        return

    print(f"\n🎯 CHOIX DE VALIDATION:")
    print("1. Une seule localisation (France)")
    print("2. Top 5 localisations")
    print("3. Top 10 localisations")
    print("4. Localisation personnalisée")

    try:
        choice = input("\nVotre choix (1-4): ").strip()

        if choice == "1":
            # France uniquement
            results = validate_single_location("France", model, scaler, feature_cols)
            if results:
                print_validation_metrics(results, "France")
                save_validation_plots(results, "France")

        elif choice == "2":
            # Top 5
            validate_multiple_locations(
                locations, model, scaler, feature_cols, max_locations=5
            )

        elif choice == "3":
            # Top 10
            validate_multiple_locations(
                locations, model, scaler, feature_cols, max_locations=10
            )

        elif choice == "4":
            # Localisation personnalisée
            print(f"\nLocalisations disponibles: {', '.join(locations[:20])}...")
            custom_location = input("Entrez le nom de la localisation: ").strip()
            if custom_location in locations:
                results = validate_single_location(
                    custom_location, model, scaler, feature_cols
                )
                if results:
                    print_validation_metrics(results, custom_location)
                    save_validation_plots(results, custom_location)
            else:
                print(f"❌ Localisation '{custom_location}' non trouvée")

        else:
            print("❌ Choix invalide. Validation de la France par défaut.")
            results = validate_single_location("France", model, scaler, feature_cols)
            if results:
                print_validation_metrics(results, "France")
                save_validation_plots(results, "France")

    except KeyboardInterrupt:
        print("\n❌ Validation interrompue par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur pendant la validation: {e}")
        # Validation de France par défaut en cas d'erreur
        results = validate_single_location("France", model, scaler, feature_cols)
        if results:
            print_validation_metrics(results, "France")
            save_validation_plots(results, "France")

    print(f"\n✅ Validation terminée!")
    print(f"📁 Graphiques sauvegardés dans: {STATIC_DIR}")


if __name__ == "__main__":
    main()
