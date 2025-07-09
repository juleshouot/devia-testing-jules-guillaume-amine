import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings

warnings.filterwarnings("ignore")

# Configuration des chemins
BASE_DIR = "/django_api"  # Pointe vers votre répertoire ./backend monté
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PLOTS_DIR = os.path.join(BASE_DIR, "reports", "figures")

# Création des répertoires nécessaires
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Style pour les graphiques
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 12


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


def load_data():
    """Chargement des données depuis la base de données"""
    engine = get_engine()

    print("Chargement des données...")
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

    # Conversion de la date
    if not pd.api.types.is_datetime64_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # Séparation par virus
    covid_df = df[df["virus"] == "COVID"].copy()
    monkeypox_df = df[df["virus"] == "Monkeypox"].copy()

    print(
        f"Données chargées: {len(df)} entrées, {df['location'].nunique()} localisations, {df['virus'].nunique()} virus"
    )
    return df, covid_df, monkeypox_df


def analyze_case_trends(df):
    """Analyse des tendances des cas"""
    print("\n--- Analyse des tendances des cas ---")

    # Agrégation des cas par virus et date
    global_trends = (
        df.groupby(["virus", "date"])
        .agg(
            {
                "new_cases": "sum",
                "total_cases": "sum",
                "new_deaths": "sum",
                "total_deaths": "sum",
            }
        )
        .reset_index()
    )

    # Graphique des cas quotidiens
    plt.figure(figsize=(14, 8))
    for virus, group in global_trends.groupby("virus"):
        plt.plot(group["date"], group["new_cases"], label=f"{virus} - Nouveaux cas")

    plt.title("Évolution des nouveaux cas par jour")
    plt.xlabel("Date")
    plt.ylabel("Nouveaux cas")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "new_cases_trends.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    # Graphique des cas cumulés
    plt.figure(figsize=(14, 8))
    for virus, group in global_trends.groupby("virus"):
        plt.plot(group["date"], group["total_cases"], label=f"{virus} - Cas cumulés")

    plt.title("Évolution des cas cumulés")
    plt.xlabel("Date")
    plt.ylabel("Cas cumulés")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "total_cases_trends.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    # Calcul des moyennes mobiles sur 7 jours
    for virus, group in global_trends.groupby("virus"):
        group_sorted = group.sort_values("date")
        group_sorted["new_cases_ma7"] = (
            group_sorted["new_cases"].rolling(window=7).mean()
        )
        group_sorted["new_deaths_ma7"] = (
            group_sorted["new_deaths"].rolling(window=7).mean()
        )

        # Graphique des moyennes mobiles
        plt.figure(figsize=(14, 8))
        plt.plot(
            group_sorted["date"],
            group_sorted["new_cases_ma7"],
            label="Nouveaux cas (moyenne 7j)",
        )
        plt.plot(
            group_sorted["date"],
            group_sorted["new_deaths_ma7"],
            label="Nouveaux décès (moyenne 7j)",
        )

        plt.title(f"Évolution des cas et décès de {virus} (moyenne mobile 7 jours)")
        plt.xlabel("Date")
        plt.ylabel("Nombre")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Sauvegarde du graphique
        output_path = os.path.join(PLOTS_DIR, f"{virus}_moving_averages.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Graphique sauvegardé: {output_path}")

    return global_trends


def analyze_rt_distribution(df):
    """Analyse de la distribution des valeurs Rt (taux de reproduction)"""
    print("\n--- Analyse de la distribution Rt ---")

    rt_data = []

    # Calcul de Rt pour chaque localisation et virus
    for (location, virus), group in df.groupby(["location", "virus"]):
        if len(group) < 21:  # Besoin d'au moins 3 semaines de données
            continue

        temp_df = group.copy().sort_values("date")

        # Moyennes mobiles
        temp_df["cases_ma7"] = (
            temp_df["new_cases"].rolling(window=7, min_periods=1).mean()
        )

        # Décalage pour Rt (7 jours d'incubation)
        temp_df["previous_cases_ma7"] = temp_df["cases_ma7"].shift(7)

        # Calcul de Rt
        temp_df["rt"] = (
            temp_df["cases_ma7"] / temp_df["previous_cases_ma7"].replace(0, 0.1)
        ).clip(0, 10)

        # Ajouter les données calculées
        rt_data.append(temp_df[["date", "location", "virus", "rt"]].dropna())

    if not rt_data:
        print("Pas assez de données pour calculer Rt")
        return None

    rt_df = pd.concat(rt_data)

    # Sauvegarder les données
    rt_df.to_csv(os.path.join(DATA_DIR, "rt_analysis.csv"), index=False)

    # Filtrer les valeurs extrêmes pour la visualisation
    rt_viz = rt_df[rt_df["rt"] < 5].copy()

    # Distribution des valeurs Rt par virus
    plt.figure(figsize=(12, 8))
    for virus, group in rt_viz.groupby("virus"):
        sns.kdeplot(group["rt"], label=f"{virus} (n={len(group)})")

    plt.axvline(
        x=1, color="red", linestyle="--", alpha=0.7, label="Rt = 1 (seuil épidémique)"
    )
    plt.title("Distribution des valeurs Rt par virus")
    plt.xlabel("Rt (Taux de reproduction effectif)")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "rt_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    # Évolution temporelle de Rt
    plt.figure(figsize=(14, 8))
    for virus, group in rt_viz.groupby("virus"):
        # Calculer la moyenne de Rt pour toutes les localisations par jour
        daily_rt = group.groupby("date")["rt"].mean()
        plt.plot(daily_rt.index, daily_rt.values, label=virus)

    plt.axhline(y=1, color="red", linestyle="--", alpha=0.7)
    plt.title("Évolution temporelle du Rt moyen")
    plt.xlabel("Date")
    plt.ylabel("Rt moyen")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "rt_temporal_evolution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    return rt_df


def analyze_mortality(df):
    """Analyse du taux de mortalité"""
    print("\n--- Analyse du taux de mortalité ---")

    mortality_data = []

    # Calcul du taux de mortalité pour chaque localisation et virus
    for (location, virus), group in df.groupby(["location", "virus"]):
        if len(group) < 28:  # Besoin d'au moins 4 semaines de données
            continue

        temp_df = group.copy().sort_values("date")

        # Sommes glissantes sur 7 jours
        temp_df["cases_7day"] = temp_df["new_cases"].rolling(7).sum().fillna(0)
        temp_df["deaths_7day"] = temp_df["new_deaths"].rolling(7).sum().fillna(0)

        # Décalage pour le délai entre cas et décès (14 jours)
        temp_df["cases_7day_lag"] = temp_df["cases_7day"].shift(14)

        # Calcul du taux de mortalité
        temp_df["mortality_ratio"] = (
            temp_df["deaths_7day"] / temp_df["cases_7day_lag"].replace(0, 1)
        ).clip(0, 1)

        # Ajouter les données calculées
        mortality_data.append(
            temp_df[["date", "location", "virus", "mortality_ratio"]].dropna()
        )

    if not mortality_data:
        print("Pas assez de données pour calculer le taux de mortalité")
        return None

    mortality_df = pd.concat(mortality_data)

    # Sauvegarder les données
    mortality_df.to_csv(os.path.join(DATA_DIR, "mortality_analysis.csv"), index=False)

    # Distribution du taux de mortalité par virus
    plt.figure(figsize=(12, 8))
    for virus, group in mortality_df.groupby("virus"):
        sns.kdeplot(group["mortality_ratio"], label=f"{virus} (n={len(group)})")

    plt.title("Distribution du taux de mortalité par virus")
    plt.xlabel("Taux de mortalité")
    plt.ylabel("Densité")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "mortality_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    # Évolution temporelle du taux de mortalité
    plt.figure(figsize=(14, 8))
    for virus, group in mortality_df.groupby("virus"):
        # Calculer la moyenne du taux de mortalité pour toutes les localisations par jour
        daily_mortality = group.groupby("date")["mortality_ratio"].mean()
        # Appliquer une moyenne mobile sur 30 jours pour lisser
        smoothed_mortality = daily_mortality.rolling(window=30, min_periods=7).mean()
        plt.plot(smoothed_mortality.index, smoothed_mortality.values, label=virus)

    plt.title("Évolution temporelle du taux de mortalité moyen (lissé)")
    plt.xlabel("Date")
    plt.ylabel("Taux de mortalité moyen")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "mortality_temporal_evolution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    return mortality_df


def analyze_geographical_spread(df):
    """Analyse de la propagation géographique"""
    print("\n--- Analyse de la propagation géographique ---")

    # Trouver la première date d'apparition par location et virus
    first_cases = (
        df[df["new_cases"] > 0]
        .groupby(["location", "virus"])["date"]
        .min()
        .reset_index()
    )

    # Ajouter des composantes temporelles
    first_cases["year"] = first_cases["date"].dt.year
    first_cases["week"] = first_cases["date"].dt.isocalendar().week
    first_cases["month"] = first_cases["date"].dt.month
    first_cases["yearweek"] = (
        first_cases["year"].astype(str)
        + "-"
        + first_cases["week"].astype(str).str.zfill(2)
    )

    # Compter les nouvelles localisations par semaine
    weekly_spread = (
        first_cases.groupby(["virus", "yearweek"])
        .size()
        .reset_index(name="new_locations")
    )

    # Convertir yearweek en date
    def yearweek_to_date(yearweek):
        year, week = yearweek.split("-")
        # Date du mercredi de la semaine
        return pd.to_datetime(f"{year}-W{week}-3", format="%Y-W%W-%w")

    weekly_spread["date"] = weekly_spread["yearweek"].apply(yearweek_to_date)
    weekly_spread = weekly_spread.sort_values(["virus", "date"])

    # Sauvegarder les données
    weekly_spread.to_csv(
        os.path.join(DATA_DIR, "geographical_spread_analysis.csv"), index=False
    )

    # Graphique des nouvelles localisations par semaine
    plt.figure(figsize=(14, 8))
    for virus, group in weekly_spread.groupby("virus"):
        plt.bar(group["date"], group["new_locations"], label=virus, alpha=0.7)

    plt.title("Nombre de nouvelles localisations touchées par semaine")
    plt.xlabel("Date")
    plt.ylabel("Nouvelles localisations")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "geographical_spread_weekly.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    # Propagation cumulative
    weekly_spread["cumulative_locations"] = weekly_spread.groupby("virus")[
        "new_locations"
    ].cumsum()

    plt.figure(figsize=(14, 8))
    for virus, group in weekly_spread.groupby("virus"):
        plt.plot(group["date"], group["cumulative_locations"], label=virus, marker="o")

    plt.title("Nombre cumulatif de localisations touchées")
    plt.xlabel("Date")
    plt.ylabel("Localisations cumulées")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "geographical_spread_cumulative.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    return weekly_spread, first_cases


def evaluate_models():
    """Évaluation des modèles entraînés"""
    print("\n--- Évaluation des modèles ---")

    models_dir = os.path.join(os.path.dirname(__file__), "models")

    if not os.path.exists(models_dir):
        print("Répertoire des modèles non trouvé")
        return

    model_files = [f for f in os.listdir(models_dir) if f.endswith("_metadata.joblib")]

    if not model_files:
        print("Aucun modèle trouvé")
        return

    # Récupérer les métadonnées des modèles
    model_metrics = []

    for file in model_files:
        try:
            metadata_path = os.path.join(models_dir, file)
            metadata = joblib.load(metadata_path)

            model_type = metadata.get("model_type", "unknown")
            model_name = metadata.get("model_name", "unknown")

            metrics = {
                "model_type": model_type,
                "model_name": model_name,
                "r2_score": metadata.get("r2_score", float("nan")),
                "rmse": metadata.get("rmse", float("nan")),
                "mae": metadata.get("mae", float("nan")),
                "cv_rmse": metadata.get("cv_rmse", float("nan")),
            }

            model_metrics.append(metrics)
        except Exception as e:
            print(f"Erreur lors du chargement des métadonnées {file}: {e}")

    if not model_metrics:
        print("Aucune métrique de modèle trouvée")
        return

    # Afficher les métriques
    df_metrics = pd.DataFrame(model_metrics)
    print("\nMétriques des modèles:")
    print(df_metrics)

    # Visualiser les métriques
    plt.figure(figsize=(12, 8))

    # Regrouper par type de modèle
    for model_type, group in df_metrics.groupby("model_type"):
        plt.subplot(
            1,
            len(df_metrics["model_type"].unique()),
            list(df_metrics["model_type"].unique()).index(model_type) + 1,
        )

        plt.bar(group["model_name"], group["rmse"])
        plt.title(f"RMSE - {model_type}")
        plt.ylabel("RMSE (erreur)")
        plt.xticks(rotation=45)
        plt.tight_layout()

    # Sauvegarde du graphique
    output_path = os.path.join(PLOTS_DIR, "model_metrics_comparison.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Graphique sauvegardé: {output_path}")

    # Sauvegarder les métriques
    df_metrics.to_csv(os.path.join(DATA_DIR, "model_metrics.csv"), index=False)

    return df_metrics


def generate_summary_report(
    global_trends, rt_df, mortality_df, spread_data, model_metrics
):
    """Génère un rapport de synthèse au format Markdown"""
    print("\n--- Génération du rapport de synthèse ---")

    # Statistiques générales
    stats = {}

    if global_trends is not None:
        for virus, group in global_trends.groupby("virus"):
            total_cases = group["total_cases"].max()
            total_deaths = group["total_deaths"].max()
            max_daily_cases = group["new_cases"].max()
            max_daily_deaths = group["new_deaths"].max()

            stats[virus] = {
                "total_cases": total_cases,
                "total_deaths": total_deaths,
                "max_daily_cases": max_daily_cases,
                "max_daily_deaths": max_daily_deaths,
                "case_fatality_rate": (
                    (total_deaths / total_cases) if total_cases > 0 else 0
                ),
            }

    # Statistiques Rt
    rt_stats = {}
    if rt_df is not None:
        for virus, group in rt_df.groupby("virus"):
            rt_stats[virus] = {
                "mean_rt": group["rt"].mean(),
                "median_rt": group["rt"].median(),
                "min_rt": group["rt"].min(),
                "max_rt": group["rt"].max(),
                "above_1_percent": (group["rt"] > 1).mean() * 100,
            }

    # Statistiques mortalité
    mortality_stats = {}
    if mortality_df is not None:
        for virus, group in mortality_df.groupby("virus"):
            mortality_stats[virus] = {
                "mean_cfr": group["mortality_ratio"].mean(),
                "median_cfr": group["mortality_ratio"].median(),
                "min_cfr": group["mortality_ratio"].min(),
                "max_cfr": group["mortality_ratio"].max(),
            }

    # Génération du rapport
    report = f"""# Rapport d'Analyse des Pandémies
*Généré le {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*

## 1. Statistiques Globales

"""

    for virus, stat in stats.items():
        report += f"""### {virus}
- **Cas totaux**: {stat['total_cases']:,.0f}
- **Décès totaux**: {stat['total_deaths']:,.0f}
- **Pic quotidien de cas**: {stat['max_daily_cases']:,.0f}
- **Pic quotidien de décès**: {stat['max_daily_deaths']:,.0f}
- **Taux de létalité global**: {stat['case_fatality_rate'] * 100:.2f}%

"""

    report += """## 2. Analyse du Taux de Reproduction (Rt)

Le taux de reproduction effectif (Rt) est un indicateur clé qui mesure combien de personnes sont infectées par un individu contagieux.

"""

    for virus, stat in rt_stats.items():
        report += f"""### {virus}
- **Rt moyen**: {stat['mean_rt']:.2f}
- **Rt médian**: {stat['median_rt']:.2f}
- **Rt minimum**: {stat['min_rt']:.2f}
- **Rt maximum**: {stat['max_rt']:.2f}
- **Pourcentage du temps avec Rt > 1**: {stat['above_1_percent']:.1f}%

"""

    report += """## 3. Analyse du Taux de Mortalité

Le taux de mortalité (Case Fatality Rate) mesure la proportion de cas confirmés qui aboutissent à un décès.

"""

    for virus, stat in mortality_stats.items():
        report += f"""### {virus}
- **CFR moyen**: {stat['mean_cfr'] * 100:.2f}%
- **CFR médian**: {stat['median_cfr'] * 100:.2f}%
- **CFR minimum**: {stat['min_cfr'] * 100:.2f}%
- **CFR maximum**: {stat['max_cfr'] * 100:.2f}%

"""

    report += """## 4. Modèles Prédictifs

Performances des modèles entraînés:

"""

    if model_metrics is not None:
        for _, row in model_metrics.iterrows():
            report += f"""### {row['model_type']} - {row['model_name']}
- **R²**: {row.get('r2_score', 'N/A')}
- **RMSE**: {row.get('rmse', 'N/A')}
- **MAE**: {row.get('mae', 'N/A')}
- **CV RMSE**: {row.get('cv_rmse', 'N/A')}

"""

    report += """## 5. Visualisations Générées

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
"""

    # Écrire le rapport dans un fichier
    report_path = os.path.join(BASE_DIR, "reports", "data_analysis_summary.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Rapport de synthèse sauvegardé: {report_path}")

    return report


def main():
    """Fonction principale pour l'exploration et l'analyse des données"""
    print("=== Début de l'exploration des données ===")

    # Chargement des données
    df, covid_df, monkeypox_df = load_data()

    # Analyses
    global_trends = analyze_case_trends(df)
    rt_df = analyze_rt_distribution(df)
    mortality_df = analyze_mortality(df)
    spread_data, first_cases = analyze_geographical_spread(df)
    model_metrics = evaluate_models()

    # Génération du rapport
    generate_summary_report(
        global_trends, rt_df, mortality_df, spread_data, model_metrics
    )

    print("\n=== Exploration des données terminée ===")
    print(f"Tous les graphiques ont été sauvegardés dans: {PLOTS_DIR}")
    print(f"Toutes les données préparées ont été sauvegardées dans: {DATA_DIR}")


if __name__ == "__main__":
    main()
