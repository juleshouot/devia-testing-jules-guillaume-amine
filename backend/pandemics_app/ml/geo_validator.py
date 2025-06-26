import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
STATIC_DIR = os.path.join(BASE_DIR, 'static', 'visualizations')


def get_engine():
    """Connexion à la base de données"""
    POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "guigui")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "pandemies")
    return create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        connect_args={"options": "-c search_path=pandemics"}
    )


def load_geographical_model():
    """Charge le modèle de propagation géographique entrainé"""
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"❌ Répertoire de modèles non trouvé: {MODEL_DIR}")
            return None, None, None

        # Chercher les fichiers de modèle de propagation géographique
        model_files = [f for f in os.listdir(MODEL_DIR) if 'geographical_spread' in f and f.endswith('_model.joblib')]

        if not model_files:
            print("❌ Aucun modèle de propagation géographique trouvé")
            return None, None, None

        model_file = model_files[0]
        print(f"✅ Chargement du modèle: {model_file}")

        # Charger le modèle
        model_path = os.path.join(MODEL_DIR, model_file)
        model = joblib.load(model_path)

        # Charger le scaler
        scaler_file = model_file.replace('_model.joblib', '_scaler.joblib')
        scaler_path = os.path.join(MODEL_DIR, scaler_file)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        # Charger les métadonnées
        metadata_file = model_file.replace('_model.joblib', '_metadata.joblib')
        metadata_path = os.path.join(MODEL_DIR, metadata_file)
        metadata = joblib.load(metadata_path) if os.path.exists(metadata_path) else {}

        feature_cols = metadata.get('feature_cols', [
            'lag1', 'lag2', 'lag3', 'lag4',
            'ma2', 'ma3', 'trend',
            'month_sin', 'month_cos'
        ])

        print(f"🔧 Features utilisées: {len(feature_cols)}")
        return model, scaler, feature_cols

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None, None


def get_geographical_spread_data():
    """Récupère les données de propagation géographique historiques"""
    try:
        engine = get_engine()

        # Récupérer les premières dates de cas positifs par localisation et virus
        first_cases_query = text("""
                                 SELECT l.name      AS location,
                                        v.name      AS virus,
                                        MIN(w.date) AS first_date
                                 FROM worldmeter w
                                          JOIN location l ON w.location_id = l.id
                                          JOIN virus v ON w.virus_id = v.id
                                 WHERE w.new_cases > 0
                                   AND w.date BETWEEN '2020-01-01' AND '2022-12-31'
                                 GROUP BY l.name, v.name
                                 ORDER BY first_date
                                 """)

        first_cases = pd.read_sql(first_cases_query, engine)

        if first_cases.empty:
            print("❌ Aucune donnée de propagation trouvée")
            return None

        # Convertir en datetime
        first_cases['first_date'] = pd.to_datetime(first_cases['first_date'])

        print(f"📊 {len(first_cases)} premières infections trouvées")
        print(f"Virus disponibles: {first_cases['virus'].unique()}")
        print(f"Période: {first_cases['first_date'].min()} → {first_cases['first_date'].max()}")

        return first_cases

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données: {e}")
        return None


def prepare_spread_time_series(first_cases_df, virus_name):
    """Prépare les séries temporelles de propagation pour un virus donné"""
    try:
        # Filtrer par virus
        virus_data = first_cases_df[first_cases_df['virus'] == virus_name].copy()

        if virus_data.empty:
            print(f"❌ Aucune donnée trouvée pour {virus_name}")
            return None

        # Ajouter des composantes temporelles
        virus_data['year'] = virus_data['first_date'].dt.year
        virus_data['week'] = virus_data['first_date'].dt.isocalendar().week
        virus_data['yearweek'] = virus_data['year'].astype(str) + '-' + virus_data['week'].astype(str).str.zfill(2)

        # Compter par semaine le nombre de nouvelles localisations touchées
        spread_data = virus_data.groupby('yearweek').size().reset_index(name='new_locations')

        # Convertir yearweek en date (milieu de la semaine)
        def yearweek_to_date(yearweek):
            year, week = yearweek.split('-')
            return pd.to_datetime(f"{year}-W{week}-3", format="%Y-W%W-%w")

        spread_data['date'] = spread_data['yearweek'].apply(yearweek_to_date)
        spread_data = spread_data.sort_values('date')

        # Créer une série temporelle complète (sans trous)
        date_range = pd.date_range(start=spread_data['date'].min(),
                                   end=spread_data['date'].max(),
                                   freq='W-WED')  # Mercredi de chaque semaine

        full_series = pd.DataFrame({'date': date_range})
        spread_data = full_series.merge(spread_data, on='date', how='left')
        spread_data['new_locations'] = spread_data['new_locations'].fillna(0)

        # Créer les features (comme dans l'entraînement)
        # Lags
        for i in range(1, 5):
            spread_data[f'lag{i}'] = spread_data['new_locations'].shift(i)

        # Moyennes mobiles
        spread_data['ma2'] = spread_data['new_locations'].rolling(2).mean()
        spread_data['ma3'] = spread_data['new_locations'].rolling(3).mean()

        # Tendance
        spread_data['trend'] = spread_data['ma2'] - spread_data['ma3'].shift(1)

        # Features saisonnières
        spread_data['month'] = spread_data['date'].dt.month
        spread_data['month_sin'] = np.sin(2 * np.pi * spread_data['month'] / 12)
        spread_data['month_cos'] = np.cos(2 * np.pi * spread_data['month'] / 12)

        # Nettoyer les données
        spread_data = spread_data.dropna()

        print(f"✅ Série temporelle créée pour {virus_name}: {len(spread_data)} semaines")
        print(f"Total de nouvelles localisations: {spread_data['new_locations'].sum()}")
        print(f"Pic de propagation: {spread_data['new_locations'].max()} nouvelles localisations en une semaine")

        return spread_data

    except Exception as e:
        print(f"❌ Erreur lors de la préparation des données pour {virus_name}: {e}")
        return None


def create_features_from_series(df, current_idx, feature_cols):
    """Crée les features pour prédiction à partir de la série temporelle"""
    try:
        if current_idx < 4:  # Besoin de 4 lags minimum
            return None

        # Utiliser seulement les données jusqu'à l'index actuel
        data = df.iloc[:current_idx + 1].copy()

        # Recalculer les features avec les données disponibles
        for i in range(1, 5):
            data[f'lag{i}'] = data['new_locations'].shift(i)

        data['ma2'] = data['new_locations'].rolling(2, min_periods=1).mean()
        data['ma3'] = data['new_locations'].rolling(3, min_periods=1).mean()
        data['trend'] = data['ma2'] - data['ma3'].shift(1)

        data['month'] = data['date'].dt.month
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

        # Récupérer les dernières valeurs
        latest = data.iloc[-1]

        features = {}
        for col in feature_cols:
            features[col] = latest.get(col, 0)

        # Nettoyer les valeurs NaN et inf
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                features[key] = 0

        return features

    except Exception:
        return None


def validate_virus_spread(virus_name, model, scaler, feature_cols, spread_data):
    """Valide le modèle de propagation géographique pour un virus"""
    print(f"\n🔍 Validation de la propagation pour {virus_name}...")

    if spread_data is None or len(spread_data) < 20:
        print(f"❌ Données insuffisantes pour {virus_name}")
        return None

    results = []

    # Parcourir la série temporelle et faire des prédictions
    for i in range(10, len(spread_data) - 1):  # Besoin d'historique

        # Créer les features avec les données jusqu'au point actuel
        features_dict = create_features_from_series(spread_data, i, feature_cols)
        if features_dict is None:
            continue

        # Faire la prédiction
        try:
            features_array = np.array([features_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
            if scaler is not None:
                features_array = scaler.transform(features_array)

            predicted_locations = model.predict(features_array)[0]
            predicted_locations = max(0, round(predicted_locations))  # Nombre entier positif

        except Exception:
            continue

        # Valeur réelle (semaine suivante)
        actual_locations = spread_data.iloc[i + 1]['new_locations']

        current_row = spread_data.iloc[i]
        results.append({
            'date': current_row['date'],
            'predicted': predicted_locations,
            'actual': actual_locations,
            'error': abs(predicted_locations - actual_locations),
            'rel_error': abs(predicted_locations - actual_locations) / max(actual_locations, 1),
            # Données de contexte
            'cumulative_locations': spread_data.iloc[:i + 1]['new_locations'].sum(),
            'recent_trend': spread_data.iloc[max(0, i - 3):i + 1]['new_locations'].mean(),
            'week_of_year': current_row['date'].isocalendar().week,
            'month': current_row['date'].month
        })

    if len(results) < 5:
        print(f"❌ Pas assez de prédictions valides pour {virus_name}")
        return None

    print(f"✅ {len(results)} prédictions générées")
    return results


def print_validation_metrics(results, virus_name):
    """Affiche les métriques de validation pour le modèle de propagation géographique"""
    df = pd.DataFrame(results)

    # Métriques de base
    mae = mean_absolute_error(df['actual'], df['predicted'])
    rmse = np.sqrt(mean_squared_error(df['actual'], df['predicted']))
    r2 = r2_score(df['actual'], df['predicted'])

    print(f"\n📊 RÉSULTATS DE VALIDATION DE PROPAGATION GÉOGRAPHIQUE POUR {virus_name.upper()}")
    print("=" * 80)
    print(f"Nombre de prédictions: {len(results)}")
    print(f"Période: {df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")

    print(f"\n🎯 MÉTRIQUES DE PRÉCISION:")
    print(f"Erreur Absolue Moyenne (MAE): {mae:.2f} nouvelles localisations")
    print(f"Erreur Quadratique Moyenne (RMSE): {rmse:.2f} nouvelles localisations")
    print(f"Score R²: {r2:.4f}")

    # Erreurs relatives
    median_rel_error = df['rel_error'].median()
    mean_rel_error = df['rel_error'].mean()

    print(f"\n📈 PERFORMANCE RELATIVE:")
    print(f"Erreur relative médiane: {median_rel_error:.1%}")
    print(f"Erreur relative moyenne: {mean_rel_error:.1%}")

    # Analyse de propagation
    avg_actual = df['actual'].mean()
    avg_predicted = df['predicted'].mean()
    max_spread_week = df['actual'].max()

    print(f"\n🌍 ANALYSE DE LA PROPAGATION:")
    print(f"Nouvelles localisations réelles/semaine (moyenne): {avg_actual:.1f}")
    print(f"Nouvelles localisations prédites/semaine (moyenne): {avg_predicted:.1f}")
    print(f"Pic de propagation: {max_spread_week} nouvelles localisations en une semaine")

    # Phases de propagation
    high_spread = (df['actual'] >= 5).sum()
    medium_spread = ((df['actual'] >= 2) & (df['actual'] < 5)).sum()
    low_spread = (df['actual'] < 2).sum()

    print(f"Semaines de forte propagation (≥5 lieux): {high_spread} ({high_spread / len(df) * 100:.1f}%)")
    print(f"Semaines de propagation modérée (2-4 lieux): {medium_spread} ({medium_spread / len(df) * 100:.1f}%)")
    print(f"Semaines de faible propagation (<2 lieux): {low_spread} ({low_spread / len(df) * 100:.1f}%)")

    # Distribution de qualité
    excellent = (df['rel_error'] < 0.20).sum()
    good = ((df['rel_error'] >= 0.20) & (df['rel_error'] < 0.40)).sum()
    fair = ((df['rel_error'] >= 0.40) & (df['rel_error'] < 0.70)).sum()
    poor = (df['rel_error'] >= 0.70).sum()

    print(f"\n🏆 QUALITÉ DES PRÉDICTIONS:")
    print(f"Excellentes (<20% erreur): {excellent} ({excellent / len(df) * 100:.1f}%)")
    print(f"Bonnes (20-40% erreur): {good} ({good / len(df) * 100:.1f}%)")
    print(f"Correctes (40-70% erreur): {fair} ({fair / len(df) * 100:.1f}%)")
    print(f"Mauvaises (>70% erreur): {poor} ({poor / len(df) * 100:.1f}%)")

    # Afficher les détails des prédictions
    show_details = input(f"\n📋 Afficher les détails des prédictions ? (o/n): ").strip().lower()
    if show_details in ['o', 'oui', 'y', 'yes']:
        show_prediction_details(df, virus_name)


def show_prediction_details(df, virus_name):
    """Affiche les détails des prédictions de propagation géographique"""
    print(f"\n📋 DÉTAILS DES PRÉDICTIONS DE PROPAGATION POUR {virus_name.upper()}")
    print("=" * 100)

    # Options d'affichage
    print("Options d'affichage:")
    print("1. Toutes les prédictions")
    print("2. Les 20 premières prédictions")
    print("3. Les 20 dernières prédictions")
    print("4. Semaines de forte propagation (≥5 nouveaux lieux)")
    print("5. Semaines de faible propagation (<2 nouveaux lieux)")
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
            display_df = df[df['actual'] >= 5].head(20)
            title = "SEMAINES DE FORTE PROPAGATION (≥5 lieux)"
        elif choice == "5":
            display_df = df[df['actual'] < 2].head(20)
            title = "SEMAINES DE FAIBLE PROPAGATION (<2 lieux)"
        elif choice == "6":
            year = input("Année (ex: 2020, 2021, 2022): ").strip()
            month = input("Mois (optionnel, ex: 03, 12): ").strip()

            filtered_df = df[df['date'].dt.year == int(year)]
            if month:
                filtered_df = filtered_df[filtered_df['date'].dt.month == int(month)]

            display_df = filtered_df
            title = f"PRÉDICTIONS {year}" + (f"-{month}" if month else "")
        else:
            display_df = df.head(20)
            title = "20 PREMIÈRES PRÉDICTIONS (par défaut)"

        if len(display_df) == 0:
            print("❌ Aucune prédiction trouvée pour cette sélection")
            return

        print(f"\n📊 {title}")
        print("=" * 130)
        print(
            f"{'Date':<12} {'Prédit':<8} {'Réel':<8} {'Erreur':<8} {'Erreur%':<8} {'Phase':<15} {'Cumul':<8} {'Tendance':<10} {'Semaine':<8} {'Mois':<6}")
        print("-" * 130)

        for _, row in display_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            predicted = f"{int(row['predicted'])}"
            actual = f"{int(row['actual'])}"
            error_abs = f"{int(row['error'])}"
            error_rel = f"{row['rel_error'] * 100:.1f}%"

            # Déterminer la phase de propagation
            if row['actual'] >= 5:
                phase = "Forte propagation"
            elif row['actual'] >= 2:
                phase = "Prop. modérée"
            else:
                phase = "Faible propagation"

            cumul = f"{int(row.get('cumulative_locations', 0))}"
            trend = f"{row.get('recent_trend', 0):.1f}"
            week = f"{int(row.get('week_of_year', 0))}"
            month = f"{int(row.get('month', 0))}"

            print(
                f"{date_str:<12} {predicted:<8} {actual:<8} {error_abs:<8} {error_rel:<8} {phase:<15} {cumul:<8} {trend:<10} {week:<8} {month:<6}")

        # Statistiques de la sélection
        print("\n📈 STATISTIQUES DE LA SÉLECTION:")
        print(f"Nombre de prédictions: {len(display_df)}")
        print(f"Nouvelles localisations réelles (moyenne): {display_df['actual'].mean():.1f}")
        print(f"Nouvelles localisations prédites (moyenne): {display_df['predicted'].mean():.1f}")
        print(f"Erreur moyenne: {display_df['rel_error'].mean() * 100:.1f}%")
        print(f"Erreur médiane: {display_df['rel_error'].median() * 100:.1f}%")

        # Analyse saisonnière
        if len(display_df) > 5:
            seasonal_analysis = display_df.groupby('month')['actual'].mean()
            print(f"\nAnalyse saisonnière (nouvelles localisations par mois):")
            for month, avg in seasonal_analysis.items():
                month_name = ['', 'Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Jun',
                              'Jul', 'Aoû', 'Sep', 'Oct', 'Nov', 'Déc'][month]
                print(f"  {month_name}: {avg:.1f} nouvelles localisations/semaine en moyenne")

        # Option pour sauvegarder
        save_csv = input(f"\n💾 Sauvegarder ces données en CSV ? (o/n): ").strip().lower()
        if save_csv in ['o', 'oui', 'y', 'yes']:
            save_prediction_details_csv(display_df, virus_name, title, "geographical")

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Affichage annulé")


def save_prediction_details_csv(df, virus_name, title, model_type):
    """Sauvegarde les détails des prédictions en CSV"""
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)

        # Préparer les données pour CSV
        csv_df = df.copy()
        csv_df['date'] = csv_df['date'].dt.strftime('%Y-%m-%d')
        csv_df['predicted_locations'] = csv_df['predicted'].astype(int)
        csv_df['actual_locations'] = csv_df['actual'].astype(int)
        csv_df['error_abs'] = csv_df['error'].astype(int)
        csv_df['rel_error_pct'] = (csv_df['rel_error'] * 100).round(1)

        # Ajouter colonnes d'analyse
        csv_df['spread_phase'] = csv_df['actual'].apply(lambda x:
                                                        'Forte propagation' if x >= 5 else
                                                        'Propagation modérée' if x >= 2 else
                                                        'Faible propagation'
                                                        )

        # Nom du fichier
        safe_title = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('≥', 'gte')
        safe_virus = virus_name.lower().replace(' ', '_')
        filename = f"predictions_{model_type}_{safe_virus}_{safe_title}.csv"
        filepath = os.path.join(STATIC_DIR, filename)

        # Sélectionner les colonnes finales
        final_df = csv_df[
            ['date', 'predicted_locations', 'actual_locations', 'error_abs', 'rel_error_pct', 'spread_phase',
             'cumulative_locations', 'recent_trend', 'week_of_year', 'month']]
        final_df.columns = ['Date', 'Nouvelles_Lieux_Prédit', 'Nouvelles_Lieux_Réel', 'Erreur_Absolue',
                            'Erreur_Relative_%', 'Phase_Propagation',
                            'Total_Cumulé', 'Tendance_Récente', 'Semaine_Année', 'Mois']

        # Sauvegarder
        final_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"💾 Données sauvegardées: {filepath}")

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde CSV: {e}")


def save_validation_plots(results, virus_name):
    """Sauvegarde les graphiques de validation pour le modèle de propagation géographique"""
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)

        df = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Validation du Modèle de Propagation Géographique: {virus_name}', fontsize=16)

        # 1. Prédit vs Réel
        axes[0, 0].scatter(df['actual'], df['predicted'], alpha=0.6, s=30)
        max_val = max(df['actual'].max(), df['predicted'].max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Prédiction Parfaite')
        axes[0, 0].set_xlabel('Nouvelles Localisations Réelles')
        axes[0, 0].set_ylabel('Nouvelles Localisations Prédites')
        axes[0, 0].set_title('Prédit vs Réel')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Évolution temporelle
        axes[0, 1].plot(df['date'], df['actual'], label='Réel', alpha=0.8, linewidth=2, marker='o', markersize=4)
        axes[0, 1].plot(df['date'], df['predicted'], label='Prédit', alpha=0.8, linewidth=2, marker='s', markersize=4)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Nouvelles Localisations')
        axes[0, 1].set_title('Évolution de la Propagation')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution des erreurs
        axes[1, 0].hist(df['rel_error'] * 100, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['rel_error'].median() * 100, color='red', linestyle='--',
                           label=f'Médiane: {df["rel_error"].median() * 100:.1f}%')
        axes[1, 0].set_xlabel('Erreur Relative (%)')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].set_title('Distribution des Erreurs')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Propagation cumulative
        df_sorted = df.sort_values('date')
        cumulative_actual = df_sorted['actual'].cumsum()
        cumulative_predicted = df_sorted['predicted'].cumsum()

        axes[1, 1].plot(df_sorted['date'], cumulative_actual, label='Réel Cumulé', alpha=0.8, linewidth=2)
        axes[1, 1].plot(df_sorted['date'], cumulative_predicted, label='Prédit Cumulé', alpha=0.8, linewidth=2)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Nouvelles Localisations (Cumulé)')
        axes[1, 1].set_title('Propagation Cumulative')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder
        plot_path = os.path.join(STATIC_DIR, f'validation_geographical_{virus_name.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Graphique sauvegardé: {plot_path}")

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du graphique: {e}")


def validate_multiple_viruses(first_cases_df, model, scaler, feature_cols):
    """Valide le modèle sur plusieurs virus"""
    print(f"\n🌍 VALIDATION DE PROPAGATION GÉOGRAPHIQUE SUR PLUSIEURS VIRUS")
    print("=" * 80)

    viruses = first_cases_df['virus'].unique()
    all_results = []
    virus_summaries = []

    for virus in viruses:
        print(f"\n🦠 Traitement de {virus}...")

        spread_data = prepare_spread_time_series(first_cases_df, virus)
        if spread_data is None:
            continue

        results = validate_virus_spread(virus, model, scaler, feature_cols, spread_data)

        if results:
            all_results.extend(results)

            # Métriques pour ce virus
            df = pd.DataFrame(results)
            mae = mean_absolute_error(df['actual'], df['predicted'])
            r2 = r2_score(df['actual'], df['predicted'])
            median_error = df['rel_error'].median()
            avg_spread = df['actual'].mean()

            virus_summaries.append({
                'Virus': virus,
                'Prédictions': len(results),
                'MAE': mae,
                'R²': r2,
                'Erreur_Médiane_%': median_error * 100,
                'Propagation_Moy': avg_spread,
                'Qualité': 'Excellente' if median_error < 0.20 else
                'Bonne' if median_error < 0.40 else
                'Correcte' if median_error < 0.70 else 'Mauvaise'
            })

            print(f"  ✅ MAE: {mae:.2f}, R²: {r2:.3f}, Propagation moy: {avg_spread:.1f}")

            # Sauvegarder le graphique
            save_validation_plots(results, virus)

    # Résultats globaux
    if all_results:
        print(f"\n📊 RÉSULTATS GLOBAUX DE PROPAGATION GÉOGRAPHIQUE")
        print("=" * 80)

        overall_df = pd.DataFrame(all_results)
        overall_mae = mean_absolute_error(overall_df['actual'], overall_df['predicted'])
        overall_r2 = r2_score(overall_df['actual'], overall_df['predicted'])

        print(f"Total des prédictions: {len(all_results)}")
        print(f"Virus testés: {len(virus_summaries)}")
        print(f"MAE globale: {overall_mae:.2f} nouvelles localisations")
        print(f"R² global: {overall_r2:.4f}")
        print(f"Propagation globale moyenne: {overall_df['actual'].mean():.1f} nouvelles lieux/semaine")

        # Tableau de comparaison
        if virus_summaries:
            summary_df = pd.DataFrame(virus_summaries)
            print(f"\n📋 COMPARAISON PAR VIRUS:")
            print(summary_df.to_string(index=False, float_format='%.3f'))


def main():
    """Fonction principale de validation du modèle de propagation géographique"""
    print("🦠 VALIDATION DU MODÈLE DE PROPAGATION GÉOGRAPHIQUE")
    print("=" * 70)
    print("📅 Période d'analyse: 2020-2022")
    print("🎯 Objectif: Tester la précision des prédictions de propagation vers de nouveaux lieux")
    print("=" * 70)

    # Charger le modèle
    model, scaler, feature_cols = load_geographical_model()
    if model is None:
        print("❌ Impossible de charger le modèle. Validation arrêtée.")
        return

    # Récupérer les données de propagation
    first_cases_df = get_geographical_spread_data()
    if first_cases_df is None:
        print("❌ Impossible de récupérer les données de propagation.")
        return

    viruses = first_cases_df['virus'].unique()
    print(f"\n🎯 CHOIX DE VALIDATION:")
    print("1. COVID-19 uniquement")
    print("2. Monkeypox uniquement")
    print("3. Tous les virus disponibles")
    print("4. Virus personnalisé")

    try:
        choice = input("\nVotre choix (1-4): ").strip()

        if choice == "1":
            # COVID uniquement
            spread_data = prepare_spread_time_series(first_cases_df, "COVID")
            if spread_data is not None:
                results = validate_virus_spread("COVID", model, scaler, feature_cols, spread_data)
                if results:
                    print_validation_metrics(results, "COVID")
                    save_validation_plots(results, "COVID")

        elif choice == "2":
            # Monkeypox uniquement
            spread_data = prepare_spread_time_series(first_cases_df, "Monkeypox")
            if spread_data is not None:
                results = validate_virus_spread("Monkeypox", model, scaler, feature_cols, spread_data)
                if results:
                    print_validation_metrics(results, "Monkeypox")
                    save_validation_plots(results, "Monkeypox")

        elif choice == "3":
            # Tous les virus
            validate_multiple_viruses(first_cases_df, model, scaler, feature_cols)

        elif choice == "4":
            # Virus personnalisé
            print(f"\nVirus disponibles: {', '.join(viruses)}")
            custom_virus = input("Entrez le nom du virus: ").strip()
            if custom_virus in viruses:
                spread_data = prepare_spread_time_series(first_cases_df, custom_virus)
                if spread_data is not None:
                    results = validate_virus_spread(custom_virus, model, scaler, feature_cols, spread_data)
                    if results:
                        print_validation_metrics(results, custom_virus)
                        save_validation_plots(results, custom_virus)
            else:
                print(f"❌ Virus '{custom_virus}' non trouvé")

        else:
            print("❌ Choix invalide. Validation COVID par défaut.")
            spread_data = prepare_spread_time_series(first_cases_df, "COVID")
            if spread_data is not None:
                results = validate_virus_spread("COVID", model, scaler, feature_cols, spread_data)
                if results:
                    print_validation_metrics(results, "COVID")
                    save_validation_plots(results, "COVID")

    except KeyboardInterrupt:
        print("\n❌ Validation interrompue par l'utilisateur")
    except Exception as e:
        print(f"❌ Erreur pendant la validation: {e}")
        # Validation COVID par défaut en cas d'erreur
        spread_data = prepare_spread_time_series(first_cases_df, "COVID")
        if spread_data is not None:
            results = validate_virus_spread("COVID", model, scaler, feature_cols, spread_data)
            if results:
                print_validation_metrics(results, "COVID")
                save_validation_plots(results, "COVID")

    print(f"\n✅ Validation de propagation géographique terminée!")
    print(f"📁 Graphiques sauvegardés dans: {STATIC_DIR}")


if __name__ == "__main__":
    main()