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


def load_transmission_model():
    """Charge le modèle de transmission entrainé"""
    try:
        if not os.path.exists(MODEL_DIR):
            print(f"❌ Répertoire de modèles non trouvé: {MODEL_DIR}")
            return None, None, None

        # Chercher les fichiers de modèle de transmission
        model_files = [f for f in os.listdir(MODEL_DIR) if 'transmission' in f and f.endswith('_model.joblib')]

        if not model_files:
            print("❌ Aucun modèle de transmission trouvé")
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
            'cases_ma7', 'deaths_ma7', 'previous_cases_ma7',
            'cases_growth', 'deaths_growth', 'cases_acceleration',
            'days_since_start', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'total_cases', 'total_deaths',
            'new_cases_per_million', 'total_cases_per_million',
            'new_deaths_per_million', 'total_deaths_per_million'
        ])

        print(f"🔧 Features utilisées: {len(feature_cols)}")
        return model, scaler, feature_cols

    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle: {e}")
        return None, None, None


def get_available_locations():
    """Récupère les localisations avec suffisamment de données COVID"""
    try:
        engine = get_engine()
        query = text("""
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
                     """)

        df = pd.read_sql(query, engine)
        print(f"📍 {len(df)} localisations trouvées avec suffisamment de données")
        print(df[['location', 'data_points', 'total_cases', 'total_deaths']].head(10).to_string(index=False))

        return df['location'].tolist()

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des localisations: {e}")
        return []


def get_covid_data(location, start_date='2020-03-01', end_date='2022-05-01'):
    """Récupère les données COVID pour une localisation"""
    try:
        engine = get_engine()
        query = text("""
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
                     """)

        df = pd.read_sql(query, engine, params={
            "location": location,
            "start_date": start_date,
            "end_date": end_date
        })

        if df.empty:
            print(f"❌ Aucune donnée trouvée pour {location}")
            return None

        df['date'] = pd.to_datetime(df['date'])
        print(f"📊 {len(df)} lignes de données récupérées pour {location}")

        return df

    except Exception as e:
        print(f"❌ Erreur lors de la récupération des données pour {location}: {e}")
        return None


def create_features_from_data(df, current_idx):
    """Crée les features pour prédiction de transmission (mêmes que l'entrainement)"""
    try:
        # Utiliser seulement les données jusqu'à l'index actuel
        data = df.iloc[:current_idx + 1].copy()

        # Calcul des moyennes mobiles sur 7 jours
        data['cases_ma7'] = data['new_cases'].rolling(window=7, min_periods=1).mean()
        data['deaths_ma7'] = data['new_deaths'].rolling(window=7, min_periods=1).mean()

        # Décalage pour calculer Rt
        incubation_period = 7
        data['previous_cases_ma7'] = data['cases_ma7'].shift(incubation_period)

        # Features supplémentaires
        data['cases_growth'] = data['cases_ma7'].pct_change(7).fillna(0).clip(-1, 2)
        data['deaths_growth'] = data['deaths_ma7'].pct_change(7).fillna(0).clip(-1, 2)
        data['days_since_start'] = (data['date'] - data['date'].min()).dt.days
        data['cases_acceleration'] = data['cases_growth'].diff().fillna(0)

        # Features cycliques
        data['day_sin'] = np.sin(2 * np.pi * data['date'].dt.dayofweek / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['date'].dt.dayofweek / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['date'].dt.month / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['date'].dt.month / 12)

        # Récupérer les dernières valeurs
        latest = data.iloc[-1]

        features = {
            'cases_ma7': latest.get('cases_ma7', 0),
            'deaths_ma7': latest.get('deaths_ma7', 0),
            'previous_cases_ma7': latest.get('previous_cases_ma7', 0),
            'cases_growth': latest.get('cases_growth', 0),
            'deaths_growth': latest.get('deaths_growth', 0),
            'cases_acceleration': latest.get('cases_acceleration', 0),
            'days_since_start': latest.get('days_since_start', 0),
            'day_sin': latest.get('day_sin', 0),
            'day_cos': latest.get('day_cos', 0),
            'month_sin': latest.get('month_sin', 0),
            'month_cos': latest.get('month_cos', 0),
            'total_cases': latest.get('total_cases', 0),
            'total_deaths': latest.get('total_deaths', 0),
            'new_cases_per_million': latest.get('new_cases_per_million', 0),
            'total_cases_per_million': latest.get('total_cases_per_million', 0),
            'new_deaths_per_million': latest.get('new_deaths_per_million', 0),
            'total_deaths_per_million': latest.get('total_deaths_per_million', 0)
        }

        # Nettoyer les valeurs NaN et inf
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                features[key] = 0

        return features

    except Exception:
        return None


def calculate_actual_rt(df, current_idx):
    """Calcule le taux de reproduction effectif (Rt) réel en utilisant les données FUTURES"""
    try:
        # Prédire le Rt qui sera observé dans 7-14 jours
        future_start = current_idx + 7
        future_end = current_idx + 14

        if future_end >= len(df):
            return None

        # Cas futurs (moyenne mobile 7 jours dans 7-14 jours)
        future_cases = df.iloc[future_start:future_end]['new_cases'].mean()

        # Cas actuels (moyenne mobile 7 jours)
        if current_idx >= 6:
            current_cases = df.iloc[current_idx - 6:current_idx + 1]['new_cases'].mean()
        else:
            return None

        if current_cases > 0.1:  # Seuil minimum
            rt = future_cases / current_cases
            return min(max(rt, 0), 10)  # Limiter entre 0 et 10

        return None

    except Exception:
        return None


def validate_single_location(location, model, scaler, feature_cols):
    """Valide le modèle de transmission pour une seule localisation"""
    print(f"\n🔍 Validation pour {location}...")

    # Récupérer les données
    df = get_covid_data(location)
    if df is None or len(df) < 50:
        print(f"❌ Données insuffisantes pour {location}")
        return None

    results = []

    # Parcourir le temps et faire des prédictions
    for i in range(30, len(df) - 14):  # Besoin d'historique et de données futures pour validation

        # Créer les features avec les données jusqu'au point actuel
        features_dict = create_features_from_data(df, i)
        if features_dict is None:
            continue

        # Faire la prédiction du Rt
        try:
            features_array = np.array([features_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
            if scaler is not None:
                features_array = scaler.transform(features_array)

            predicted_rt = model.predict(features_array)[0]
            predicted_rt = max(0, min(predicted_rt, 10))  # Limiter à [0,10]

            # DEBUG: Afficher quelques prédictions pour vérifier
            if i < 35:  # Premières prédictions seulement
                print(f"DEBUG {i}: Features sample: cases_ma7={features_dict.get('cases_ma7', 0):.2f}, "
                      f"previous_cases_ma7={features_dict.get('previous_cases_ma7', 0):.2f}, "
                      f"predicted_rt={predicted_rt:.3f}")

        except Exception as e:
            print(f"Erreur de prédiction à l'index {i}: {e}")
            continue

        # Calculer le Rt réel qui s'est produit dans les 7-14 jours suivants
        actual_rt = calculate_actual_rt(df, i)

        if actual_rt is not None:
            # DEBUG: Afficher quelques comparaisons
            if i < 35:
                print(f"DEBUG {i}: actual_rt={actual_rt:.3f}, predicted_rt={predicted_rt:.3f}, "
                      f"error={abs(predicted_rt - actual_rt):.3f}")

            # Ajouter les données de contexte
            current_row = df.iloc[i]
            results.append({
                'date': current_row['date'],
                'predicted': predicted_rt,
                'actual': actual_rt,
                'error': abs(predicted_rt - actual_rt),
                'rel_error': abs(predicted_rt - actual_rt) / max(actual_rt, 0.1),
                # Données de contexte
                'total_cases': current_row.get('total_cases', 0),
                'total_deaths': current_row.get('total_deaths', 0),
                'new_cases_7d': df.iloc[max(0, i - 6):i + 1]['new_cases'].sum(),
                'new_deaths_7d': df.iloc[max(0, i - 6):i + 1]['new_deaths'].sum(),
                'cases_per_million': current_row.get('total_cases_per_million', 0)
            })

    if len(results) < 10:
        print(f"❌ Pas assez de prédictions valides pour {location}")
        return None

    print(f"✅ {len(results)} prédictions générées pour prédire Rt 7-14 jours à l'avance")
    return results


def print_validation_metrics(results, location):
    """Affiche les métriques de validation pour le modèle de transmission"""
    df = pd.DataFrame(results)

    # Métriques de base
    mae = mean_absolute_error(df['actual'], df['predicted'])
    rmse = np.sqrt(mean_squared_error(df['actual'], df['predicted']))
    r2 = r2_score(df['actual'], df['predicted'])

    print(f"\n📊 RÉSULTATS DE VALIDATION DE TRANSMISSION POUR {location.upper()}")
    print("=" * 80)
    print(f"Nombre de prédictions: {len(results)}")
    print(f"Période: {df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"⚠️  TYPE DE PRÉDICTION: Rt futur (7-14 jours à l'avance)")

    print(f"\n🎯 MÉTRIQUES DE PRÉCISION:")
    print(f"Erreur Absolue Moyenne (MAE): {mae:.3f}")
    print(f"Erreur Quadratique Moyenne (RMSE): {rmse:.3f}")
    print(f"Score R²: {r2:.4f}")

    # Erreurs relatives
    median_rel_error = df['rel_error'].median()
    mean_rel_error = df['rel_error'].mean()

    print(f"\n📈 PERFORMANCE RELATIVE:")
    print(f"Erreur relative médiane: {median_rel_error:.1%}")
    print(f"Erreur relative moyenne: {mean_rel_error:.1%}")

    # Analyse de Rt
    avg_actual_rt = df['actual'].mean()
    avg_predicted_rt = df['predicted'].mean()

    print(f"\n🦠 ANALYSE DU TAUX DE REPRODUCTION (Rt):")
    print(f"Rt futur réel moyen: {avg_actual_rt:.2f}")
    print(f"Rt futur prédit moyen: {avg_predicted_rt:.2f}")

    # Périodes épidémiques vs contrôlées
    epidemic_periods = (df['actual'] > 1).sum()
    controlled_periods = (df['actual'] <= 1).sum()

    print(f"Périodes épidémiques futures (Rt > 1): {epidemic_periods} ({epidemic_periods / len(df) * 100:.1f}%)")
    print(f"Périodes contrôlées futures (Rt ≤ 1): {controlled_periods} ({controlled_periods / len(df) * 100:.1f}%)")

    # Distribution de qualité
    excellent = (df['rel_error'] < 0.15).sum()
    good = ((df['rel_error'] >= 0.15) & (df['rel_error'] < 0.30)).sum()
    fair = ((df['rel_error'] >= 0.30) & (df['rel_error'] < 0.50)).sum()
    poor = (df['rel_error'] >= 0.50).sum()

    print(f"\n🏆 QUALITÉ DES PRÉDICTIONS:")
    print(f"Excellentes (<15% erreur): {excellent} ({excellent / len(df) * 100:.1f}%)")
    print(f"Bonnes (15-30% erreur): {good} ({good / len(df) * 100:.1f}%)")
    print(f"Correctes (30-50% erreur): {fair} ({fair / len(df) * 100:.1f}%)")
    print(f"Mauvaises (>50% erreur): {poor} ({poor / len(df) * 100:.1f}%)")

    # Performance de classification (épidémique vs contrôlé)
    predicted_epidemic = (df['predicted'] > 1)
    actual_epidemic = (df['actual'] > 1)

    true_positives = ((predicted_epidemic) & (actual_epidemic)).sum()
    true_negatives = ((~predicted_epidemic) & (~actual_epidemic)).sum()
    false_positives = ((predicted_epidemic) & (~actual_epidemic)).sum()
    false_negatives = ((~predicted_epidemic) & (actual_epidemic)).sum()

    accuracy = (true_positives + true_negatives) / len(df)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

    print(f"\n🎯 PERFORMANCE DE CLASSIFICATION (Rt > 1 vs Rt ≤ 1):")
    print(f"Précision globale: {accuracy:.1%}")
    print(f"Précision (épidémique): {precision:.1%}")
    print(f"Rappel (épidémique): {recall:.1%}")

    # Afficher les détails des prédictions
    show_details = input(f"\n📋 Afficher les détails des prédictions ? (o/n): ").strip().lower()
    if show_details in ['o', 'oui', 'y', 'yes']:
        show_prediction_details(df, location)


def show_prediction_details(df, location):
    """Affiche les détails des prédictions de transmission (Rt futur)"""
    print(f"\n📋 DÉTAILS DES PRÉDICTIONS DE TRANSMISSION POUR {location.upper()}")
    print("=" * 90)

    # Options d'affichage
    print("Options d'affichage:")
    print("1. Toutes les prédictions")
    print("2. Les 20 premières prédictions")
    print("3. Les 20 dernières prédictions")
    print("4. Périodes épidémiques futures (Rt > 1)")
    print("5. Périodes contrôlées futures (Rt ≤ 1)")
    print("6. Prédictions d'une période spécifique")
    print("7. Meilleures prédictions (erreur < 15%)")
    print("8. Pires prédictions (erreur > 50%)")

    try:
        choice = input("\nVotre choix (1-8): ").strip()

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
            display_df = df[df['actual'] > 1].head(30)
            title = "PÉRIODES ÉPIDÉMIQUES FUTURES (Rt > 1)"
        elif choice == "5":
            display_df = df[df['actual'] <= 1].head(30)
            title = "PÉRIODES CONTRÔLÉES FUTURES (Rt ≤ 1)"
        elif choice == "6":
            year = input("Année (ex: 2020, 2021, 2022): ").strip()
            month = input("Mois (optionnel, ex: 03, 12): ").strip()

            filtered_df = df[df['date'].dt.year == int(year)]
            if month:
                filtered_df = filtered_df[filtered_df['date'].dt.month == int(month)]

            display_df = filtered_df
            title = f"PRÉDICTIONS {year}" + (f"-{month}" if month else "")
        elif choice == "7":
            display_df = df[df['rel_error'] < 0.15].head(30)
            title = "MEILLEURES PRÉDICTIONS (Erreur < 15%)"
        elif choice == "8":
            display_df = df[df['rel_error'] > 0.50].head(30)
            title = "PIRES PRÉDICTIONS (Erreur > 50%)"
        else:
            display_df = df.head(20)
            title = "20 PREMIÈRES PRÉDICTIONS (par défaut)"

        if len(display_df) == 0:
            print("❌ Aucune prédiction trouvée pour cette sélection")
            return

        print(f"\n📊 {title}")
        print("=" * 140)
        print(f"{'Date':<12} {'Rt Prédit':<10} {'Rt Futur':<10} {'Erreur':<8} {'Erreur%':<8} {'État Futur':<12} "
              f"{'Qualité':<10} {'Total Cas':<10} {'Cas 7j':<8} {'Contexte':<15}")
        print("-" * 140)

        for _, row in display_df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            predicted_rt = f"{row['predicted']:.2f}"
            actual_rt = f"{row['actual']:.2f}"
            error_abs = f"{row['error']:.2f}"
            error_rel = f"{row['rel_error'] * 100:.1f}%"

            # Données de contexte
            total_cases = f"{int(row.get('total_cases', 0)):,}"
            cases_7d = f"{int(row.get('new_cases_7d', 0)):,}"

            # Déterminer l'état épidémique futur
            if row['actual'] > 1.5:
                epidemic_state = "Forte croiss."
            elif row['actual'] > 1.0:
                epidemic_state = "Croissance"
            elif row['actual'] > 0.8:
                epidemic_state = "Stable"
            else:
                epidemic_state = "Déclin"

            # Qualité de la prédiction
            if row['rel_error'] < 0.15:
                quality = "Excellente"
            elif row['rel_error'] < 0.30:
                quality = "Bonne"
            elif row['rel_error'] < 0.50:
                quality = "Correcte"
            else:
                quality = "Mauvaise"

            # Contexte général
            if row.get('total_cases', 0) < 10000:
                context = "Début épidémie"
            elif row.get('total_cases', 0) < 100000:
                context = "Phase active"
            else:
                context = "Phase mature"

            print(f"{date_str:<12} {predicted_rt:<10} {actual_rt:<10} {error_abs:<8} {error_rel:<8} "
                  f"{epidemic_state:<12} {quality:<10} {total_cases:<10} {cases_7d:<8} {context:<15}")

        # Statistiques de la sélection
        print("\n📈 STATISTIQUES DE LA SÉLECTION:")
        print(f"Nombre de prédictions: {len(display_df)}")
        print(f"Rt futur réel moyen: {display_df['actual'].mean():.2f}")
        print(f"Rt futur prédit moyen: {display_df['predicted'].mean():.2f}")
        print(f"Erreur moyenne: {display_df['rel_error'].mean() * 100:.1f}%")
        print(f"Erreur médiane: {display_df['rel_error'].median() * 100:.1f}%")

        # Option pour sauvegarder
        save_csv = input(f"\n💾 Sauvegarder ces données en CSV ? (o/n): ").strip().lower()
        if save_csv in ['o', 'oui', 'y', 'yes']:
            save_prediction_details_csv(display_df, location, title, "transmission")

    except (ValueError, KeyboardInterrupt):
        print("\n❌ Affichage annulé")


def save_prediction_details_csv(df, location, title, model_type):
    """Sauvegarde les détails des prédictions en CSV"""
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)

        # Préparer les données pour CSV
        csv_df = df.copy()
        csv_df['date'] = csv_df['date'].dt.strftime('%Y-%m-%d')
        csv_df['predicted_rt'] = csv_df['predicted'].round(3)
        csv_df['actual_rt_future'] = csv_df['actual'].round(3)
        csv_df['error_abs'] = csv_df['error'].round(3)
        csv_df['rel_error_pct'] = (csv_df['rel_error'] * 100).round(1)

        # Ajouter colonnes d'analyse
        csv_df['future_epidemic_state'] = csv_df['actual'].apply(lambda x:
                                                                 'Forte croissance' if x > 1.5 else
                                                                 'Croissance' if x > 1.0 else
                                                                 'Stable' if x > 0.8 else
                                                                 'Déclin'
                                                                 )

        csv_df['prediction_quality'] = csv_df['rel_error'].apply(lambda x:
                                                                 'Excellente' if x < 0.15 else
                                                                 'Bonne' if x < 0.30 else
                                                                 'Correcte' if x < 0.50 else
                                                                 'Mauvaise'
                                                                 )

        # Nom du fichier
        safe_title = title.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct')
        safe_location = location.lower().replace(' ', '_')
        filename = f"predictions_{model_type}_{safe_location}_{safe_title}.csv"
        filepath = os.path.join(STATIC_DIR, filename)

        # Sélectionner les colonnes finales
        final_df = csv_df[['date', 'predicted_rt', 'actual_rt_future', 'error_abs', 'rel_error_pct',
                           'future_epidemic_state', 'prediction_quality', 'total_cases', 'total_deaths',
                           'new_cases_7d', 'new_deaths_7d']]
        final_df.columns = ['Date', 'Rt_Prédit', 'Rt_Futur_Réel', 'Erreur_Absolue', 'Erreur_Relative_%',
                            'État_Épidémique_Futur', 'Qualité_Prédiction', 'Total_Cas', 'Total_Décès',
                            'Nouveaux_Cas_7j', 'Nouveaux_Décès_7j']

        # Sauvegarder
        final_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"💾 Données sauvegardées: {filepath}")

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde CSV: {e}")


def save_validation_plots(results, location):
    """Sauvegarde les graphiques de validation pour le modèle de transmission"""
    try:
        os.makedirs(STATIC_DIR, exist_ok=True)

        df = pd.DataFrame(results)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Validation Transmission (Rt Futur): {location}', fontsize=16)

        # 1. Prédit vs Réel
        axes[0, 0].scatter(df['actual'], df['predicted'], alpha=0.6, s=30)
        max_val = max(df['actual'].max(), df['predicted'].max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Prédiction Parfaite')
        axes[0, 0].axhline(y=1, color='orange', linestyle=':', alpha=0.7, label='Rt = 1 (seuil épidémique)')
        axes[0, 0].axvline(x=1, color='orange', linestyle=':', alpha=0.7)
        axes[0, 0].set_xlabel('Rt Futur Réel')
        axes[0, 0].set_ylabel('Rt Futur Prédit')
        axes[0, 0].set_title('Prédit vs Réel (7-14 jours à l\'avance)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Évolution temporelle
        axes[0, 1].plot(df['date'], df['actual'], label='Rt Futur Réel', alpha=0.8, linewidth=2)
        axes[0, 1].plot(df['date'], df['predicted'], label='Rt Futur Prédit', alpha=0.8, linewidth=2)
        axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Seuil épidémique (Rt=1)')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Taux de Reproduction (Rt)')
        axes[0, 1].set_title('Prédiction de Rt Futur dans le Temps')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Distribution des erreurs
        axes[1, 0].hist(df['rel_error'] * 100, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['rel_error'].median() * 100, color='red', linestyle='--',
                           label=f'Médiane: {df["rel_error"].median() * 100:.1f}%')
        axes[1, 0].set_xlabel('Erreur Relative (%)')
        axes[1, 0].set_ylabel('Fréquence')
        axes[1, 0].set_title('Distribution des Erreurs de Prédiction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Erreur vs contexte épidémiologique
        axes[1, 1].scatter(df['total_cases'], df['rel_error'] * 100, alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Nombre Total de Cas')
        axes[1, 1].set_ylabel('Erreur Relative (%)')
        axes[1, 1].set_title('Erreur vs Contexte Épidémiologique')
        axes[1, 1].set_xscale('log')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Sauvegarder
        plot_path = os.path.join(STATIC_DIR, f'validation_transmission_fixed_{location.lower().replace(" ", "_")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📈 Graphique sauvegardé: {plot_path}")

    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde du graphique: {e}")


def validate_multiple_locations(locations, model, scaler, feature_cols, max_locations=10):
    """Valide le modèle sur plusieurs localisations"""
    print(f"\n🌍 VALIDATION DE TRANSMISSION SUR PLUSIEURS LOCALISATIONS")
    print("=" * 70)

    all_results = []
    location_summaries = []

    for i, location in enumerate(locations[:max_locations]):
        print(f"\n({i + 1}/{min(len(locations), max_locations)}) {location}")

        results = validate_single_location(location, model, scaler, feature_cols)

        if results:
            all_results.extend(results)

            # Métriques pour cette localisation
            df = pd.DataFrame(results)
            mae = mean_absolute_error(df['actual'], df['predicted'])
            r2 = r2_score(df['actual'], df['predicted'])
            median_error = df['rel_error'].median()
            avg_rt = df['actual'].mean()

            location_summaries.append({
                'Localisation': location,
                'Prédictions': len(results),
                'MAE': mae,
                'R²': r2,
                'Erreur_Médiane_%': median_error * 100,
                'Rt_Moyen': avg_rt,
                'Qualité': 'Excellente' if median_error < 0.15 else
                'Bonne' if median_error < 0.30 else
                'Correcte' if median_error < 0.50 else 'Mauvaise'
            })

            print(f"  ✅ MAE: {mae:.3f}, R²: {r2:.3f}, Rt moyen: {avg_rt:.2f}")

            # Sauvegarder le graphique
            save_validation_plots(results, location)

    # Résultats globaux
    if all_results:
        print(f"\n📊 RÉSULTATS GLOBAUX DE TRANSMISSION")
        print("=" * 70)

        overall_df = pd.DataFrame(all_results)
        overall_mae = mean_absolute_error(overall_df['actual'], overall_df['predicted'])
        overall_r2 = r2_score(overall_df['actual'], overall_df['predicted'])

        print(f"Total des prédictions: {len(all_results)}")
        print(f"Localisations testées: {len(location_summaries)}")
        print(f"MAE globale: {overall_mae:.3f}")
        print(f"R² global: {overall_r2:.4f}")
        print(f"Rt global moyen: {overall_df['actual'].mean():.2f}")

        # Tableau de comparaison
        if location_summaries:
            summary_df = pd.DataFrame(location_summaries)
            print(f"\n📋 COMPARAISON PAR LOCALISATION:")
            print(summary_df.to_string(index=False, float_format='%.3f'))


def main():
    """Fonction principale de validation du modèle de transmission"""
    print("🦠 VALIDATION CORRIGÉE DU MODÈLE DE TRANSMISSION COVID-19 (Rt)")
    print("=" * 80)
    print("📅 Période d'analyse: Mars 2020 - Mai 2022")
    print("🎯 Objectif: Tester la capacité à prédire le Rt 7-14 jours à l'avance")
    print("⚠️  Note: Prédiction FUTURE basée sur l'état actuel de l'épidémie")
    print("🔧 Correction: Évite le data leakage en utilisant vraiment des données futures")
    print("=" * 80)

    # Charger le modèle
    model, scaler, feature_cols = load_transmission_model()
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
            validate_multiple_locations(locations, model, scaler, feature_cols, max_locations=5)

        elif choice == "3":
            # Top 10
            validate_multiple_locations(locations, model, scaler, feature_cols, max_locations=10)

        elif choice == "4":
            # Localisation personnalisée
            print(f"\nLocalisations disponibles: {', '.join(locations[:20])}...")
            custom_location = input("Entrez le nom de la localisation: ").strip()
            if custom_location in locations:
                results = validate_single_location(custom_location, model, scaler, feature_cols)
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

    print(f"\n✅ Validation de transmission corrigée terminée!")
    print(f"📁 Graphiques sauvegardés dans: {STATIC_DIR}")
    print(f"🔧 Cette validation évite le data leakage et teste vraiment la prédiction future!")


if __name__ == "__main__":
    main()