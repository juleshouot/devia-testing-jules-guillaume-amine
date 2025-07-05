import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

warnings.filterwarnings('ignore')

# Configuration des chemins
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR = os.path.join(BASE_DIR, 'reports', 'figures')

# Création des répertoires nécessaires
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


# Connexion à la base de données
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


def load_data():
    """Chargement des données depuis la base de données"""
    engine = get_engine()

    print("Chargement des données pour les modèles avancés...")
    query = text("""
                 SELECT w.*, l.name AS location, v.name AS virus
                 FROM worldmeter w
                          JOIN location l ON w.location_id = l.id
                          JOIN virus v ON w.virus_id = v.id
                 ORDER BY w.date ASC
                 """)

    df = pd.read_sql(query, engine)

    # Conversion de la date
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Séparation par virus
    covid_df = df[df['virus'] == 'COVID'].copy()
    monkeypox_df = df[df['virus'] == 'Monkeypox'].copy()

    print(
        f"Données chargées: {len(df)} entrées, {df['location'].nunique()} localisations, {df['virus'].nunique()} virus")
    return df, covid_df, monkeypox_df


###########################################
# 1. MODÈLE DE CLASSIFICATION DES PHASES #
###########################################

def label_epidemic_phases(df):
    """Labelliser automatiquement les phases épidémiques"""
    print("\n=== Préparation des données pour le modèle de phases épidémiques ===")
    df_phases = df.copy()

    processed_regions = []

    # Traiter chaque localisation individuellement pour éviter les fuites de données
    for (location, virus), group in df_phases.groupby(['location', 'virus']):
        if len(group) < 30:  # Ignorer les localisations avec trop peu de données
            continue

        temp_df = group.copy().sort_values('date')

        # Calculer les moyennes mobiles
        temp_df['cases_ma14'] = temp_df['new_cases'].rolling(window=14, min_periods=3).mean()

        # Calculer la vitesse et l'accélération
        temp_df['cases_velocity'] = temp_df['cases_ma14'].pct_change(7).fillna(0)
        temp_df['cases_acceleration'] = temp_df['cases_velocity'].diff().fillna(0)

        # Normaliser les cas par rapport au maximum pour cette localisation
        max_cases = temp_df['cases_ma14'].max()
        if max_cases > 0:
            temp_df['cases_normalized'] = temp_df['cases_ma14'] / max_cases
        else:
            temp_df['cases_normalized'] = 0

        # Définir les phases selon des règles heuristiques
        conditions = [
            # Phase 1: Début - Croissance positive mais faible nombre de cas
            (temp_df['cases_velocity'] > 0.1) & (temp_df['cases_normalized'] < 0.1),

            # Phase 2: Accélération - Croissance positive significative
            (temp_df['cases_velocity'] > 0.05) & (temp_df['cases_acceleration'] >= 0),

            # Phase 3: Pic - Croissance proche de zéro ou changement de signe récent
            (abs(temp_df['cases_velocity']) < 0.03) |
            ((temp_df['cases_velocity'] * temp_df['cases_velocity'].shift(1)) < 0),

            # Phase 4: Déclin - Croissance négative
            (temp_df['cases_velocity'] < -0.05),

            # Phase 5: Fin - Croissance négative et faible nombre de cas
            (temp_df['cases_velocity'] < 0) & (temp_df['cases_normalized'] < 0.1)
        ]

        choices = [1, 2, 3, 4, 5]
        temp_df['epidemic_phase'] = np.select(conditions, choices, default=0)

        # Lisser les phases (éviter les changements trop rapides)
        temp_df['epidemic_phase_smoothed'] = temp_df['epidemic_phase'].rolling(window=7, min_periods=1).median().astype(
            int)

        # Ajouter au DataFrame final
        processed_regions.append(temp_df)

    # Combiner tous les résultats
    if not processed_regions:
        print("❌ Aucune région n'a suffisamment de données pour l'analyse des phases.")
        return None

    result_df = pd.concat(processed_regions)

    # Nettoyer
    result_df = result_df.dropna(subset=['epidemic_phase_smoothed'])

    # Sauvegarder les données labellisées
    result_df.to_csv(os.path.join(DATA_DIR, 'epidemic_phases_data.csv'), index=False)
    print(f"✅ Données de phases épidémiques sauvegardées: {len(result_df)} entrées")

    # Afficher la distribution des phases
    phase_counts = result_df['epidemic_phase_smoothed'].value_counts().sort_index()
    print("Distribution des phases épidémiques:")
    for phase, count in phase_counts.items():
        phase_name = {
            1: "Début",
            2: "Accélération",
            3: "Pic",
            4: "Déclin",
            5: "Fin"
        }.get(phase, "Indéfinie")
        print(f"  Phase {phase} ({phase_name}): {count} observations ({count / len(result_df) * 100:.1f}%)")

    return result_df


def prepare_phase_prediction_features(df_phases):
    """Prépare les features pour la prédiction des phases épidémiques"""
    print("\n=== Préparation des features pour le modèle de phases ===")

    # Features à inclure dans le modèle
    feature_cols = [
        'cases_ma14', 'cases_velocity', 'cases_acceleration', 'cases_normalized',
        'new_cases_per_million', 'total_cases_per_million',
        'new_deaths_per_million', 'total_deaths_per_million'
    ]

    # Ajouter des features temporelles (cycliques)
    df_phases['day_of_week'] = df_phases['date'].dt.dayofweek
    df_phases['day_sin'] = np.sin(2 * np.pi * df_phases['day_of_week'] / 7)
    df_phases['day_cos'] = np.cos(2 * np.pi * df_phases['day_of_week'] / 7)

    df_phases['month'] = df_phases['date'].dt.month
    df_phases['month_sin'] = np.sin(2 * np.pi * df_phases['month'] / 12)
    df_phases['month_cos'] = np.cos(2 * np.pi * df_phases['month'] / 12)

    # Ajouter ces features cycliques à notre liste
    feature_cols.extend(['day_sin', 'day_cos', 'month_sin', 'month_cos'])

    # Pour chaque région, calculer le nombre de jours depuis le premier cas
    df_phases['days_since_first_case'] = df_phases.groupby(['location', 'virus'])['date'].transform(
        lambda x: (x - x.min()).dt.days
    )
    feature_cols.append('days_since_first_case')

    # S'assurer que toutes les colonnes existent
    existing_cols = [col for col in feature_cols if col in df_phases.columns]

    print(f"Features pour le modèle de phases: {existing_cols}")
    return df_phases, existing_cols


def train_epidemic_phase_model():
    """Entraîne un modèle pour prédire la phase épidémique"""
    print("\n=== Modèle : Classification des Phases Épidémiques ===")

    # Charger les données
    df, covid_df, monkeypox_df = load_data()

    # Labelliser les phases
    df_phases = label_epidemic_phases(df)

    if df_phases is None:
        print("❌ Impossible de créer le modèle de phases épidémiques.")
        return None, None, None

    # Préparer les features
    df_phases, feature_cols = prepare_phase_prediction_features(df_phases)

    # Target variable
    target_col = 'epidemic_phase_smoothed'

    # Séparation temporelle
    df_phases = df_phases.sort_values('date')
    split_date = df_phases['date'].iloc[int(len(df_phases) * 0.8)]  # 80% train, 20% test
    print(f"Date de séparation train/test: {split_date}")

    train_data = df_phases[df_phases['date'] < split_date]
    test_data = df_phases[df_phases['date'] >= split_date]

    # Extraire X et y
    X_train = train_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_data[target_col]
    X_test = test_data[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_test = test_data[target_col]

    # Standardisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Données préparées : {X_train.shape[0]} train, {X_test.shape[0]} test")

    # Entraînement du modèle
    print("\n→ Entraînement du Random Forest Classifier pour les phases épidémiques")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )

    model.fit(X_train_scaled, y_train)

    # Évaluation
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Visualiser
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Début', 'Accélération', 'Pic', 'Déclin', 'Fin'],
                yticklabels=['Début', 'Accélération', 'Pic', 'Déclin', 'Fin'])
    plt.xlabel('Prédite')
    plt.ylabel('Réelle')
    plt.title('Matrice de Confusion - Phases Épidémiques')
    plt.tight_layout()

    # Sauvegarder
    confusion_matrix_path = os.path.join(PLOTS_DIR, 'epidemic_phases_confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    plt.close()
    print(f"Matrice de confusion sauvegardée: {confusion_matrix_path}")

    # Importance des features
    feature_importance = pd.DataFrame(
        {'feature': feature_cols, 'importance': model.feature_importances_}
    ).sort_values('importance', ascending=False)

    print("\nImportance des features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Visualiser
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Features - Prédiction des Phases Épidémiques')
    plt.tight_layout()

    # Sauvegarder
    feature_importance_path = os.path.join(PLOTS_DIR, 'epidemic_phases_feature_importance.png')
    plt.savefig(feature_importance_path)
    plt.close()
    print(f"Importance des features sauvegardée: {feature_importance_path}")

    # Sauvegarder le modèle
    model_path = os.path.join(MODEL_DIR, 'epidemic_phases_model.joblib')
    scaler_path = os.path.join(MODEL_DIR, 'epidemic_phases_scaler.joblib')
    metadata_path = os.path.join(MODEL_DIR, 'epidemic_phases_metadata.joblib')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        'feature_cols': feature_cols,
        'accuracy': accuracy,
        'phase_names': {
            1: "Début",
            2: "Accélération",
            3: "Pic",
            4: "Déclin",
            5: "Fin"
        }
    }

    joblib.dump(metadata, metadata_path)
    print(f"✅ Modèle de phases épidémiques sauvegardé: {model_path}")

    return model, scaler, metadata


#####################################
# 2. MODÈLE D'ANALYSE SAISONNIÈRE SIMPLIFIÉ #
#####################################

def simplified_seasonal_analysis(df):
    """Version simplifiée de l'analyse saisonnière sans statsmodels"""
    print("\n=== Analyse Saisonnière Simplifiée ===")

    seasonal_results = []

    # Pour chaque virus, analyser les tendances mensuelles
    for virus, virus_df in df.groupby('virus'):
        print(f"Analyse de la saisonnalité pour: {virus}")

        # Ajouter le mois à chaque entrée
        virus_df['month'] = virus_df['date'].dt.month
        virus_df['month_name'] = virus_df['date'].dt.strftime('%B')

        # Calculer la moyenne des nouveaux cas par mois
        monthly_avg = virus_df.groupby('month').agg({
            'new_cases': 'mean',
            'month_name': 'first'
        }).reset_index()

        if len(monthly_avg) < 6:
            print(f"  Pas assez de données mensuelles pour {virus}")
            continue

        # Normaliser pour obtenir l'effet saisonnier
        avg_all_months = monthly_avg['new_cases'].mean()
        if avg_all_months > 0:
            monthly_avg['seasonal_effect'] = (monthly_avg['new_cases'] / avg_all_months) - 1
        else:
            monthly_avg['seasonal_effect'] = 0

        # Trier par mois pour l'affichage
        monthly_avg = monthly_avg.sort_values('month')

        # Visualiser
        plt.figure(figsize=(14, 7))
        bars = plt.bar(monthly_avg['month_name'], monthly_avg['seasonal_effect'])

        # Colorer en rouge les mois avec effet positif, en bleu les négatifs
        for i, effect in enumerate(monthly_avg['seasonal_effect']):
            if effect > 0:
                bars[i].set_color('salmon')
            else:
                bars[i].set_color('skyblue')

        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'Effet saisonnier par mois pour {virus}')
        plt.xlabel('Mois')
        plt.ylabel('Effet saisonnier (positif = plus de cas que la moyenne)')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Sauvegarder
        monthly_plot_path = os.path.join(PLOTS_DIR, f'seasonality_monthly_{virus}.png')
        plt.savefig(monthly_plot_path)
        plt.close()
        print(f"  Graphique saisonnier mensuel sauvegardé: {monthly_plot_path}")

        # Ajouter l'information du virus
        monthly_avg['virus'] = virus
        seasonal_results.append(monthly_avg)

        # Analyse par région
        top_regions = virus_df.groupby('location')['new_cases'].sum().nlargest(5).index
        for location in top_regions:
            location_df = virus_df[virus_df['location'] == location]

            # Calculer la moyenne des nouveaux cas par mois pour cette région
            location_monthly = location_df.groupby('month').agg({
                'new_cases': 'mean',
                'month_name': 'first'
            }).reset_index()

            if len(location_monthly) < 6:
                continue

            # Normaliser
            avg_all_months_location = location_monthly['new_cases'].mean()
            if avg_all_months_location > 0:
                location_monthly['seasonal_effect'] = (location_monthly['new_cases'] / avg_all_months_location) - 1
            else:
                location_monthly['seasonal_effect'] = 0

            # Ajouter info
            location_monthly['virus'] = virus
            location_monthly['location'] = location
            seasonal_results.append(location_monthly)

    # Combiner tous les résultats
    if seasonal_results:
        combined_results = pd.concat(seasonal_results)
        results_path = os.path.join(DATA_DIR, 'seasonal_effects_data.csv')
        combined_results.to_csv(results_path, index=False)
        print(f"✅ Données d'effets saisonniers sauvegardées: {results_path}")
        return combined_results
    else:
        print("❌ Aucun résultat d'analyse saisonnière n'a pu être généré.")
        return None


def create_simplified_seasonal_model(seasonal_data):
    """Crée un modèle simplifié de risque saisonnier"""
    print("\n=== Création du modèle simplifié de risque saisonnier ===")

    if seasonal_data is None:
        print("❌ Données saisonnières non disponibles.")
        return None

    # Normaliser les effets saisonniers entre 0 et 1 pour chaque virus
    seasonal_models = {}

    for virus in seasonal_data['virus'].unique():
        virus_data = seasonal_data[(seasonal_data['virus'] == virus) &
                                   (~seasonal_data['location'].notna())].copy()

        if len(virus_data) < 6:  # Besoin d'au moins 6 mois de données
            continue

        # Normaliser entre 0 et 1
        min_effect = virus_data['seasonal_effect'].min()
        max_effect = virus_data['seasonal_effect'].max()

        if max_effect - min_effect == 0:
            continue

        virus_data['seasonal_risk'] = (virus_data['seasonal_effect'] - min_effect) / (max_effect - min_effect)

        # Identifier les mois à haut risque (>0.7)
        high_risk_months = virus_data[virus_data['seasonal_risk'] > 0.7]['month'].tolist()

        # Créer le modèle
        seasonal_models[virus] = {
            'monthly_risk': virus_data[['month', 'seasonal_risk']].set_index('month')['seasonal_risk'].to_dict(),
            'high_risk_months': high_risk_months
        }

        # Aussi pour les principales localisations
        location_models = {}
        for location in seasonal_data[seasonal_data['virus'] == virus]['location'].dropna().unique():
            loc_data = seasonal_data[(seasonal_data['virus'] == virus) &
                                     (seasonal_data['location'] == location)].copy()

            if len(loc_data) < 6:
                continue

            # Normaliser
            min_loc = loc_data['seasonal_effect'].min()
            max_loc = loc_data['seasonal_effect'].max()

            if max_loc - min_loc == 0:
                continue

            loc_data['seasonal_risk'] = (loc_data['seasonal_effect'] - min_loc) / (max_loc - min_loc)
            high_risk_loc = loc_data[loc_data['seasonal_risk'] > 0.7]['month'].tolist()

            location_models[location] = {
                'monthly_risk': loc_data[['month', 'seasonal_risk']].set_index('month')['seasonal_risk'].to_dict(),
                'high_risk_months': high_risk_loc
            }

        if location_models:
            seasonal_models[f"{virus}_locations"] = location_models

    # Sauvegarder le modèle
    if seasonal_models:
        model_path = os.path.join(MODEL_DIR, 'seasonal_risk_model.joblib')
        joblib.dump(seasonal_models, model_path)
        print(f"✅ Modèle de risque saisonnier sauvegardé: {model_path}")

        # Visualisation pour chaque virus
        for virus, model in seasonal_models.items():
            if isinstance(model, dict) and 'monthly_risk' in model:
                # Graphique des risques mensuels
                plt.figure(figsize=(12, 6))
                months = range(1, 13)
                risk_values = [model['monthly_risk'].get(month, 0) for month in months]

                bars = plt.bar(months, risk_values)

                # Colorer les mois à haut risque
                for i, month in enumerate(months):
                    if month in model['high_risk_months']:
                        bars[i].set_color('red')
                    else:
                        bars[i].set_color('skyblue')

                plt.title(f"Risque saisonnier pour {virus}")
                plt.ylabel("Risque relatif")
                plt.xlabel("Mois")
                plt.xticks(months, ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin',
                                    'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc'])
                plt.grid(True, alpha=0.3)
                plt.tight_layout()

                # Sauvegarder
                risk_viz_path = os.path.join(PLOTS_DIR, f'seasonal_risk_{virus}.png')
                plt.savefig(risk_viz_path)
                plt.close()
                print(f"  Visualisation du risque saisonnier pour {virus} sauvegardée")

        return seasonal_models
    else:
        print("❌ Impossible de créer un modèle de risque saisonnier.")
        return None


def main():
    """Fonction principale pour exécuter les deux modèles"""
    print("=== MODÈLES AVANCÉS DE PRÉDICTION PANDÉMIQUE ===")

    # Charger les données une seule fois
    df, covid_df, monkeypox_df = load_data()

    # 1. Modèle de phases épidémiques
    phase_model, phase_scaler, phase_metadata = train_epidemic_phase_model()

    # 2. Modèle de saisonnalité simplifié
    seasonal_data = simplified_seasonal_analysis(df)
    seasonal_model = create_simplified_seasonal_model(seasonal_data)

    print("\n=== MODÈLES AVANCÉS TERMINÉS ===")

    return phase_model, seasonal_model


if __name__ == "__main__":
    main()