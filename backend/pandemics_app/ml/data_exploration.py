import pandas as pd
import numpy as np
import os
from sqlalchemy import create_engine, text

# ========================
# CONFIGURATION & BDD
# ========================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')

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

# ========================
# CHARGEMENT DES DONNÉES
# ========================

def load_data():
    engine = get_engine()
    query = text("""
        SELECT w.*, l.name AS location, v.name AS virus
        FROM worldmeter w
        JOIN location l ON w.location_id = l.id
        JOIN virus v ON w.virus_id = v.id
        ORDER BY w.date ASC
    """)
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'])
    return df

# ========================
# PRÉPARATION DES FEATURES
# ========================

def prepare_transmission_features(df):
    print("\nPréparation des features de transmission...")

    df = df.copy()
    df = df.sort_values(['virus', 'location', 'date'])

    df['cases_7d'] = df['new_cases'].rolling(window=7).sum()
    df['cases_7d_lag7'] = df['cases_7d'].shift(7)
    df['cases_7d_lag14'] = df['cases_7d'].shift(14)
    df['deaths_7d'] = df['new_deaths'].rolling(window=7).sum()
    df['deaths_7d_lag7'] = df['deaths_7d'].shift(7)
    df['deaths_7d_lag14'] = df['deaths_7d'].shift(14)

    df['cases_growth'] = df['cases_7d'].pct_change(7)
    df['deaths_growth'] = df['deaths_7d'].pct_change(7)
    df['target_growth_rate'] = df['cases_7d'].pct_change(7)

    df = df.dropna(subset=['target_growth_rate'])

    return df

def prepare_mortality_features(df):
    print("\nPréparation des features de mortalité...")

    df = df.copy()
    df = df.sort_values(['virus', 'location', 'date'])

    df['cases_7d'] = df['new_cases'].rolling(window=7).sum()
    df['deaths_7d'] = df['new_deaths'].rolling(window=7).sum()

    df['cases_7d_lag7'] = df['cases_7d'].shift(7)
    df['deaths_7d_lag7'] = df['deaths_7d'].shift(7)

    df['mortality_ratio'] = df['deaths_7d_lag7'] / df['cases_7d_lag7'].replace(0, 1)
    df['mortality_ratio'] = df['mortality_ratio'].clip(0, 1)

    df = df.dropna(subset=['mortality_ratio'])

    return df

def prepare_geographical_features(df):
    print("\nPréparation des features géographiques...")

    first_cases = df[df['new_cases'] > 0].groupby(['virus', 'location'])['date'].min().reset_index()
    first_cases['week'] = first_cases['date'].dt.isocalendar().week
    first_cases['year'] = first_cases['date'].dt.isocalendar().year
    first_cases['yearweek'] = first_cases['year'].astype(str) + '-' + first_cases['week'].astype(str).str.zfill(2)

    spread = first_cases.groupby('yearweek').size().reset_index(name='new_locations')
    for lag in range(1, 5):
        spread[f'new_locations_lag{lag}'] = spread['new_locations'].shift(lag)
    spread['new_locations_ma2'] = spread['new_locations'].rolling(2).mean()
    spread['new_locations_ma3'] = spread['new_locations'].rolling(3).mean()

    spread = spread.dropna()
    return spread

# ========================
# EXPORT
# ========================

def save_to_csv(df, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f'{name}.csv')
    df.to_csv(path, index=False)
    print(f"✅ Données sauvegardées: {path}")

# ========================
# MAIN
# ========================

def main():
    print("=== Lancement de l'exploration des données ===")
    df = load_data()

    covid_df = df[df['virus'] == 'COVID'].copy()
    monkeypox_df = df[df['virus'] == 'Monkeypox'].copy()

    transmission_df = prepare_transmission_features(pd.concat([covid_df, monkeypox_df]))
    mortality_df = prepare_mortality_features(pd.concat([covid_df, monkeypox_df]))
    geographical_df = prepare_geographical_features(pd.concat([covid_df, monkeypox_df]))

    save_to_csv(transmission_df, 'transmission_features')
    save_to_csv(mortality_df, 'mortality_features')
    save_to_csv(geographical_df, 'geographical_features')

    print("\n✅ Exploration et préparation des données terminées.")

if __name__ == "__main__":
    main()
