import datetime
import os
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, MetaData, select, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert as pg_insert


def get_engine():
    """Fonction utilitaire pour créer l'engine SQLAlchemy."""
    POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "guigui")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "pandemies")
    # Enlever le search_path pour éviter les conflits
    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )
    return engine


def safe_int(val):
    """Convertit la valeur en int si elle n'est pas NaN, sinon retourne None."""
    return int(val) if pd.notna(val) else None


def safe_float(val):
    """Convertit la valeur en float si elle n'est pas NaN, sinon retourne None."""
    return float(val) if pd.notna(val) else None


def task_insert_locations():
    """
    Lecture des CSV pour extraire les localisations et insertion dans la table 'location'
    sans doublon (vérification avant insertion).
    NOUVEAU: Inclut l'import des données de population.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Forcer l'utilisation explicite du schéma pandemics
        metadata = MetaData()
        metadata.reflect(bind=engine, schema='pandemics')

        # Utiliser explicitement la table du bon schéma
        location_table = metadata.tables['pandemics.location']

        print(f"Table utilisée: {location_table}")

        # Lecture des CSV principaux
        monkeypox_path = "/opt/airflow/data/monkeypox.csv"
        covid_path = "/opt/airflow/data/covid.csv"
        population_path = "/opt/airflow/data/population.csv"  # NOUVEAU

        df_monkeypox = pd.read_csv(monkeypox_path)
        df_covid = pd.read_csv(covid_path)

        # NOUVEAU: Lecture du fichier population
        df_population = pd.read_csv(population_path)
        print(f"Données de population chargées: {len(df_population)} pays")

        # Extraction des localisations
        locations_monkeypox = df_monkeypox[['location', 'iso_code']].drop_duplicates().rename(
            columns={'location': 'name'})
        locations_covid = df_covid[['country']].drop_duplicates().rename(columns={'country': 'name'})
        locations_covid['iso_code'] = None  # pas de code ISO dans le CSV covid

        locations_all = pd.concat([locations_monkeypox, locations_covid], ignore_index=True)
        locations_all = locations_all.drop_duplicates(subset=['name'])

        # NOUVEAU: Merger avec les données de population
        locations_all = locations_all.merge(
            df_population[['name', 'population']],
            on='name',
            how='left'
        )

        # Remplir les populations manquantes par une valeur par défaut
        locations_all['population'] = locations_all['population'].fillna(0)

        for _, row in locations_all.iterrows():
            # Vérification si la location existe déjà
            stmt = select(location_table.c.id).where(location_table.c.name == row['name'])
            result = session.execute(stmt).fetchone()
            if result is None:
                # NOUVEAU: Inclure la population dans l'insertion
                ins = location_table.insert().values(
                    name=row['name'],
                    iso_code=row['iso_code'],
                    population=int(row['population']) if pd.notna(row['population']) else 0
                )
                session.execute(ins)
                print(
                    f"Inséré: {row['name']} - Population: {int(row['population']) if pd.notna(row['population']) else 0:,}")
            else:
                # Mise à jour de la population si la localisation existe déjà
                update_stmt = location_table.update().where(
                    location_table.c.name == row['name']
                ).values(
                    population=int(row['population']) if pd.notna(row['population']) else 0
                )
                session.execute(update_stmt)
                print(f"Mis à jour population pour: {row['name']}")

        session.commit()
        print("Insertion des locations avec population terminée.")

    except Exception as e:
        session.rollback()
        print(f"Erreur dans task_insert_locations: {e}")
        raise
    finally:
        session.close()


def task_insert_viruses():
    """
    Insertion des virus dans la table 'virus' en vérifiant qu'ils n'existent pas déjà.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        metadata = MetaData()
        metadata.reflect(bind=engine, schema='pandemics')
        virus_table = metadata.tables['pandemics.virus']

        print(f"Table utilisée: {virus_table}")

        virus_data = [{'id': 1, 'name': 'Monkeypox'}, {'id': 2, 'name': 'COVID'}]
        for virus in virus_data:
            stmt = select(virus_table.c.id).where(virus_table.c.id == virus['id'])
            result = session.execute(stmt).fetchone()
            if result is None:
                ins = virus_table.insert().values(id=virus['id'], name=virus['name'])
                session.execute(ins)
        session.commit()
        print("Insertion des virus terminée.")

    except Exception as e:
        session.rollback()
        print(f"Erreur dans task_insert_viruses: {e}")
        raise
    finally:
        session.close()


def task_insert_worldmeter():
    """
    Transformation et insertion des données de worldmeter à partir des fichiers CSV.
    """
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        metadata = MetaData()
        metadata.reflect(bind=engine, schema='pandemics')
        location_table = metadata.tables['pandemics.location']
        worldmeter_table = metadata.tables['pandemics.worldmeter']

        print(f"Tables utilisées: {location_table}, {worldmeter_table}")

        # Construction d'un mapping nom -> id pour les localisations
        loc_mapping = {}
        for row in session.execute(select(location_table.c.id, location_table.c.name)):
            loc_mapping[row[1]] = row[0]

        monkeypox_path = "/opt/airflow/data/monkeypox.csv"
        covid_path = "/opt/airflow/data/covid.csv"
        df_monkeypox = pd.read_csv(monkeypox_path)
        df_covid = pd.read_csv(covid_path)

        worldmeter_data = []
        # Traitement du fichier monkeypox
        for _, row in df_monkeypox.iterrows():
            loc_id = loc_mapping.get(row['location'])
            if loc_id is None:
                continue
            worldmeter_data.append({
                'date': row['date'],
                'total_cases': safe_int(row['total_cases']),
                'total_deaths': safe_int(row['total_deaths']),
                'new_cases': safe_int(row['new_cases']),
                'new_deaths': safe_int(row['new_deaths']),
                'new_cases_smoothed': safe_float(row['new_cases_smoothed']),
                'new_deaths_smoothed': safe_float(row['new_deaths_smoothed']),
                'new_cases_per_million': safe_float(row['new_cases_per_million']),
                'total_cases_per_million': safe_float(row['total_cases_per_million']),
                'new_cases_smoothed_per_million': safe_float(row['new_cases_smoothed_per_million']),
                'new_deaths_per_million': safe_float(row['new_deaths_per_million']),
                'total_deaths_per_million': safe_float(row['total_deaths_per_million']),
                'new_deaths_smoothed_per_million': safe_float(row['new_deaths_smoothed_per_million']),
                'location_id': int(loc_id),
                'virus_id': 1  # Monkeypox
            })
        # Traitement du fichier covid
        for _, row in df_covid.iterrows():
            loc_id = loc_mapping.get(row['country'])
            if loc_id is None:
                continue
            worldmeter_data.append({
                'date': row['date'],
                'total_cases': safe_int(row['cumulative_total_cases']),
                'total_deaths': safe_int(row['cumulative_total_deaths']),
                'new_cases': safe_int(row['daily_new_cases']),
                'new_deaths': safe_int(row['daily_new_deaths']),
                'new_cases_smoothed': None,
                'new_deaths_smoothed': None,
                'new_cases_per_million': None,
                'total_cases_per_million': None,
                'new_cases_smoothed_per_million': None,
                'new_deaths_per_million': None,
                'total_deaths_per_million': None,
                'new_deaths_smoothed_per_million': None,
                'location_id': int(loc_id),
                'virus_id': 2  # COVID
            })

        # Insertion avec ON CONFLICT DO NOTHING pour éviter les doublons
        stmt = pg_insert(worldmeter_table).values(worldmeter_data)
        stmt = stmt.on_conflict_do_nothing(index_elements=['date', 'location_id', 'virus_id'])
        session.execute(stmt)
        session.commit()
        print("Insertion des données dans worldmeter terminée.")

    except Exception as e:
        session.rollback()
        print(f"Erreur dans task_insert_worldmeter: {e}")
        raise
    finally:
        session.close()


default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2025, 3, 10),
    'retries': 1,
}

with DAG(
        'Airflow_ETL_DAG_with_Population',
        default_args=default_args,
        schedule_interval='@once',
        catchup=False,
        description="DAG ETL avec données de population"
) as dag:
    t1 = PythonOperator(
        task_id='insert_locations_with_population',
        python_callable=task_insert_locations
    )

    t2 = PythonOperator(
        task_id='insert_viruses',
        python_callable=task_insert_viruses
    )

    t3 = PythonOperator(
        task_id='insert_worldmeter',
        python_callable=task_insert_worldmeter
    )

    t1 >> t2 >> t3