import os
import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from sqlalchemy import create_engine, text


def truncate_tables():
    """
    Fonction pour vider les tables en exécutant une commande TRUNCATE.
    La commande RESTART IDENTITY permet de réinitialiser les séquences et CASCADE supprime les enregistrements dépendants.
    """
    # Récupération des paramètres de connexion depuis les variables d'environnement
    POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "guigui")
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "pandemies")

    # Création de l'engine SQLAlchemy avec le schéma "pandemics"
    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        connect_args={"options": "-c search_path=pandemics"}
    )

    # Exécution de la commande TRUNCATE dans un contexte de transaction
    with engine.begin() as connection:
        connection.execute(
            text("TRUNCATE TABLE location, virus, worldmeter RESTART IDENTITY CASCADE;")
        )
        print("Les tables ont été vidées avec succès.")


# Définition des paramètres par défaut du DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime.datetime(2025, 3, 10),
    'retries': 1,
}

with DAG(
        'clean_tables_dag',
        default_args=default_args,
        schedule_interval='@once',  # ou modifiez selon vos besoins
        catchup=False,
        description="DAG pour vider les tables 'location', 'virus' et 'worldmeter'"
) as dag:
    task_truncate = PythonOperator(
        task_id='truncate_tables',
        python_callable=truncate_tables
    )

    task_truncate
