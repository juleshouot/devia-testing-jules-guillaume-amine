from django.db import connection
from pandemics_app.models import Worldmeter


def run():
    print("⏳ Importation des données Worldmeter...")

    # 1. Supprimer les anciennes données
    Worldmeter.objects.all().delete()

    # 2. Requête SQL
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT id, date, total_cases, total_deaths, new_cases, new_deaths,
                   new_cases_smoothed, new_deaths_smoothed, new_cases_per_million,
                   total_cases_per_million, new_cases_smoothed_per_million,
                   new_deaths_per_million, total_deaths_per_million,
                   new_deaths_smoothed_per_million, location_id, virus_id
            FROM pandemics.worldmeter
        """
        )
        rows = cursor.fetchall()

    # 3. Création des objets
    worldmeters = [
        Worldmeter(
            id=row[0],
            date=row[1],
            total_cases=row[2],
            total_deaths=row[3],
            new_cases=row[4],
            new_deaths=row[5],
            new_cases_smoothed=row[6],
            new_deaths_smoothed=row[7],
            new_cases_per_million=row[8],
            total_cases_per_million=row[9],
            new_cases_smoothed_per_million=row[10],
            new_deaths_per_million=row[11],
            total_deaths_per_million=row[12],
            new_deaths_smoothed_per_million=row[13],
            location_id=row[14],
            virus_id=row[15],
        )
        for row in rows
    ]

    # 4. Insertion
    Worldmeter.objects.bulk_create(worldmeters, ignore_conflicts=True)
    print(f"✅ {len(worldmeters)} lignes insérées dans Worldmeter.")
