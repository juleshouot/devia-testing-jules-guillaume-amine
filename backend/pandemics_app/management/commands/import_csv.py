from django.core.management.base import BaseCommand
import pandas as pd


class Command(BaseCommand):
    help = "Importer un fichier CSV dans la base de données"

    def add_arguments(self, parser):
        parser.add_argument(
            "file_path", type=str, help="Le chemin du fichier CSV à importer"
        )

    def handle(self, *args, **kwargs):
        file_path = kwargs["file_path"]
        try:
            data = pd.read_csv(file_path, encoding="latin-1")
            # Traitez les données ici
            print(f"Données importées avec succès depuis {file_path}.")
        except UnicodeDecodeError as e:
            print(f"Erreur d'encodage détectée : {e}")
