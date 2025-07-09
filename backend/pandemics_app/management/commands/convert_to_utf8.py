from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Convertir un fichier CSV en UTF-8"

    def add_arguments(self, parser):
        parser.add_argument("file_path", type=str, help="Chemin du fichier source")
        parser.add_argument(
            "output_path", type=str, help="Chemin du fichier converti (UTF-8)"
        )

    def handle(self, *args, **kwargs):
        file_path = kwargs["file_path"]
        output_path = kwargs["output_path"]

        try:
            with open(file_path, "r", encoding="ISO-8859-1") as source_file:
                content = source_file.read()

            with open(output_path, "w", encoding="utf-8") as target_file:
                target_file.write(content)

            self.stdout.write(
                self.style.SUCCESS(
                    f"Fichier converti avec succès : {file_path} -> {output_path}"
                )
            )
        except Exception as e:
            self.stderr.write(
                self.style.ERROR(
                    f"Erreur lors de la conversion du fichier {file_path} : {e}"
                )
            )
