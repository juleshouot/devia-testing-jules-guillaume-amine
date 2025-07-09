import datetime
from django.core.management.base import BaseCommand
from django.utils import timezone
from pandemics_app.models import (
    Location,
    Virus,
    Prediction,
    GeographicalSpreadPrediction,
    ModelMetrics,
)
from pandemics_app.ml.predictor import (
    PandemicPredictor,
    fetch_location_features_from_db,
    fetch_geographical_features,
)


class Command(BaseCommand):
    help = "Charge les prédictions générées par le PandemicPredictor dans la base de données Django."

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Début du chargement des prédictions..."))
        predictor = PandemicPredictor()

        locations = Location.objects.all()
        viruses = Virus.objects.all()

        if not locations.exists():
            self.stdout.write(
                self.style.WARNING(
                    'Aucune localisation trouvée en base. Veuillez charger les localisations d"abord.'
                )
            )
            return
        if not viruses.exists():
            self.stdout.write(
                self.style.WARNING(
                    'Aucun virus trouvé en base. Veuillez charger les virus d"abord.'
                )
            )
            return

        prediction_date = timezone.now().date() + datetime.timedelta(days=7)

        for virus in viruses:
            self.stdout.write(
                self.style.HTTP_INFO(
                    f"Traitement des prédictions pour le virus: {virus.name}"
                )
            )
            geo_data = None
            try:
                geo_data = fetch_geographical_features(virus_name=virus.name)
                self.stdout.write(
                    f"  Données géographiques pour {virus.name} récupérées."
                )
            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(
                        f"  Erreur lors de la récupération des données géographiques pour {virus.name}: {e}"
                    )
                )
                continue

            if geo_data:
                try:
                    spread_prediction_value = predictor.predict_geographical_spread(
                        geo_data
                    )
                    GeographicalSpreadPrediction.objects.update_or_create(
                        virus=virus,
                        prediction_date=prediction_date,
                        defaults={
                            "predicted_new_locations": int(
                                round(spread_prediction_value)
                            )
                        },
                    )
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"  Prédiction de propagation géographique pour {virus.name} sauvegardée: {int(round(spread_prediction_value))} nouvelles localisations."
                        )
                    )
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"  Erreur lors de la prédiction/sauvegarde de la propagation géographique pour {virus.name}: {e}"
                        )
                    )

            for location in locations:
                self.stdout.write(
                    f"  Traitement pour {location.name} avec {virus.name}..."
                )
                location_data = None
                try:
                    location_data = fetch_location_features_from_db(
                        location_name=location.name, virus_name=virus.name
                    )
                    self.stdout.write(
                        f"    Données de localisation pour {location.name} ({virus.name}) récupérées."
                    )
                except ValueError as ve:
                    self.stdout.write(
                        self.style.WARNING(
                            f"    Données insuffisantes pour {location.name} ({virus.name}): {ve}"
                        )
                    )
                    continue
                except Exception as e:
                    self.stdout.write(
                        self.style.ERROR(
                            f"    Erreur lors de la récupération des données de localisation pour {location.name} ({virus.name}): {e}"
                        )
                    )
                    continue

                if location_data:
                    predicted_cases_val = None
                    transmission_rate_val = None
                    try:
                        transmission_rate_val = predictor.predict_transmission_rate(
                            location_data
                        )
                        if (
                            "cases_7day_lag7" in location_data
                            and location_data["cases_7day_lag7"] is not None
                        ):
                            predicted_cases_val = int(
                                round(
                                    location_data["cases_7day_lag7"]
                                    * (1 + transmission_rate_val)
                                )
                            )
                        else:
                            self.stdout.write(
                                self.style.WARNING(
                                    f"    cases_7day_lag7 manquant ou None pour {location.name} ({virus.name}), impossible de calculer predicted_cases."
                                )
                            )

                        self.stdout.write(
                            f"    Taux de transmission pour {location.name} ({virus.name}): {transmission_rate_val:.4f}"
                        )
                        if predicted_cases_val is not None:
                            self.stdout.write(
                                f"    Cas prédits pour {location.name} ({virus.name}): {predicted_cases_val}"
                            )

                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(
                                f"    Erreur lors de la prédiction du taux de transmission pour {location.name} ({virus.name}): {e}"
                            )
                        )

                    predicted_deaths_val = None
                    mortality_rate_val = None
                    try:
                        mortality_rate_val = predictor.predict_mortality_rate(
                            location_data
                        )
                        if predicted_cases_val is not None:
                            predicted_deaths_val = int(
                                round(predicted_cases_val * mortality_rate_val)
                            )
                        elif (
                            "cases_7day_lag7" in location_data
                            and location_data["cases_7day_lag7"] is not None
                            and "cases_growth" in location_data
                            and location_data["cases_growth"] is not None
                        ):
                            temp_predicted_cases = int(
                                round(
                                    location_data["cases_7day_lag7"]
                                    * (1 + location_data["cases_growth"])
                                )
                            )
                            predicted_deaths_val = int(
                                round(temp_predicted_cases * mortality_rate_val)
                            )
                        else:
                            self.stdout.write(
                                self.style.WARNING(
                                    f"    Données manquantes pour calculer predicted_deaths pour {location.name} ({virus.name})."
                                )
                            )

                        self.stdout.write(
                            f"    Taux de mortalité pour {location.name} ({virus.name}): {mortality_rate_val:.4f}"
                        )
                        if predicted_deaths_val is not None:
                            self.stdout.write(
                                f"    Décès prédits pour {location.name} ({virus.name}): {predicted_deaths_val}"
                            )  # Ligne 116 corrigée

                    except Exception as e:
                        self.stdout.write(
                            self.style.ERROR(
                                f"    Erreur lors de la prédiction du taux de mortalité pour {location.name} ({virus.name}): {e}"
                            )
                        )

                    if (
                        transmission_rate_val is not None
                        or mortality_rate_val is not None
                    ):
                        Prediction.objects.update_or_create(
                            location=location,
                            virus=virus,
                            prediction_date=prediction_date,
                            defaults={
                                "transmission_rate": transmission_rate_val,
                                "mortality_rate": mortality_rate_val,
                                "predicted_cases": predicted_cases_val,
                                "predicted_deaths": predicted_deaths_val,
                                "timestamp": timezone.now(),
                            },
                        )
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"    Prédiction (transmission/mortalité) pour {location.name} ({virus.name}) sauvegardée/mise à jour."
                            )
                        )
                    else:
                        self.stdout.write(
                            self.style.WARNING(
                                f"    Aucune donnée de prédiction (transmission/mortalité) à sauvegarder pour {location.name} ({virus.name})."
                            )
                        )

        for model_type, metadata in predictor.model_metadata.items():
            if (
                "model_name" in metadata
                and "mse" in metadata
                and "rmse" in metadata
                and "mae" in metadata
                and "r2_score" in metadata
            ):
                ModelMetrics.objects.update_or_create(
                    model_type=model_type,
                    model_name=metadata.get("model_name", "default_model_name"),
                    defaults={
                        "mse": metadata["mse"],
                        "rmse": metadata["rmse"],
                        "mae": metadata["mae"],
                        "r2_score": metadata["r2_score"],
                        "cv_rmse": metadata.get("cv_rmse"),
                        "timestamp": timezone.now(),
                    },
                )
                self.stdout.write(
                    self.style.SUCCESS(
                        f'  Métrique pour le modèle {model_type} ({metadata.get("model_name")}) sauvegardée/mise à jour.'
                    )
                )
            else:
                self.stdout.write(
                    self.style.WARNING(
                        f"  Métriques incomplètes pour le modèle {model_type}, non sauvegardées."
                    )
                )

        self.stdout.write(
            self.style.SUCCESS("Chargement des prédictions terminé avec succès !")
        )
