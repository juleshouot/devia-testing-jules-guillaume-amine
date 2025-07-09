import unittest
from django.test import TestCase, Client
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APIClient
from .models import (
    Location,
    Virus,
    Prediction,
    GeographicalSpreadPrediction,
    ModelMetrics,
)
from .ml.predictor import PandemicPredictor
import json
import os
import numpy as np


class PredictorTests(unittest.TestCase):
    """
    Tests unitaires pour la classe PandemicPredictor
    """

    def setUp(self):
        """
        Configuration initiale pour les tests
        """
        # Vérifier si les modèles existent, sinon les tests seront ignorés
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pandemics_app", "ml", "models"
        )
        self.models_exist = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0

        if self.models_exist:
            self.predictor = PandemicPredictor()

            # Données de test
            self.location_data = {
                "cases_7day_lag7": 1000,
                "cases_7day_lag14": 800,
                "deaths_7day_lag7": 20,
                "deaths_7day_lag14": 15,
                "cases_growth": 0.25,
                "deaths_growth": 0.33,
            }

            self.spread_data = {
                "new_locations_lag1": 5,
                "new_locations_lag2": 7,
                "new_locations_lag3": 10,
                "new_locations_lag4": 8,
                "new_locations_ma2": 6,
                "new_locations_ma3": 7.33,
            }

    def test_predict_transmission_rate(self):
        """
        Teste la prédiction du taux de transmission
        """
        if not self.models_exist:
            self.skipTest("Les modèles n'existent pas encore")

        transmission_rate = self.predictor.predict_transmission_rate(self.location_data)

        # Vérifier que la prédiction est un nombre
        self.assertIsInstance(transmission_rate, (int, float, np.number))

        # Vérifier que la prédiction est dans une plage raisonnable
        self.assertGreaterEqual(transmission_rate, 0)
        self.assertLessEqual(transmission_rate, 10)

    def test_predict_mortality_rate(self):
        """
        Teste la prédiction du taux de mortalité
        """
        if not self.models_exist:
            self.skipTest("Les modèles n'existent pas encore")

        mortality_rate = self.predictor.predict_mortality_rate(self.location_data)

        # Vérifier que la prédiction est un nombre
        self.assertIsInstance(mortality_rate, (int, float, np.number))

        # Vérifier que la prédiction est dans une plage raisonnable (entre 0 et 1)
        self.assertGreaterEqual(mortality_rate, 0)
        self.assertLessEqual(mortality_rate, 1)

    def test_predict_geographical_spread(self):
        """
        Teste la prédiction de la propagation géographique
        """
        if not self.models_exist:
            self.skipTest("Les modèles n'existent pas encore")

        geographical_spread = self.predictor.predict_geographical_spread(
            self.spread_data
        )

        # Vérifier que la prédiction est un nombre entier
        self.assertIsInstance(geographical_spread, (int, np.integer))

        # Vérifier que la prédiction est positive
        self.assertGreaterEqual(geographical_spread, 0)

    def test_predict_combined(self):
        """
        Teste les prédictions combinées
        """
        if not self.models_exist:
            self.skipTest("Les modèles n'existent pas encore")

        combined = self.predictor.predict_combined(self.location_data, self.spread_data)

        # Vérifier que le résultat est un dictionnaire
        self.assertIsInstance(combined, dict)

        # Vérifier que le dictionnaire contient les clés attendues
        self.assertIn("timestamp", combined)
        self.assertIn("predictions", combined)

        # Vérifier que les prédictions contiennent les valeurs attendues
        predictions = combined["predictions"]
        if "transmission_rate" in predictions:
            self.assertIsInstance(
                predictions["transmission_rate"], (int, float, np.number)
            )

        if "mortality_rate" in predictions:
            self.assertIsInstance(
                predictions["mortality_rate"], (int, float, np.number)
            )

        if "geographical_spread" in predictions:
            self.assertIsInstance(predictions["geographical_spread"], (int, np.integer))

    def test_generate_forecast(self):
        """
        Teste la génération de prévisions sur plusieurs semaines
        """
        if not self.models_exist:
            self.skipTest("Les modèles n'existent pas encore")

        forecasts = self.predictor.generate_forecast(
            self.location_data, self.spread_data, weeks=4
        )

        # Vérifier que le résultat est une liste
        self.assertIsInstance(forecasts, list)

        # Vérifier que la liste contient 4 prévisions
        self.assertEqual(len(forecasts), 4)

        # Vérifier que chaque prévision est un dictionnaire avec les clés attendues
        for forecast in forecasts:
            self.assertIsInstance(forecast, dict)
            self.assertIn("week", forecast)
            self.assertIn("date", forecast)
            self.assertIn("predictions", forecast)


class APITests(TestCase):
    """
    Tests pour les endpoints de l'API
    """

    def setUp(self):
        """
        Configuration initiale pour les tests
        """
        self.client = APIClient()

        # Créer des données de test
        self.location = Location.objects.create(name="Test Location", iso_code="TST")
        self.virus = Virus.objects.create(name="Test Virus")

        # Données pour les requêtes
        self.transmission_data = {
            "location_id": self.location.id,
            "virus_id": self.virus.id,
            "cases_7day_lag7": 1000,
            "cases_7day_lag14": 800,
            "deaths_7day_lag7": 20,
            "deaths_7day_lag14": 15,
            "cases_growth": 0.25,
            "deaths_growth": 0.33,
        }

        self.mortality_data = self.transmission_data.copy()

        self.geographical_data = {
            "virus_id": self.virus.id,
            "new_locations_lag1": 5,
            "new_locations_lag2": 7,
            "new_locations_lag3": 10,
            "new_locations_lag4": 8,
            "new_locations_ma2": 6,
            "new_locations_ma3": 7.33,
        }

        self.combined_data = {**self.transmission_data, **self.geographical_data}

        self.forecast_data = {
            "location_id": self.location.id,
            "virus_id": self.virus.id,
            "weeks": 4,
        }

    def test_location_list(self):
        """
        Teste l'endpoint de liste des localisations
        """
        url = reverse("location-list")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["name"], "Test Location")

    def test_virus_list(self):
        """
        Teste l'endpoint de liste des virus
        """
        url = reverse("virus-list")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0]["name"], "Test Virus")

    def test_prediction_endpoints(self):
        """
        Teste les endpoints de prédiction
        """
        # Vérifier si les modèles existent
        model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "pandemics_app", "ml", "models"
        )
        models_exist = os.path.exists(model_dir) and len(os.listdir(model_dir)) > 0

        if not models_exist:
            self.skipTest("Les modèles n'existent pas encore")

        # Test de l'endpoint de prédiction du taux de transmission
        url = reverse("predict-transmission")
        response = self.client.post(url, self.transmission_data, format="json")

        # Si le prédicteur n'est pas disponible, le test sera ignoré
        if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
            self.skipTest("Le prédicteur n'est pas disponible")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("transmission_rate", response.data)
        self.assertIn("predicted_cases", response.data)

        # Test de l'endpoint de prédiction du taux de mortalité
        url = reverse("predict-mortality")
        response = self.client.post(url, self.mortality_data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("mortality_rate", response.data)
        self.assertIn("predicted_deaths", response.data)

        # Test de l'endpoint de prédiction de la propagation géographique
        url = reverse("predict-geographical-spread")
        response = self.client.post(url, self.geographical_data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("geographical_spread", response.data)

        # Test de l'endpoint de prédiction combinée
        url = reverse("predict-combined")
        response = self.client.post(url, self.combined_data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("predictions", response.data)

        # Test de l'endpoint de prévision
        url = reverse("predict-forecast")
        response = self.client.post(url, self.forecast_data, format="json")

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("forecasts", response.data)

    def test_latest_data(self):
        """
        Teste l'endpoint des dernières données
        """
        url = reverse("latest-data")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("latest_data", response.data)
        self.assertIn("geographical_predictions", response.data)

    def test_model_metrics(self):
        """
        Teste l'endpoint des métriques des modèles
        """
        # Créer des métriques de test
        ModelMetrics.objects.create(
            model_type="transmission",
            model_name="Random Forest",
            mse=0.01,
            rmse=0.1,
            mae=0.08,
            r2_score=0.85,
            cv_rmse=0.12,
        )

        url = reverse("model-metrics-summary")
        response = self.client.get(url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("transmission", response.data)
        self.assertEqual(response.data["transmission"]["model_name"], "Random Forest")


class FrontendTests(TestCase):
    """
    Tests pour l'interface utilisateur
    """

    def setUp(self):
        """
        Configuration initiale pour les tests
        """
        self.client = Client()

    def test_index_page(self):
        """
        Teste que la page d'index est accessible
        """
        response = self.client.get("/")

        # Si la page d'index n'est pas configurée, le test sera ignoré
        if response.status_code == 404:
            self.skipTest("La page d'index n'est pas configurée")

        self.assertEqual(response.status_code, 200)


class SettingsTests(TestCase):
    """
    Tests pour les paramètres Django
    """

    def test_deploy_country_setting(self):
        """
        Teste que la variable d'environnement DEPLOY_COUNTRY est correctement lue
        """
        os.environ["DEPLOY_COUNTRY"] = "FR"
        # Recharger les paramètres pour que la variable d'environnement soit prise en compte
        # C'est une astuce de test, ne pas faire ça en production
        import importlib
        import pandemics_project.settings as settings_module
        importlib.reload(settings_module)
        from pandemics_project import settings

        self.assertEqual(settings.DEPLOY_COUNTRY, "FR")
        del os.environ["DEPLOY_COUNTRY"]

        os.environ["DEPLOY_COUNTRY"] = "CH"
        importlib.reload(settings_module) # Recharger à nouveau
        from pandemics_project import settings

        self.assertEqual(settings.DEPLOY_COUNTRY, "CH")
        del os.environ["DEPLOY_COUNTRY"]

        # Teste la valeur par défaut (quand DEPLOY_COUNTRY n'est pas défini)
        importlib.reload(settings_module) # Recharger à nouveau
        from pandemics_project import settings

        self.assertEqual(settings.DEPLOY_COUNTRY, "US")
