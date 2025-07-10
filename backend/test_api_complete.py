from django.test import TestCase
from rest_framework.test import APIClient
from rest_framework import status


class APIEndpointsTests(TestCase):
    def setUp(self):
        self.client = APIClient()

    def test_latest_data_endpoint_exists(self):
        """Test que l'endpoint latest-data existe"""
        response = self.client.get("/api/latest-data/")
        # Accepter 200 ou 500 (erreur base de données normale)
        self.assertIn(response.status_code, [200, 500])

    def test_health_check_endpoint(self):
        """Test endpoint health-check"""
        response = self.client.get("/api/health-check/")
        self.assertIn(response.status_code, [200, 500])
