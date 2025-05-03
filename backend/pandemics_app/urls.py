from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'covid', views.CovidDataViewSet)
router.register(r'monkeypox', views.MonkeyPoxDataViewSet)
router.register(r'locations', views.LocationViewSet)
router.register(r'viruses', views.VirusViewSet)
router.register(r'worldmeter', views.WorldmeterViewSet)
router.register(r'predictions', views.PredictionViewSet)
router.register(r'geographical-predictions', views.GeographicalSpreadPredictionViewSet)
router.register(r'model-metrics', views.ModelMetricsViewSet)

urlpatterns = [
    path('', include(router.urls)),
    # Endpoints pour les prédictions
    path('predict/transmission/', views.TransmissionPredictionView.as_view(), name='predict-transmission'),
    path('predict/mortality/', views.MortalityPredictionView.as_view(), name='predict-mortality'),
    path('predict/geographical-spread/', views.GeographicalSpreadPredictionView.as_view(), name='predict-geographical-spread'),
    path('predict/combined/', views.CombinedPredictionView.as_view(), name='predict-combined'),
    path('predict/forecast/', views.ForecastView.as_view(), name='predict-forecast'),
    # Endpoints pour les données de l'interface utilisateur
    path('latest-data/', views.get_latest_data, name='latest-data'),
    path('model-metrics-summary/', views.get_model_metrics, name='model-metrics-summary'),
]
