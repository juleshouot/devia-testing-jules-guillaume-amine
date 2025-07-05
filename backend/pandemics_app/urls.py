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
    path('predict/geographical-spread/', views.GeographicalSpreadPredictionView.as_view(),
         name='predict-geographical-spread'),
    path('predict/combined/', views.CombinedPredictionView.as_view(), name='predict-combined'),
    path('predict/forecast/', views.ForecastView.as_view(), name='predict-forecast'),

    # Endpoints pour les données de l'interface utilisateur
    path('latest-data/', views.get_latest_data, name='latest-data'),
    path('aggregated-data/', views.aggregated_data, name='aggregated-data'),

    # 🔧 ANCIENS endpoints pour les métriques (gardés pour compatibilité)
    path('model-metrics-summary/', views.get_model_metrics, name='model-metrics-summary'),

    # 🆕 NOUVEAUX endpoints améliorés
    path('model-metrics-complete/', views.get_model_metrics_complete, name='model-metrics-complete'),
    path('model-metrics/sync/', views.sync_model_metrics, name='sync-model-metrics'),
    path('prediction-summary/', views.get_prediction_summary, name='prediction-summary'),
    path('health-check/', views.health_check, name='health-check'),
    path('location-details/', views.get_location_details, name='location-details'),

    # Endpoints pour les visualisations
    path('visualizations/generate/', views.generate_visualizations, name='generate_visualizations'),
    path('visualizations/', views.list_visualizations, name='list_visualizations'),
    path('visualizations/<str:viz_name>/', views.interactive_visualization, name='interactive_visualization'),
    path('validate-model/', views.validate_model_predictions, name='validate-model'),
    path('available-countries/', views.get_available_countries_for_validation, name='available-countries'),
    path('validate-transmission-model/', views.validate_transmission_model_predictions,
         name='validate-transmission-model'),
]