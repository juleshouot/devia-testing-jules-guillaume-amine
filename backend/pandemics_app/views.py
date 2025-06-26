from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.views import APIView
from datetime import datetime, timedelta
import pandas as pd
from django.db.models import Max, Sum, Avg
from django.db.models.functions import TruncDate
from django.utils import timezone
import os
import joblib

from .models import (
    CovidData, MonkeyPoxData, Location, Virus, Worldmeter,
    Prediction, GeographicalSpreadPrediction, ModelMetrics
)
from .serializers import (
    CovidDataSerializer, MonkeyPoxDataSerializer, LocationSerializer,
    VirusSerializer, WorldmeterSerializer, PredictionSerializer,
    GeographicalSpreadPredictionSerializer, ModelMetricsSerializer,
    TransmissionPredictionRequestSerializer, MortalityPredictionRequestSerializer,
    GeographicalSpreadPredictionRequestSerializer, CombinedPredictionRequestSerializer,
    ForecastRequestSerializer
)

# Import du prédicteur
from .ml.predictor import PandemicPredictor

# Création d'une instance globale du prédicteur
try:
    predictor = PandemicPredictor()
except Exception as e:
    print(f"Erreur lors de l'initialisation du prédicteur: {e}")
    predictor = None


class CovidDataViewSet(viewsets.ModelViewSet):
    queryset = CovidData.objects.all()
    serializer_class = CovidDataSerializer


class MonkeyPoxDataViewSet(viewsets.ModelViewSet):
    queryset = MonkeyPoxData.objects.all()
    serializer_class = MonkeyPoxDataSerializer


class LocationViewSet(viewsets.ModelViewSet):
    queryset = Location.objects.all()
    serializer_class = LocationSerializer


class VirusViewSet(viewsets.ModelViewSet):
    queryset = Virus.objects.all()
    serializer_class = VirusSerializer


class WorldmeterViewSet(viewsets.ModelViewSet):
    queryset = Worldmeter.objects.all()
    serializer_class = WorldmeterSerializer

    def get_queryset(self):
        queryset = Worldmeter.objects.all()
        location = self.request.query_params.get('location', None)
        virus = self.request.query_params.get('virus', None)
        location_name = self.request.query_params.get('location_name', None)
        virus_name = self.request.query_params.get('virus_name', None)
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        date = self.request.query_params.get('date', None)

        if location:
            queryset = queryset.filter(location_id=location)
        if virus:
            queryset = queryset.filter(virus_id=virus)
        if location_name:
            queryset = queryset.filter(location__name=location_name)
        if virus_name:
            queryset = queryset.filter(virus__name=virus_name)
        if date:
            queryset = queryset.filter(date=date)
        if start_date:
            queryset = queryset.filter(date__gte=start_date)
        if end_date:
            queryset = queryset.filter(date__lte=end_date)

        return queryset


# Nouvelles vues pour les prédictions

class PredictionViewSet(viewsets.ModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer

    def get_queryset(self):
        queryset = Prediction.objects.all()
        location = self.request.query_params.get('location', None)
        virus = self.request.query_params.get('virus', None)
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)

        if location:
            queryset = queryset.filter(location__name=location)
        if virus:
            queryset = queryset.filter(virus__name=virus)
        if start_date:
            queryset = queryset.filter(prediction_date__gte=start_date)
        if end_date:
            queryset = queryset.filter(prediction_date__lte=end_date)

        return queryset


class GeographicalSpreadPredictionViewSet(viewsets.ModelViewSet):
    queryset = GeographicalSpreadPrediction.objects.all()
    serializer_class = GeographicalSpreadPredictionSerializer

    def get_queryset(self):
        queryset = GeographicalSpreadPrediction.objects.all()
        virus = self.request.query_params.get('virus', None)
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)

        if virus:
            queryset = queryset.filter(virus__name=virus)
        if start_date:
            queryset = queryset.filter(prediction_date__gte=start_date)
        if end_date:
            queryset = queryset.filter(prediction_date__lte=end_date)

        return queryset


class ModelMetricsViewSet(viewsets.ModelViewSet):
    queryset = ModelMetrics.objects.all()
    serializer_class = ModelMetricsSerializer

    def get_queryset(self):
        queryset = ModelMetrics.objects.all()
        model_type = self.request.query_params.get('model_type', None)

        if model_type:
            queryset = queryset.filter(model_type=model_type)

        return queryset


# API pour les prédictions

class TransmissionPredictionView(APIView):
    """
    API pour prédire le taux de transmission - Version simplifiée
    """

    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Validation simple
        location_id = request.data.get('location_id')
        virus_id = request.data.get('virus_id')

        if not location_id or not virus_id:
            return Response(
                {"error": "location_id and virus_id are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            location = Location.objects.get(id=location_id)
            virus = Virus.objects.get(id=virus_id)

            # Faire la prédiction directement avec le prédicteur
            rt_pred = predictor.predict_rt(location.name, virus.name)

            if rt_pred is None:
                return Response(
                    {"error": f"Impossible de générer une prédiction pour {virus.name} à {location.name}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Sauvegarder la prédiction
            prediction = Prediction(
                location=location,
                virus=virus,
                prediction_date=datetime.now().date() + timedelta(days=7),
                transmission_rate=rt_pred
            )
            # prediction.save()  # DÉSACTIVÉ TEMPORAIREMENT

            return Response({
                'location': location.name,
                'virus': virus.name,
                'transmission_rate': rt_pred,
                'prediction_date': prediction.prediction_date.isoformat()
            })

        except Location.DoesNotExist:
            return Response(
                {"error": f"Location with id {location_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Virus.DoesNotExist:
            return Response(
                {"error": f"Virus with id {virus_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class MortalityPredictionView(APIView):
    """
    API pour prédire le taux de mortalité - Version simplifiée
    """

    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Validation simple
        location_id = request.data.get('location_id')
        virus_id = request.data.get('virus_id')

        if not location_id or not virus_id:
            return Response(
                {"error": "location_id and virus_id are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            location = Location.objects.get(id=location_id)
            virus = Virus.objects.get(id=virus_id)

            # Faire la prédiction directement avec le prédicteur
            mortality_pred = predictor.predict_mortality_ratio(location.name, virus.name)

            if mortality_pred is None:
                return Response(
                    {"error": f"Impossible de générer une prédiction de mortalité pour {virus.name} à {location.name}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Sauvegarder la prédiction
            prediction, created = Prediction.objects.get_or_create(
                location=location,
                virus=virus,
                prediction_date=datetime.now().date() + timedelta(days=7),
                defaults={'mortality_rate': mortality_pred}
            )

            if not created:
                prediction.mortality_rate = mortality_pred
                # prediction.save()  # DÉSACTIVÉ TEMPORAIREMENT

            return Response({
                'location': location.name,
                'virus': virus.name,
                'mortality_rate': mortality_pred,
                'prediction_date': prediction.prediction_date.isoformat()
            })

        except Location.DoesNotExist:
            return Response(
                {"error": f"Location with id {location_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Virus.DoesNotExist:
            return Response(
                {"error": f"Virus with id {virus_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class GeographicalSpreadPredictionView(APIView):
    """
    API pour prédire la propagation géographique - Version simplifiée
    """

    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Validation simple
        virus_id = request.data.get('virus_id')

        if not virus_id:
            return Response(
                {"error": "virus_id is required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            virus = Virus.objects.get(id=virus_id)

            # Faire la prédiction directement avec le prédicteur
            spread_pred = predictor.predict_geographical_spread(virus.name)

            if spread_pred is None:
                return Response(
                    {"error": f"Impossible de générer une prédiction de propagation pour {virus.name}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Sauvegarder la prédiction
            prediction = GeographicalSpreadPrediction(
                virus=virus,
                prediction_date=datetime.now().date() + timedelta(days=7),
                predicted_new_locations=spread_pred
            )
            # prediction.save()  # DÉSACTIVÉ TEMPORAIREMENT

            return Response({
                'virus': virus.name,
                'predicted_new_locations': spread_pred,
                'prediction_date': prediction.prediction_date.isoformat()
            })

        except Virus.DoesNotExist:
            return Response(
                {"error": f"Virus with id {virus_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class CombinedPredictionView(APIView):
    """
    API pour faire des prédictions combinées
    """

    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        serializer = CombinedPredictionRequestSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data

            # Extraire les données pour la prédiction
            location_data = {
                'cases_7day_lag7': data['cases_7day_lag7'],
                'cases_7day_lag14': data['cases_7day_lag14'],
                'deaths_7day_lag7': data['deaths_7day_lag7'],
                'deaths_7day_lag14': data['deaths_7day_lag14'],
                'cases_growth': data['cases_growth'],
                'deaths_growth': data['deaths_growth']
            }

            spread_data = {
                'new_locations_lag1': data['new_locations_lag1'],
                'new_locations_lag2': data['new_locations_lag2'],
                'new_locations_lag3': data['new_locations_lag3'],
                'new_locations_lag4': data['new_locations_lag4'],
                'new_locations_ma2': data['new_locations_ma2'],
                'new_locations_ma3': data['new_locations_ma3']
            }

            try:
                # Faire la prédiction
                combined = predictor.predict_combined(location_data, spread_data)

                # Sauvegarder les prédictions dans la base de données
                location = Location.objects.get(id=data['location_id'])
                virus = Virus.objects.get(id=data['virus_id'])

                # Prédiction pour la localisation
                if 'predictions' in combined and 'transmission_rate' in combined['predictions']:
                    prediction = Prediction(
                        location=location,
                        virus=virus,
                        prediction_date=datetime.now().date() + timedelta(days=7),
                        transmission_rate=combined['predictions']['transmission_rate'],
                        mortality_rate=combined['predictions'].get('mortality_rate'),
                        predicted_cases=combined['predictions'].get('predicted_cases_next_week'),
                        predicted_deaths=combined['predictions'].get('predicted_deaths_next_week')
                    )
                    # prediction.save()  # DÉSACTIVÉ TEMPORAIREMENT

                # Prédiction pour la propagation géographique
                if 'predictions' in combined and 'geographical_spread' in combined['predictions']:
                    geo_prediction = GeographicalSpreadPrediction(
                        virus=virus,
                        prediction_date=datetime.now().date() + timedelta(days=7),
                        predicted_new_locations=combined['predictions']['geographical_spread']
                    )
                    geo_# prediction.save()  # DÉSACTIVÉ TEMPORAIREMENT

                # Retourner les prédictions
                return Response(combined)
            except Exception as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
        else:
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )


class ForecastView(APIView):
    """
    API pour générer des prévisions sur plusieurs semaines
    """

    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Simplifier le serializer pour n'accepter que location_id, virus_id et weeks
        location_id = request.data.get('location_id')
        virus_id = request.data.get('virus_id')
        weeks = request.data.get('weeks', 4)

        # Validation simple
        if not location_id or not virus_id:
            return Response(
                {"error": "location_id and virus_id are required"},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Récupérer les objets location et virus
            location = Location.objects.get(id=location_id)
            virus = Virus.objects.get(id=virus_id)

            # Récupérer les données Worldmeter les plus récentes
            worldmeter_data = Worldmeter.objects.filter(
                location_id=location_id,
                virus_id=virus_id
            ).order_by('-date')[:21]  # Récupérer les 21 derniers jours

            if len(worldmeter_data) < 14:
                return Response(
                    {
                        "error": f"Données historiques insuffisantes pour {location.name} et {virus.name}. Minimum requis: 14 jours, trouvé: {len(worldmeter_data)} jours"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Convertir en liste pour faciliter l'accès
            data_list = list(worldmeter_data)

            # Calculer les caractéristiques pour la prédiction
            cases_7day = sum([w.new_cases or 0 for w in data_list[:7]])
            cases_14day = sum([w.new_cases or 0 for w in data_list[7:14]])
            deaths_7day = sum([w.new_deaths or 0 for w in data_list[:7]])
            deaths_14day = sum([w.new_deaths or 0 for w in data_list[7:14]])

            cases_growth = (cases_7day / cases_14day) - 1 if cases_14day > 0 else 0
            deaths_growth = (deaths_7day / deaths_14day) - 1 if deaths_14day > 0 else 0

            # Récupérer les données de propagation géographique
            geo_predictions = GeographicalSpreadPrediction.objects.filter(
                virus_id=virus_id
            ).order_by('-prediction_date')[:4]

            new_locations = [p.predicted_new_locations for p in geo_predictions]
            while len(new_locations) < 4:
                new_locations.append(0)

            # Préparer les données pour la prédiction
            location_data = {
                'cases_7day_lag7': cases_7day,
                'cases_7day_lag14': cases_14day,
                'deaths_7day_lag7': deaths_7day,
                'deaths_7day_lag14': deaths_14day,
                'cases_growth': cases_growth,
                'deaths_growth': deaths_growth
            }

            spread_data = {
                'new_locations_lag1': new_locations[0],
                'new_locations_lag2': new_locations[1] if len(new_locations) > 1 else 0,
                'new_locations_lag3': new_locations[2] if len(new_locations) > 2 else 0,
                'new_locations_lag4': new_locations[3] if len(new_locations) > 3 else 0,
                'new_locations_ma2': sum(new_locations[:2]) / 2 if len(new_locations) >= 2 else new_locations[0],
                'new_locations_ma3': sum(new_locations[:3]) / 3 if len(new_locations) >= 3 else sum(
                    new_locations) / len(new_locations)
            }

            # Générer les prévisions
            forecasts = []

            # Pour l'instant, générer des prédictions simples semaine par semaine
            for week in range(weeks):
                try:
                    # Prédire Rt et mortalité pour cette localisation
                    rt_pred = predictor.predict_rt(location.name, virus.name)
                    mortality_pred = predictor.predict_mortality_ratio(location.name, virus.name)

                    if rt_pred is None:
                        rt_pred = 1.0  # Valeur par défaut
                    if mortality_pred is None:
                        mortality_pred = 0.02  # Valeur par défaut

                    # Calculer les cas et décès prédits
                    base_cases = cases_7day
                    predicted_cases = int(base_cases * (rt_pred ** (week + 1)))
                    predicted_deaths = int(predicted_cases * mortality_pred)

                    forecast = {
                        'week': week + 1,
                        'predictions': {
                            'transmission_rate': round(rt_pred, 3),
                            'mortality_rate': round(mortality_pred, 4),
                            'predicted_cases_next_week': predicted_cases,
                            'predicted_deaths_next_week': predicted_deaths
                        }
                    }
                    forecasts.append(forecast)

                except Exception as e:
                    # En cas d'erreur, ajouter une prédiction par défaut
                    forecast = {
                        'week': week + 1,
                        'predictions': {
                            'transmission_rate': 1.0,
                            'mortality_rate': 0.02,
                            'predicted_cases_next_week': int(cases_7day * 0.9),
                            'predicted_deaths_next_week': int(cases_7day * 0.02)
                        }
                    }
                    forecasts.append(forecast)

            # Sauvegarder les prévisions dans la base de données
            for i, forecast in enumerate(forecasts):
                if 'predictions' in forecast:
                    prediction_date = datetime.now().date() + timedelta(weeks=i + 1)

                    # Prédiction pour la localisation
                    if 'transmission_rate' in forecast['predictions']:
                        prediction = Prediction(
                            location=location,
                            virus=virus,
                            prediction_date=prediction_date,
                            transmission_rate=forecast['predictions'].get('transmission_rate'),
                            mortality_rate=forecast['predictions'].get('mortality_rate'),
                            predicted_cases=forecast['predictions'].get('predicted_cases_next_week'),
                            predicted_deaths=forecast['predictions'].get('predicted_deaths_next_week')
                        )
                        # prediction.save()  # DÉSACTIVÉ TEMPORAIREMENT

            # Retourner les prévisions
            return Response({
                'location': location.name,
                'virus': virus.name,
                'weeks_requested': weeks,
                'data_points_used': len(data_list),
                'forecasts': forecasts
            })

        except Location.DoesNotExist:
            return Response(
                {"error": f"Location with id {location_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Virus.DoesNotExist:
            return Response(
                {"error": f"Virus with id {virus_id} not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
def get_latest_data(request):
    """
    Récupère les dernières données pour l'interface utilisateur
    """
    try:
        # Récupérer les dernières données Worldmeter pour chaque localisation et virus
        latest_data = []

        for virus in Virus.objects.all():
            for location in Location.objects.all():
                latest_worldmeter = Worldmeter.objects.filter(
                    location=location,
                    virus=virus
                ).order_by('-date').first()

                if latest_worldmeter:
                    # Récupérer les prédictions pour cette localisation et ce virus
                    predictions = Prediction.objects.filter(
                        location=location,
                        virus=virus,
                        prediction_date__gte=datetime.now().date()
                    ).order_by('prediction_date')

                    prediction_data = []
                    for pred in predictions:
                        prediction_data.append({
                            'date': pred.prediction_date.isoformat(),
                            'transmission_rate': pred.transmission_rate,
                            'mortality_rate': pred.mortality_rate,
                            'predicted_cases': pred.predicted_cases,
                            'predicted_deaths': pred.predicted_deaths
                        })

                    # Ajouter les données à la liste
                    latest_data.append({
                        'location': location.name,
                        'virus': virus.name,
                        'date': latest_worldmeter.date.isoformat(),
                        'total_cases': latest_worldmeter.total_cases,
                        'total_deaths': latest_worldmeter.total_deaths,
                        'new_cases': latest_worldmeter.new_cases,
                        'new_deaths': latest_worldmeter.new_deaths,
                        'predictions': prediction_data
                    })

        # Récupérer les prédictions de propagation géographique
        geo_predictions = {}
        for virus in Virus.objects.all():
            virus_geo_predictions = GeographicalSpreadPrediction.objects.filter(
                virus=virus,
                prediction_date__gte=datetime.now().date()
            ).order_by('prediction_date')

            if virus_geo_predictions:
                geo_predictions[virus.name] = [{
                    'date': pred.prediction_date.isoformat(),
                    'predicted_new_locations': pred.predicted_new_locations
                } for pred in virus_geo_predictions]

        return Response({
            'latest_data': latest_data,
            'geographical_predictions': geo_predictions
        })
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# 🔧 ANCIEN endpoint pour les métriques (gardé pour compatibilité)
@api_view(['GET'])
def get_model_metrics(request):
    """
    Récupère les métriques des modèles d'IA (version basique)
    """
    try:
        metrics = {}

        for model_type in ['transmission', 'mortality', 'geographical_spread']:
            model_metrics = ModelMetrics.objects.filter(model_type=model_type).order_by('-timestamp').first()

            if model_metrics:
                metrics[model_type] = {
                    'model_name': model_metrics.model_name,
                    'mse': model_metrics.mse,
                    'rmse': model_metrics.rmse,
                    'mae': model_metrics.mae,
                    'r2_score': model_metrics.r2_score,
                    'cv_rmse': model_metrics.cv_rmse
                }

        return Response(metrics)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# 🆕 NOUVEAUX endpoints améliorés

@api_view(['GET'])
def get_model_metrics_complete(request):
    """
    Récupère les métriques des modèles depuis la DB ET depuis les fichiers joblib
    """
    try:
        metrics = {}

        # 1. Essayer de récupérer depuis la base de données
        for model_type in ['transmission', 'mortality', 'geographical_spread']:
            model_metrics = ModelMetrics.objects.filter(model_type=model_type).order_by('-timestamp').first()

            if model_metrics:
                metrics[model_type] = {
                    'source': 'database',
                    'model_name': model_metrics.model_name,
                    'mse': model_metrics.mse,
                    'rmse': model_metrics.rmse,
                    'mae': model_metrics.mae,
                    'r2_score': model_metrics.r2_score,
                    'cv_rmse': model_metrics.cv_rmse,
                    'timestamp': model_metrics.timestamp.isoformat()
                }

        # 2. Si pas de données en base, essayer de lire depuis les fichiers joblib
        if not metrics:
            MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")
            
            if os.path.exists(MODEL_DIR):
                model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_metadata.joblib')]
                
                for file in model_files:
                    try:
                        metadata_path = os.path.join(MODEL_DIR, file)
                        metadata = joblib.load(metadata_path)
                        
                        model_type = metadata.get('model_type', 'unknown')
                        model_name = metadata.get('model_name', 'unknown')
                        
                        if model_type not in metrics:
                            metrics[model_type] = {
                                'source': 'joblib_file',
                                'model_name': model_name,
                                'mse': metadata.get('mse', 0),
                                'rmse': metadata.get('rmse', 0),
                                'mae': metadata.get('mae', 0),
                                'r2_score': metadata.get('r2_score', 0),
                                'cv_rmse': metadata.get('cv_rmse', 0),
                                'timestamp': None
                            }
                    except Exception as e:
                        print(f"Erreur lors du chargement de {file}: {e}")

        return Response({
            'metrics': metrics,
            'count': len(metrics),
            'available_types': list(metrics.keys())
        })
        
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['POST'])
def sync_model_metrics(request):
    """
    Synchronise les métriques des fichiers joblib vers la base de données
    """
    try:
        MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")
        
        if not os.path.exists(MODEL_DIR):
            return Response(
                {"error": f"Répertoire de modèles non trouvé: {MODEL_DIR}"},
                status=status.HTTP_404_NOT_FOUND
            )

        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('_metadata.joblib')]
        synced_count = 0
        errors = []

        for file in model_files:
            try:
                metadata_path = os.path.join(MODEL_DIR, file)
                metadata = joblib.load(metadata_path)
                
                model_type = metadata.get('model_type', 'unknown')
                model_name = metadata.get('model_name', 'unknown')
                
                # Créer ou mettre à jour l'entrée
                metric_entry, created = ModelMetrics.objects.update_or_create(
                    model_type=model_type,
                    model_name=model_name,
                    defaults={
                        'mse': metadata.get('mse', 0),
                        'rmse': metadata.get('rmse', 0),
                        'mae': metadata.get('mae', 0),
                        'r2_score': metadata.get('r2_score', 0),
                        'cv_rmse': metadata.get('cv_rmse', 0)
                    }
                )
                
                synced_count += 1
                
            except Exception as e:
                errors.append(f"Erreur avec {file}: {str(e)}")

        return Response({
            'message': f'{synced_count} métriques synchronisées',
            'synced_count': synced_count,
            'total_files': len(model_files),
            'errors': errors
        })
        
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_prediction_summary(request):
    """
    Récupère un résumé complet des prédictions récentes
    """
    try:
        # Paramètres de requête
        days = request.query_params.get('days', 7)
        location_name = request.query_params.get('location_name', None)
        virus_name = request.query_params.get('virus_name', None)
        
        # Date limite
        date_limit = datetime.now().date() - timedelta(days=int(days))
        
        # Prédictions par localisation
        predictions_query = Prediction.objects.filter(
            timestamp__gte=date_limit
        ).select_related('location', 'virus')
        
        if location_name:
            predictions_query = predictions_query.filter(location__name=location_name)
        if virus_name:
            predictions_query = predictions_query.filter(virus__name=virus_name)
            
        predictions = predictions_query.order_by('-timestamp')[:50]
        
        # Prédictions géographiques
        geo_predictions_query = GeographicalSpreadPrediction.objects.filter(
            timestamp__gte=date_limit
        ).select_related('virus')
        
        if virus_name:
            geo_predictions_query = geo_predictions_query.filter(virus__name=virus_name)
            
        geo_predictions = geo_predictions_query.order_by('-timestamp')[:20]
        
        # Sérialiser les données
        predictions_data = []
        for pred in predictions:
            predictions_data.append({
                'id': pred.id,
                'location': pred.location.name,
                'virus': pred.virus.name,
                'prediction_date': pred.prediction_date.isoformat(),
                'transmission_rate': pred.transmission_rate,
                'mortality_rate': pred.mortality_rate,
                'predicted_cases': pred.predicted_cases,
                'predicted_deaths': pred.predicted_deaths,
                'timestamp': pred.timestamp.isoformat()
            })
            
        geo_predictions_data = []
        for geo_pred in geo_predictions:
            geo_predictions_data.append({
                'id': geo_pred.id,
                'virus': geo_pred.virus.name,
                'prediction_date': geo_pred.prediction_date.isoformat(),
                'predicted_new_locations': geo_pred.predicted_new_locations,
                'timestamp': geo_pred.timestamp.isoformat()
            })
        
        # Statistiques
        stats = {
            'total_predictions': len(predictions_data),
            'total_geo_predictions': len(geo_predictions_data),
            'locations_covered': len(set(p['location'] for p in predictions_data)),
            'viruses_covered': len(set(p['virus'] for p in predictions_data)),
            'date_range': f"Derniers {days} jours",
            'avg_transmission_rate': sum(p['transmission_rate'] for p in predictions_data if p['transmission_rate']) / max(len([p for p in predictions_data if p['transmission_rate']]), 1),
            'avg_mortality_rate': sum(p['mortality_rate'] for p in predictions_data if p['mortality_rate']) / max(len([p for p in predictions_data if p['mortality_rate']]), 1)
        }
        
        return Response({
            'predictions': predictions_data,
            'geographical_predictions': geo_predictions_data,
            'statistics': stats,
            'filters_applied': {
                'days': days,
                'location_name': location_name,
                'virus_name': virus_name
            }
        })
        
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def health_check(request):
    """
    Vérifie que tous les composants fonctionnent correctement
    """
    try:
        checks = {}
        
        # 1. Vérifier la base de données
        try:
            location_count = Location.objects.count()
            virus_count = Virus.objects.count()
            worldmeter_count = Worldmeter.objects.count()
            prediction_count = Prediction.objects.count()
            
            checks['database'] = {
                'status': 'OK',
                'locations': location_count,
                'viruses': virus_count,
                'worldmeter_records': worldmeter_count,
                'predictions': prediction_count
            }
        except Exception as e:
            checks['database'] = {'status': 'ERROR', 'error': str(e)}
        
        # 2. Vérifier le prédicteur
        try:
            if predictor is not None:
                model_count = len(predictor.models)
                checks['predictor'] = {
                    'status': 'OK',
                    'models_loaded': model_count,
                    'available_types': list(set(k[0] for k in predictor.models.keys()))
                }
            else:
                checks['predictor'] = {'status': 'ERROR', 'error': 'Predictor not initialized'}
        except Exception as e:
            checks['predictor'] = {'status': 'ERROR', 'error': str(e)}
        
        # 3. Vérifier les fichiers de modèles
        try:
            MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml", "models")
            
            if os.path.exists(MODEL_DIR):
                model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.joblib')]
                checks['model_files'] = {
                    'status': 'OK',
                    'model_files_count': len(model_files),
                    'model_directory': MODEL_DIR
                }
            else:
                checks['model_files'] = {'status': 'WARNING', 'error': 'Model directory not found'}
        except Exception as e:
            checks['model_files'] = {'status': 'ERROR', 'error': str(e)}
        
        # Status global
        all_statuses = [check.get('status', 'ERROR') for check in checks.values()]
        overall_status = 'OK' if all(s == 'OK' for s in all_statuses) else 'WARNING' if any(s == 'WARNING' for s in all_statuses) else 'ERROR'
        
        return Response({
            'overall_status': overall_status,
            'timestamp': datetime.now().isoformat(),
            'checks': checks
        })
        
    except Exception as e:
        return Response(
            {"error": str(e), "overall_status": "ERROR"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# Nouvelle vue pour les données agrégées
@api_view(['GET'])
def aggregated_data(request):
    """
    Récupère les données agrégées par jour pour une localisation et/ou un virus
    """
    try:
        # Récupérer les paramètres de requête
        location_name = request.query_params.get('location_name')
        virus_name = request.query_params.get('virus_name')
        start_date = request.query_params.get('start_date')
        end_date = request.query_params.get('end_date')

        # Construire la requête
        query = Worldmeter.objects.all()

        # Appliquer les filtres
        if location_name:
            query = query.filter(location__name=location_name)
        if virus_name:
            query = query.filter(virus__name=virus_name)
        if start_date:
            query = query.filter(date__gte=start_date)
        if end_date:
            query = query.filter(date__lte=end_date)

        # Agréger les données par jour
        aggregated = query.annotate(
            date_only=TruncDate('date')
        ).values(
            'date_only'
        ).annotate(
            sum_new_cases=Sum('new_cases'),
            sum_new_deaths=Sum('new_deaths'),
            avg_new_cases_per_million=Avg('new_cases_per_million'),
            avg_new_deaths_per_million=Avg('new_deaths_per_million')
        ).order_by('date_only')

        # Convertir le résultat en liste de dictionnaires
        result = list(aggregated)

        # Formater les dates pour la réponse JSON
        for item in result:
            item['date'] = item.pop('date_only').isoformat()

        return Response(result)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


# Vue pour la page d'accueil
def index(request):
    return render(request, 'index.html')


# Vues pour les visualisations
from django.http import HttpResponse, JsonResponse
from django.conf import settings


@api_view(['GET'])
def generate_visualizations(request):
    """
    Génère ou régénère les visualisations interactives
    """
    try:
        from .ml.interactive_viz import generate_interactive_visualizations
        result = generate_interactive_visualizations()
        return Response(result)
    except Exception as e:
        return Response(
            {"error": str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def list_visualizations(request):
    """
    Liste toutes les visualisations disponibles
    """
    vis_dir = os.path.join(settings.BASE_DIR, 'static', 'visualizations')

    if not os.path.exists(vis_dir):
        return Response({'visualizations': []})

    html_files = [f for f in os.listdir(vis_dir) if f.endswith('.html')]
    json_files = [f for f in os.listdir(vis_dir) if f.endswith('.json')]

    visualizations = []

    for html_file in html_files:
        name = html_file.replace('.html', '').replace('_', ' ').title()
        visualizations.append({
            'name': name,
            'html_path': f'/static/visualizations/{html_file}',
            'json_path': f'/static/visualizations/{html_file.replace(".html", ".json")}'
            if html_file.replace(".html", ".json") in json_files else None
        })

    return Response({'visualizations': visualizations})


def interactive_visualization(request, viz_name):
    """
    Sert une visualisation HTML interactive
    """
    viz_path = os.path.join(settings.BASE_DIR, 'static', 'visualizations', f"{viz_name}.html")

    if os.path.exists(viz_path):
        with open(viz_path, 'r') as f:
            content = f.read()
        return HttpResponse(content)
    else:
        return HttpResponse('Visualisation non trouvée', status=404)


@api_view(['GET'])
def get_location_details(request):
    """
    Récupère les détails d'une localisation (incluant population)
    """
    location_name = request.query_params.get('name')

    if not location_name:
        return Response({"error": "Le paramètre 'name' est requis"}, status=400)

    try:
        location = Location.objects.get(name=location_name)
        return Response({
            'id': location.id,
            'name': location.name,
            'iso_code': location.iso_code,
            'population': getattr(location, 'population', None)
        })
    except Location.DoesNotExist:
        return Response({"error": "Localisation non trouvée"}, status=404)
    except Exception as e:
        return Response({"error": str(e)}, status=500)


@api_view(['POST'])
def validate_model_predictions(request):
    """
    API pour exécuter la validation du modèle sur une localisation et période donnée
    """
    try:
        # Récupérer les paramètres
        location_name = request.data.get('location_name')
        start_date = request.data.get('start_date', '2020-03-01')
        end_date = request.data.get('end_date', '2022-05-01')
        max_results = int(request.data.get('max_results', 100))

        if not location_name:
            return Response(
                {"error": "location_name est requis"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Importer les fonctions du validator
        from .ml.validator import load_mortality_model, get_covid_data, create_features_from_data, \
            calculate_actual_mortality
        import pandas as pd
        import numpy as np

        # Charger le modèle
        model, scaler, feature_cols = load_mortality_model()
        if model is None:
            return Response(
                {"error": "Impossible de charger le modèle de mortalité"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Récupérer les données COVID pour cette localisation
        df = get_covid_data(location_name, start_date, end_date)
        if df is None or len(df) < 50:
            return Response(
                {"error": f"Données insuffisantes pour {location_name} sur la période {start_date} - {end_date}"},
                status=status.HTTP_404_NOT_FOUND
            )

        results = []

        # Générer les prédictions de validation
        for i in range(30, min(len(df) - 14, max_results + 30)):
            # Créer les features avec les données jusqu'au point actuel
            features_dict = create_features_from_data(df, i)
            if features_dict is None:
                continue

            try:
                # Faire la prédiction
                features_array = np.array([features_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
                if scaler is not None:
                    features_array = scaler.transform(features_array)

                predicted_mortality = model.predict(features_array)[0]
                predicted_mortality = max(0, min(predicted_mortality, 1))  # Limiter à [0,1]

                # Calculer le taux de mortalité réel
                actual_mortality = calculate_actual_mortality(df, i)

                if actual_mortality is not None and actual_mortality > 0:
                    current_row = df.iloc[i]
                    error = abs(predicted_mortality - actual_mortality)
                    rel_error = error / max(actual_mortality, 0.001)

                    # Déterminer la qualité
                    if rel_error < 0.15:
                        quality = "Excellente"
                        quality_class = "excellent"
                    elif rel_error < 0.30:
                        quality = "Bonne"
                        quality_class = "good"
                    elif rel_error < 0.50:
                        quality = "Correcte"
                        quality_class = "fair"
                    else:
                        quality = "Mauvaise"
                        quality_class = "poor"

                    results.append({
                        'date': current_row['date'].strftime('%Y-%m-%d'),
                        'predicted_mortality': round(predicted_mortality * 100, 3),  # En pourcentage
                        'actual_mortality': round(actual_mortality * 100, 3),  # En pourcentage
                        'error_absolute': round(error * 100, 3),  # En pourcentage
                        'error_relative': round(rel_error * 100, 1),  # En pourcentage
                        'quality': quality,
                        'quality_class': quality_class,
                        # Données de contexte
                        'total_cases': int(current_row.get('total_cases', 0)),
                        'total_deaths': int(current_row.get('total_deaths', 0)),
                        'new_cases_7d': int(df.iloc[max(0, i - 6):i + 1]['new_cases'].sum()),
                        'new_deaths_7d': int(df.iloc[max(0, i - 6):i + 1]['new_deaths'].sum())
                    })

            except Exception as e:
                continue

        if len(results) < 10:
            return Response(
                {"error": f"Pas assez de prédictions valides générées pour {location_name}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Limiter le nombre de résultats
        results = results[:max_results]

        # Calculer les statistiques globales
        df_results = pd.DataFrame(results)
        stats = {
            'total_predictions': len(results),
            'period_start': results[0]['date'],
            'period_end': results[-1]['date'],
            'avg_error_relative': round(df_results['error_relative'].mean(), 1),
            'median_error_relative': round(df_results['error_relative'].median(), 1),
            'excellent_count': len(df_results[df_results['quality'] == 'Excellente']),
            'good_count': len(df_results[df_results['quality'] == 'Bonne']),
            'fair_count': len(df_results[df_results['quality'] == 'Correcte']),
            'poor_count': len(df_results[df_results['quality'] == 'Mauvaise']),
            'cases_range': f"{df_results['total_cases'].min():,} - {df_results['total_cases'].max():,}",
            'deaths_range': f"{df_results['total_deaths'].min():,} - {df_results['total_deaths'].max():,}"
        }

        return Response({
            'location': location_name,
            'start_date': start_date,
            'end_date': end_date,
            'predictions': results,
            'statistics': stats,
            'model_info': {
                'features_count': len(feature_cols),
                'model_type': 'Random Forest Mortality Model'
            }
        })

    except Exception as e:
        return Response(
            {"error": f"Erreur lors de la validation: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_available_countries_for_validation(request):
    """
    Récupère les pays avec suffisamment de données COVID pour la validation
    """
    try:
        from .ml.validator import get_engine
        from sqlalchemy import text
        import pandas as pd

        engine = get_engine()

        query = text("""
                     SELECT l.name                         as location,
                            COUNT(*)                       as data_points,
                            COALESCE(SUM(w.new_cases), 0)  as total_cases,
                            COALESCE(SUM(w.new_deaths), 0) as total_deaths,
                            MIN(w.date)                    as start_date,
                            MAX(w.date)                    as end_date
                     FROM worldmeter w
                              JOIN location l ON w.location_id = l.id
                              JOIN virus v ON w.virus_id = v.id
                     WHERE v.name = 'COVID'
                       AND w.date BETWEEN '2020-03-01' AND '2022-05-01'
                     GROUP BY l.name
                     HAVING COUNT(*) >= 100
                        AND COALESCE(SUM(w.new_cases), 0) >= 1000
                     ORDER BY COALESCE(SUM(w.new_cases), 0) DESC
                     """)

        df = pd.read_sql(query, engine)

        countries = []
        for _, row in df.iterrows():
            # Gérer les valeurs NaN et None de manière sécurisée
            total_cases = row['total_cases']
            total_deaths = row['total_deaths']
            data_points = row['data_points']

            # Convertir en sécurité
            total_cases_int = 0
            total_deaths_int = 0
            data_points_int = 0

            try:
                if pd.notna(total_cases) and total_cases is not None:
                    total_cases_int = int(float(total_cases))
            except (ValueError, TypeError):
                total_cases_int = 0

            try:
                if pd.notna(total_deaths) and total_deaths is not None:
                    total_deaths_int = int(float(total_deaths))
            except (ValueError, TypeError):
                total_deaths_int = 0

            try:
                if pd.notna(data_points) and data_points is not None:
                    data_points_int = int(float(data_points))
            except (ValueError, TypeError):
                data_points_int = 0

            # Déterminer la qualité des données
            if data_points_int > 400:
                data_quality = 'Excellente'
            elif data_points_int > 200:
                data_quality = 'Bonne'
            else:
                data_quality = 'Correcte'

            countries.append({
                'name': str(row['location']),
                'data_points': data_points_int,
                'total_cases': total_cases_int,
                'total_deaths': total_deaths_int,
                'start_date': row['start_date'].strftime('%Y-%m-%d'),
                'end_date': row['end_date'].strftime('%Y-%m-%d'),
                'data_quality': data_quality
            })

        return Response({
            'countries': countries,
            'total_count': len(countries),
            'message': f'{len(countries)} pays disponibles pour la validation'
        })

    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Erreur dans get_available_countries_for_validation: {error_detail}")

        return Response(
            {"error": str(e), "detail": error_detail},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )

@api_view(['POST'])
def validate_transmission_model_predictions(request):
    """
    API pour exécuter la validation du modèle de transmission sur une localisation et période donnée
    """
    try:
        # Récupérer les paramètres
        location_name = request.data.get('location_name')
        start_date = request.data.get('start_date', '2020-03-01')
        end_date = request.data.get('end_date', '2022-05-01')
        max_results = int(request.data.get('max_results', 100))

        if not location_name:
            return Response(
                {"error": "location_name est requis"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Import des fonctions spécifiques transmission
        from .ml.transmission_validator import (
            load_transmission_model, get_covid_data, create_features_from_data,
            calculate_actual_rt
        )
        import pandas as pd
        import numpy as np

        # Charger le modèle de transmission
        model, scaler, feature_cols = load_transmission_model()
        if model is None:
            return Response(
                {"error": "Impossible de charger le modèle de transmission"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )

        # Récupérer les données COVID pour cette localisation
        df = get_covid_data(location_name, start_date, end_date)
        if df is None or len(df) < 50:
            return Response(
                {"error": f"Données insuffisantes pour {location_name} sur la période {start_date} - {end_date}"},
                status=status.HTTP_404_NOT_FOUND
            )

        results = []

        # Générer les prédictions de validation Rt
        for i in range(30, min(len(df) - 14, max_results + 30)):
            # Créer les features avec les données jusqu'au point actuel
            features_dict = create_features_from_data(df, i)
            if features_dict is None:
                continue

            try:
                # Faire la prédiction Rt
                features_array = np.array([features_dict.get(col, 0) for col in feature_cols]).reshape(1, -1)
                if scaler is not None:
                    features_array = scaler.transform(features_array)

                predicted_rt = model.predict(features_array)[0]
                predicted_rt = max(0, min(predicted_rt, 10))  # Limiter à [0,10]

                # Calculer le Rt réel futur (7-14 jours)
                actual_rt = calculate_actual_rt(df, i)

                if actual_rt is not None and actual_rt > 0:
                    current_row = df.iloc[i]
                    error = abs(predicted_rt - actual_rt)
                    rel_error = error / max(actual_rt, 0.1)

                    # Déterminer la qualité (même système que mortalité)
                    if rel_error < 0.15:
                        quality = "Excellente"
                        quality_class = "excellent"
                    elif rel_error < 0.30:
                        quality = "Bonne"
                        quality_class = "good"
                    elif rel_error < 0.50:
                        quality = "Correcte"
                        quality_class = "fair"
                    else:
                        quality = "Mauvaise"
                        quality_class = "poor"

                    # Déterminer l'état épidémique futur
                    if actual_rt > 1.5:
                        future_epidemic_state = "Forte croissance"
                    elif actual_rt > 1.0:
                        future_epidemic_state = "Croissance"
                    elif actual_rt > 0.8:
                        future_epidemic_state = "Stable"
                    else:
                        future_epidemic_state = "Déclin"

                    results.append({
                        'date': current_row['date'].strftime('%Y-%m-%d'),
                        'predicted_rt': round(predicted_rt, 3),
                        'actual_rt_future': round(actual_rt, 3),
                        'error_absolute': round(error, 3),
                        'error_relative': round(rel_error * 100, 1),  # En pourcentage
                        'quality': quality,
                        'quality_class': quality_class,
                        'future_epidemic_state': future_epidemic_state,
                        # Données de contexte
                        'total_cases': int(current_row.get('total_cases', 0)),
                        'total_deaths': int(current_row.get('total_deaths', 0)),
                        'new_cases_7d': int(df.iloc[max(0, i - 6):i + 1]['new_cases'].sum()),
                        'new_deaths_7d': int(df.iloc[max(0, i - 6):i + 1]['new_deaths'].sum())
                    })

            except Exception as e:
                continue

        if len(results) < 10:
            return Response(
                {"error": f"Pas assez de prédictions Rt valides générées pour {location_name}"},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Limiter le nombre de résultats
        results = results[:max_results]

        # Calculer les statistiques globales
        df_results = pd.DataFrame(results)
        stats = {
            'total_predictions': len(results),
            'period_start': results[0]['date'],
            'period_end': results[-1]['date'],
            'avg_error_relative': round(df_results['error_relative'].mean(), 1),
            'median_error_relative': round(df_results['error_relative'].median(), 1),
            'excellent_count': len(df_results[df_results['quality'] == 'Excellente']),
            'good_count': len(df_results[df_results['quality'] == 'Bonne']),
            'fair_count': len(df_results[df_results['quality'] == 'Correcte']),
            'poor_count': len(df_results[df_results['quality'] == 'Mauvaise']),
            'avg_predicted_rt': round(df_results['predicted_rt'].mean(), 2),
            'avg_actual_rt': round(df_results['actual_rt_future'].mean(), 2),
            'epidemic_periods': len(df_results[df_results['actual_rt_future'] > 1]),
            'controlled_periods': len(df_results[df_results['actual_rt_future'] <= 1])
        }

        return Response({
            'location': location_name,
            'start_date': start_date,
            'end_date': end_date,
            'predictions': results,
            'statistics': stats,
            'model_info': {
                'features_count': len(feature_cols),
                'model_type': 'Random Forest Transmission Model (Rt)',
                'prediction_type': 'Rt futur (7-14 jours à l\'avance)'
            }
        })

    except Exception as e:
        return Response(
            {"error": f"Erreur lors de la validation transmission: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )