from django.shortcuts import render
from rest_framework import viewsets, status
from rest_framework.decorators import api_view, action
from rest_framework.response import Response
from rest_framework.views import APIView
from datetime import datetime, timedelta
import pandas as pd
from django.db.models import Max, Sum
from django.utils import timezone

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
        start_date = self.request.query_params.get('start_date', None)
        end_date = self.request.query_params.get('end_date', None)
        
        if location:
            queryset = queryset.filter(location__name=location)
        if virus:
            queryset = queryset.filter(virus__name=virus)
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
    API pour prédire le taux de transmission
    """
    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        serializer = TransmissionPredictionRequestSerializer(data=request.data)
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
            
            try:
                # Faire la prédiction
                transmission_rate = predictor.predict_transmission_rate(location_data)
                
                # Sauvegarder la prédiction dans la base de données
                location = Location.objects.get(id=data['location_id'])
                virus = Virus.objects.get(id=data['virus_id'])
                
                prediction = Prediction(
                    location=location,
                    virus=virus,
                    prediction_date=datetime.now().date() + timedelta(days=7),
                    transmission_rate=transmission_rate,
                    predicted_cases=int(data['cases_7day_lag7'] * (1 + transmission_rate))
                )
                prediction.save()
                
                # Retourner la prédiction
                return Response({
                    'transmission_rate': transmission_rate,
                    'predicted_cases': int(data['cases_7day_lag7'] * (1 + transmission_rate)),
                    'prediction_date': prediction.prediction_date.isoformat()
                })
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

class MortalityPredictionView(APIView):
    """
    API pour prédire le taux de mortalité
    """
    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        serializer = MortalityPredictionRequestSerializer(data=request.data)
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
            
            try:
                # Faire la prédiction
                mortality_rate = predictor.predict_mortality_rate(location_data)
                
                # Sauvegarder la prédiction dans la base de données
                location = Location.objects.get(id=data['location_id'])
                virus = Virus.objects.get(id=data['virus_id'])
                
                # Calculer les cas prédits
                predicted_cases = int(data['cases_7day_lag7'] * (1 + data['cases_growth']))
                predicted_deaths = int(predicted_cases * mortality_rate)
                
                prediction = Prediction.objects.filter(
                    location=location,
                    virus=virus,
                    prediction_date=datetime.now().date() + timedelta(days=7)
                ).first()
                
                if prediction:
                    prediction.mortality_rate = mortality_rate
                    prediction.predicted_deaths = predicted_deaths
                    prediction.save()
                else:
                    prediction = Prediction(
                        location=location,
                        virus=virus,
                        prediction_date=datetime.now().date() + timedelta(days=7),
                        mortality_rate=mortality_rate,
                        predicted_deaths=predicted_deaths
                    )
                    prediction.save()
                
                # Retourner la prédiction
                return Response({
                    'mortality_rate': mortality_rate,
                    'predicted_deaths': predicted_deaths,
                    'prediction_date': prediction.prediction_date.isoformat()
                })
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

class GeographicalSpreadPredictionView(APIView):
    """
    API pour prédire la propagation géographique
    """
    def post(self, request):
        if predictor is None:
            return Response(
                {"error": "Le prédicteur n'est pas disponible"},
                status=status.HTTP_503_SERVICE_UNAVAILABLE
            )
        
        serializer = GeographicalSpreadPredictionRequestSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            
            # Extraire les données pour la prédiction
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
                geographical_spread = predictor.predict_geographical_spread(spread_data)
                
                # Sauvegarder la prédiction dans la base de données
                virus = Virus.objects.get(id=data['virus_id'])
                
                prediction = GeographicalSpreadPrediction(
                    virus=virus,
                    prediction_date=datetime.now().date() + timedelta(days=7),
                    predicted_new_locations=geographical_spread
                )
                prediction.save()
                
                # Retourner la prédiction
                return Response({
                    'geographical_spread': geographical_spread,
                    'prediction_date': prediction.prediction_date.isoformat()
                })
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
                    prediction.save()
                
                # Prédiction pour la propagation géographique
                if 'predictions' in combined and 'geographical_spread' in combined['predictions']:
                    geo_prediction = GeographicalSpreadPrediction(
                        virus=virus,
                        prediction_date=datetime.now().date() + timedelta(days=7),
                        predicted_new_locations=combined['predictions']['geographical_spread']
                    )
                    geo_prediction.save()
                
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
        
        serializer = ForecastRequestSerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            location_id = data['location_id']
            virus_id = data['virus_id']
            weeks = data.get('weeks', 4)
            
            try:
                # Récupérer les données historiques pour la localisation et le virus
                location = Location.objects.get(id=location_id)
                virus = Virus.objects.get(id=virus_id)
                
                # Récupérer les données Worldmeter les plus récentes
                worldmeter_data = Worldmeter.objects.filter(
                    location_id=location_id,
                    virus_id=virus_id
                ).order_by('-date')[:14]  # Récupérer les 14 derniers jours
                
                if not worldmeter_data:
                    return Response(
                        {"error": "Données historiques insuffisantes pour cette localisation et ce virus"},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                
                # Calculer les caractéristiques pour la prédiction
                cases_7day = sum([w.new_cases or 0 for w in worldmeter_data[:7]])
                cases_14day = sum([w.new_cases or 0 for w in worldmeter_data[7:14]])
                deaths_7day = sum([w.new_deaths or 0 for w in worldmeter_data[:7]])
                deaths_14day = sum([w.new_deaths or 0 for w in worldmeter_data[7:14]])
                
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
                    'new_locations_ma3': sum(new_locations[:3]) / 3 if len(new_locations) >= 3 else sum(new_locations) / len(new_locations)
                }
                
                # Générer les prévisions
                forecasts = predictor.generate_forecast(location_data, spread_data, weeks=weeks)
                
                # Sauvegarder les prévisions dans la base de données
                for i, forecast in enumerate(forecasts):
                    if 'predictions' in forecast:
                        prediction_date = datetime.now().date() + timedelta(weeks=i+1)
                        
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
                            prediction.save()
                        
                        # Prédiction pour la propagation géographique
                        if 'geographical_spread' in forecast['predictions']:
                            geo_prediction = GeographicalSpreadPrediction(
                                virus=virus,
                                prediction_date=prediction_date,
                                predicted_new_locations=forecast['predictions']['geographical_spread']
                            )
                            geo_prediction.save()
                
                # Retourner les prévisions
                return Response({
                    'location': location.name,
                    'virus': virus.name,
                    'forecasts': forecasts
                })
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

# Vue pour obtenir les dernières données pour l'interface utilisateur
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

# Vue pour obtenir les métriques des modèles
@api_view(['GET'])
def get_model_metrics(request):
    """
    Récupère les métriques des modèles d'IA
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

# Vue pour la page d'accueil
def index(request):
    return render(request, 'index.html')
