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
            prediction.save()

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
                prediction.save()

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
            prediction.save()

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
                        prediction.save()

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




from django.http import HttpResponse, JsonResponse
import os
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