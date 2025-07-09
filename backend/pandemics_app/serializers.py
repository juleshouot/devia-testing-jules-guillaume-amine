from rest_framework import serializers
from .models import (
    CovidData,
    MonkeyPoxData,
    Location,
    Virus,
    Worldmeter,
    Prediction,
    GeographicalSpreadPrediction,
    ModelMetrics,
)


class CovidDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = CovidData
        fields = "__all__"


class MonkeyPoxDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonkeyPoxData
        fields = "__all__"


class LocationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Location
        fields = "__all__"


class VirusSerializer(serializers.ModelSerializer):
    class Meta:
        model = Virus
        fields = "__all__"


class WorldmeterSerializer(serializers.ModelSerializer):
    location_name = serializers.CharField(source="location.name", read_only=True)
    virus_name = serializers.CharField(source="virus.name", read_only=True)

    class Meta:
        model = Worldmeter
        fields = "__all__"


# Nouveaux sérialiseurs pour les prédictions
class PredictionSerializer(serializers.ModelSerializer):
    location_name = serializers.CharField(source="location.name", read_only=True)
    virus_name = serializers.CharField(source="virus.name", read_only=True)

    class Meta:
        model = Prediction
        fields = "__all__"


class GeographicalSpreadPredictionSerializer(serializers.ModelSerializer):
    virus_name = serializers.CharField(source="virus.name", read_only=True)

    class Meta:
        model = GeographicalSpreadPrediction
        fields = "__all__"


class ModelMetricsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelMetrics
        fields = "__all__"


# Sérialiseurs pour les requêtes de prédiction
class TransmissionPredictionRequestSerializer(serializers.Serializer):
    location_id = serializers.IntegerField()
    virus_id = serializers.IntegerField()
    cases_7day_lag7 = serializers.IntegerField()
    cases_7day_lag14 = serializers.IntegerField()
    deaths_7day_lag7 = serializers.IntegerField()
    deaths_7day_lag14 = serializers.IntegerField()
    cases_growth = serializers.FloatField()
    deaths_growth = serializers.FloatField()


class MortalityPredictionRequestSerializer(serializers.Serializer):
    location_id = serializers.IntegerField()
    virus_id = serializers.IntegerField()
    cases_7day_lag7 = serializers.IntegerField()
    cases_7day_lag14 = serializers.IntegerField()
    deaths_7day_lag7 = serializers.IntegerField()
    deaths_7day_lag14 = serializers.IntegerField()
    cases_growth = serializers.FloatField()
    deaths_growth = serializers.FloatField()


class GeographicalSpreadPredictionRequestSerializer(serializers.Serializer):
    virus_id = serializers.IntegerField()
    new_locations_lag1 = serializers.IntegerField()
    new_locations_lag2 = serializers.IntegerField()
    new_locations_lag3 = serializers.IntegerField()
    new_locations_lag4 = serializers.IntegerField()
    new_locations_ma2 = serializers.FloatField()
    new_locations_ma3 = serializers.FloatField()


class CombinedPredictionRequestSerializer(serializers.Serializer):
    location_id = serializers.IntegerField()
    virus_id = serializers.IntegerField()
    cases_7day_lag7 = serializers.IntegerField()
    cases_7day_lag14 = serializers.IntegerField()
    deaths_7day_lag7 = serializers.IntegerField()
    deaths_7day_lag14 = serializers.IntegerField()
    cases_growth = serializers.FloatField()
    deaths_growth = serializers.FloatField()
    new_locations_lag1 = serializers.IntegerField()
    new_locations_lag2 = serializers.IntegerField()
    new_locations_lag3 = serializers.IntegerField()
    new_locations_lag4 = serializers.IntegerField()
    new_locations_ma2 = serializers.FloatField()
    new_locations_ma3 = serializers.FloatField()
    weeks = serializers.IntegerField(required=False, default=4)


class ForecastRequestSerializer(serializers.Serializer):
    location_id = serializers.IntegerField()
    virus_id = serializers.IntegerField()
    weeks = serializers.IntegerField(required=False, default=4)
