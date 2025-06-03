from django.db import models


class CovidData(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateField()
    location = models.CharField(max_length=100)
    total_cases = models.IntegerField()
    daily_cases = models.IntegerField()
    active_cases = models.IntegerField()
    total_deaths = models.IntegerField()
    daily_deaths = models.IntegerField()

    class Meta:
        db_table = 'worldometer_coronavirus_daily_data'

    def __str__(self):
        return f"{self.location} - {self.date}"


class MonkeyPoxData(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateField()
    location = models.CharField(max_length=100)
    total_cases = models.IntegerField()
    daily_cases = models.IntegerField()
    total_deaths = models.IntegerField()
    daily_deaths = models.IntegerField()

    class Meta:
        db_table = 'owid_monkeypox_data'

    def __str__(self):
        return f"{self.location} - {self.date}"


class Location(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=45)
    iso_code = models.CharField(max_length=45, null=True, blank=True)

    class Meta:
        db_table = 'pandemics.location'
        managed = False

    def __str__(self):
        return self.name


class Virus(models.Model):
    id = models.AutoField(primary_key=True)
    name = models.CharField(max_length=45)

    db_table = 'pandemics.virus'
    managed = False

    def __str__(self):
        return self.name


class Worldmeter(models.Model):
    id = models.AutoField(primary_key=True)
    date = models.DateField(null=True, blank=True)
    total_cases = models.IntegerField(null=True, blank=True)
    total_deaths = models.IntegerField(null=True, blank=True)
    new_cases = models.IntegerField(null=True, blank=True)
    new_deaths = models.IntegerField(null=True, blank=True)
    new_cases_smoothed = models.IntegerField(null=True, blank=True)
    new_deaths_smoothed = models.IntegerField(null=True, blank=True)
    new_cases_per_million = models.IntegerField(null=True, blank=True)
    total_cases_per_million = models.IntegerField(null=True, blank=True)
    new_cases_smoothed_per_million = models.IntegerField(null=True, blank=True)
    new_deaths_per_million = models.IntegerField(null=True, blank=True)
    total_deaths_per_million = models.IntegerField(null=True, blank=True)
    new_deaths_smoothed_per_million = models.IntegerField(null=True, blank=True)
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    virus = models.ForeignKey(Virus, on_delete=models.CASCADE)

    db_table = 'pandemics.Worldmeter'
    managed = False

    def __str__(self):
        return f"{self.location.name} - {self.virus.name} - {self.date}"


# Nouveaux modèles pour les prédictions
class Prediction(models.Model):
    id = models.AutoField(primary_key=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    virus = models.ForeignKey(Virus, on_delete=models.CASCADE)
    prediction_date = models.DateField()
    transmission_rate = models.FloatField(null=True, blank=True)
    mortality_rate = models.FloatField(null=True, blank=True)
    predicted_cases = models.IntegerField(null=True, blank=True)
    predicted_deaths = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ['-prediction_date']

    def __str__(self):
        return f"{self.location.name} - {self.virus.name} - {self.prediction_date}"


class GeographicalSpreadPrediction(models.Model):
    id = models.AutoField(primary_key=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    prediction_date = models.DateField()
    virus = models.ForeignKey(Virus, on_delete=models.CASCADE)
    predicted_new_locations = models.IntegerField()

    class Meta:
        ordering = ['-prediction_date']

    def __str__(self):
        return f"{self.virus.name} - {self.prediction_date} - {self.predicted_new_locations} nouvelles localisations"


class ModelMetrics(models.Model):
    id = models.AutoField(primary_key=True)
    model_type = models.CharField(max_length=50)  # 'transmission', 'mortality', 'geographical_spread'
    model_name = models.CharField(max_length=100)  # 'Random Forest', 'Gradient Boosting', etc.
    timestamp = models.DateTimeField(auto_now_add=True)
    mse = models.FloatField()
    rmse = models.FloatField()
    mae = models.FloatField()
    r2_score = models.FloatField()
    cv_rmse = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.model_type} - {self.model_name} - {self.timestamp.strftime('%Y-%m-%d')}"