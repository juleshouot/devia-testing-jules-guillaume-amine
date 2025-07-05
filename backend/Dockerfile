# Dockerfile ULTRA simple comme MSPR2 qui fonctionne
FROM python:3.10

WORKDIR /django_api

COPY ./requirements.txt /django_api/
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /django_api/pandemics_app/ml/models

ENV DJANGO_SETTINGS_MODULE=pandemics_project.settings

EXPOSE 8000

# Commande simple SANS collectstatic qui pose problème
CMD ["sh", "-c", "sleep 10 && python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]