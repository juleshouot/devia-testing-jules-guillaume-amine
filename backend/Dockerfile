FROM python:3.10

WORKDIR /django_api

COPY ./requirements.txt /django_api/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /django_api/

# Assure-toi que le dossier 'models' existe
RUN mkdir -p /django_api/pandemics_app/ml/models

ENV DJANGO_SETTINGS_MODULE=pandemics_project.settings

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:8000"]
