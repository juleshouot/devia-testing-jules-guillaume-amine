import os
from pathlib import Path

# ========================================
# CONFIGURATION DE BASE - MSPR 2 (qui marche)
# ========================================

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Security settings
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "your-secret-key")
DEBUG = os.getenv("DJANGO_DEBUG", "True") == "True"
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")

# ========================================
# APPLICATIONS INSTALLÉES - MSPR 2 conservé
# ========================================

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "django_filters",
    "pandemics_app",
    "drf_yasg",
    "corsheaders",  # MSPR 2
]

# ========================================
# MIDDLEWARE - MSPR 2 conservé
# ========================================

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "corsheaders.middleware.CorsMiddleware",  # MSPR 2
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

# ========================================
# CONFIGURATION CORS - MSPR 2 conservé
# ========================================

CORS_ALLOW_ALL_ORIGINS = True  # Pour le développement

# ========================================
# ROOT URL ET TEMPLATES - MSPR 2
# ========================================

ROOT_URLCONF = "pandemics_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "pandemics_project.wsgi.application"

# ========================================
# BASE DE DONNÉES - MSPR 2 conservé
# ========================================

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql",
        "NAME": os.getenv("POSTGRES_DB", "pandemies"),
        "USER": os.getenv("POSTGRES_USER", "user"),
        "PASSWORD": os.getenv("POSTGRES_PASSWORD", "guigui"),
        "HOST": os.getenv("POSTGRES_HOST", "postgres"),
        "PORT": os.getenv("POSTGRES_PORT", "5432"),
        "OPTIONS": {"options": "-c search_path=pandemics,public"},
    }
}

# ========================================
# REST FRAMEWORK - MSPR 2 conservé
# ========================================

REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 10,
}

# ========================================
# VALIDATION DES MOTS DE PASSE - MSPR 2
# ========================================

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ========================================
# INTERNATIONALISATION - MSPR 2 + AJOUT MSPR 3 SIMPLE
# ========================================

# Configuration par défaut
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_L10N = True
USE_TZ = True

# AJOUT SIMPLE MSPR 3 - Configuration pays (SANS COMPLEXITÉ)
DEPLOY_COUNTRY = os.getenv("DEPLOY_COUNTRY", "US")

# Configuration simple par pays (sans logique complexe)
if DEPLOY_COUNTRY == "FR":
    LANGUAGE_CODE = "fr-fr"
    TIME_ZONE = "Europe/Paris"
elif DEPLOY_COUNTRY == "CH":
    LANGUAGE_CODE = "fr-ch"
    TIME_ZONE = "Europe/Zurich"
elif DEPLOY_COUNTRY == "US":
    LANGUAGE_CODE = "en-us"
    TIME_ZONE = "America/New_York"

# ========================================
# FICHIERS STATIQUES - MSPR 2
# ========================================

STATIC_URL = "/static/"

# ========================================
# CONFIGURATION PAR DÉFAUT
# ========================================

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ========================================
# AJOUT MSPR 3 - Variables simples pour l'application
# ========================================
# Configuration Assistant Vocal Simple
VOICE_ASSISTANT_SIMPLE = {
    "ENABLED": True,
    "SUPPORTED_LANGUAGES": ["fr", "en", "de", "it"],
    "DEFAULT_LANGUAGE": "fr",
    "RATE_LIMIT_COMMANDS_PER_MINUTE": 30,
    "LOG_COMMANDS": False,  # True seulement pour debug
}

# Configuration par pays
if DEPLOY_COUNTRY == "FR":
    VOICE_ASSISTANT_SIMPLE["LOG_COMMANDS"] = False  # RGPD
    VOICE_ASSISTANT_SIMPLE["SUPPORTED_LANGUAGES"] = ["fr"]
elif DEPLOY_COUNTRY == "CH":
    VOICE_ASSISTANT_SIMPLE["SUPPORTED_LANGUAGES"] = ["fr", "de", "it"]
elif DEPLOY_COUNTRY == "US":
    VOICE_ASSISTANT_SIMPLE["LOG_COMMANDS"] = True
"""

# Variable pays pour les templates (simple)
COUNTRY_CONFIG = {
    'code': DEPLOY_COUNTRY,
    'name': {
        'US': 'United States',
        'FR': 'France',
        'CH': 'Switzerland'
    }.get(DEPLOY_COUNTRY, 'Unknown'),
}

# Mode API technique (simple)
API_MODE = os.getenv('API_MODE', 'STANDARD')

# Information de version
APP_VERSION = '2.0.0'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
    
# Log simple de démarrage
print(f"🚀 Django démarré - Pays: {DEPLOY_COUNTRY} | Environnement: {ENVIRONMENT}")

# ========================================
# PAS DE LOGGING COMPLEXE QUI CAUSE DES ERREURS !
# ========================================
# On utilise la configuration de logging par défaut de Django
# Pas de configuration JSON qui cause des problèmes
"""
