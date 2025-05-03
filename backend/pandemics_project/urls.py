from django.contrib import admin
from django.urls import path, include, re_path
from django.http import JsonResponse
from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi
from pandemics_app.views import LocationViewSet, VirusViewSet, WorldmeterViewSet
def home(request):
    return JsonResponse({"message": "Bienvenue sur l'API Pandémies ! Consultez /api/ pour voir les données."})

schema_view = get_schema_view(
    openapi.Info(
        title="Pandemics API",
        default_version='v1',
        description="Documentation de l'API pour la gestion des pandémies.",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contact@pandemics.com"),
        license=openapi.License(name="MIT License"),
    ),
    public=True,
    permission_classes=(permissions.AllowAny,),
)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('pandemics_app.urls')),
    path('', home),  # Ajoute une page d'accueil simple
    re_path(r'^swagger(?P<format>\.json|\.yaml)$', schema_view.without_ui(cache_timeout=0), name='schema-json'),
    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),  # ✅ Ajout Swagger
    path('redoc/', schema_view.with_ui('redoc', cache_timeout=0), name='schema-redoc'),
]
