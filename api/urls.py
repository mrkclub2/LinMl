from django.contrib import admin
from django.urls import path
from api.views import HealthCheck,Login

urlpatterns = [
    path('logins/', Login.as_view()),
    path('health/', HealthCheck.as_view()),
    path('detect/object/', admin.site.urls),
]
