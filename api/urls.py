from django.contrib import admin
from django.urls import path
from api.views import HealthCheck, Login
from alpr.views import Detect

urlpatterns = [
    path('logins/', Login.as_view()),
    path('health/', HealthCheck.as_view()),
    path('plate-reader', Detect.as_view(), name='plate-reader'),

]
