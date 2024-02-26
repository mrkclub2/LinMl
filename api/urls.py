from django.contrib import admin
from django.urls import path
from api.views import HealthCheck, Login
from alpr.views import Detect, AlprHealthCheck

urlpatterns = [
    path('logins/', Login.as_view()),
    path('api-health/', HealthCheck.as_view()),
    path('plate-reader', Detect.as_view(), name='plate-reader'),
    path('alpr-health', AlprHealthCheck.as_view(), name='alpr-health'),

]
