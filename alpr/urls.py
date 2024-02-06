from django.contrib import admin
from django.urls import path
from alpr.views import Detect

urlpatterns = [

    path('detect/object/', Detect.as_view()),
]
