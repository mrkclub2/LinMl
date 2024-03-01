from django.contrib import admin
from alpr.models import LicencePlate


@admin.register(LicencePlate)
class Licence(admin.ModelAdmin):
    model = LicencePlate
    list_display = ['id', 'plate_number', 'identified', 'score', 'dscore', 'time', 'processing_time',
                    'initial_image', 'plate_image']
