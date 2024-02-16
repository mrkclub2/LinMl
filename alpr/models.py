from django.db import models


class LicencePlate(models.Model):
    plate_number = models.CharField(max_length=15)
    identified = models.BooleanField(default=False)
    initial_image = models.ImageField(upload_to='initial')
    plate_image = models.ImageField(upload_to='plates')
    processed_image = models.ImageField(upload_to='processed_image')

    def __str__(self):
        if self.plate_number:
            return self.plate_number
        else:
            return 'unknown plate number'
