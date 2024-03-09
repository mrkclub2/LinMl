from django.db import models


class LicencePlate(models.Model):
    plate_number = models.CharField(max_length=15)
    identified = models.BooleanField(default=False)
    initial_image = models.ImageField(upload_to='initial')
    plate_image = models.ImageField(upload_to='plates')
    processed_image = models.ImageField(upload_to='processed_image')
    score = models.CharField(max_length=5, blank=True, null=True)
    dscore = models.CharField(max_length=5, blank=True, null=True)
    time = models.DateTimeField(auto_now_add=True, blank=True, null=True)
    processing_time = models.CharField(max_length=30, blank=True, null=True)

    def __str__(self):
        if self.plate_number:
            return self.plate_number + ' --> ' + str(self.time.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            return str(self.id) + ') unknown plate number' + ' --> ' + str(self.time.strftime("%Y-%m-%d %H:%M:%S"))


class NeedToTrain(models.Model):
    initial_image = models.ImageField(upload_to='need2train-car')
    plate_image = models.ImageField(upload_to='need2train-plate')
