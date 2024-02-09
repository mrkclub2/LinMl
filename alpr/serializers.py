from rest_framework import serializers


class PlateBox(serializers.Serializer):
    xmin = serializers.IntegerField()
    ymin = serializers.IntegerField()
    xmax = serializers.IntegerField()
    ymax = serializers.IntegerField()


class AlprResult(serializers.Serializer):
    box = PlateBox(many=False)
    plate = serializers.CharField()
    score = serializers.FloatField()
    dscore = serializers.FloatField()


class AlprDetectionSerializer(serializers.Serializer):
    processing_time = serializers.CharField()
    results = AlprResult(many=True)
