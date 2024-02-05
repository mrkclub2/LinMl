from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse

class Login(APIView):
    def post(self, request, *args, **kwargs):
        return JsonResponse({'status': 'login view created'}, status=status.HTTP_200_OK)


class HealthCheck(APIView):
    def get(self, request, *args, **kwargs):
        return JsonResponse({'status': 'healthy'}, status=status.HTTP_200_OK)


class DetectObject(APIView):
    def post(self, request, *args, **kwargs):
        return JsonResponse({'status': 'detect object view created'}, status=status.HTTP_200_OK)

