import uuid

from django.contrib.auth.models import User
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken

from LinMl.settings import USERNAME, PASSWORD


# generates access_token and refresh_token for requests
# this view is not fully implemented yet

class Login(APIView):
    def post(self, request, *args, **kwargs):
        username = request.get('username', None)
        password = request.get('password', None)

        if username is None or password is None:
            return JsonResponse({'status': 'wrong credential'}, status=status.HTTP_403_FORBIDDEN)

        if username == USERNAME and password == PASSWORD:
            user, is_user_created = User.objects.get_or_create(username='admin')

            if is_user_created:
                user.password = uuid.uuid4().hex
                user.save()

            refresh_token = RefreshToken().for_user(user)
            tokens = {'refresh': str(refresh_token),
                      'access': str(refresh_token.access_token)}
            return JsonResponse(tokens, status=status.HTTP_200_OK)

        else:
            return JsonResponse({'status': 'wrong credential'}, status=status.HTTP_403_FORBIDDEN)


# simple health_check just for django response
class HealthCheck(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        return JsonResponse({'status': 'healthy'}, status=status.HTTP_200_OK)


# object_detection Apiview
class DetectObject(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, *args, **kwargs):
        return JsonResponse({'status': 'detect object view created'}, status=status.HTTP_200_OK)
