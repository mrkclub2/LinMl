import json

from django.test import TestCase
from rest_framework.test import APITestCase, APIClient
from rest_framework import status
from django.urls import reverse


class ModelTests(APITestCase):
    def test_detection(self):
        url = reverse("plate-reader")

        # plate recognition api test
        data = {'upload': open('alpr/assets/car1.jpg', 'rb')}
        response = self.client.post(url, data)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(json.loads(response.content)['results'][0]['plate'], '69s55613')

        #

