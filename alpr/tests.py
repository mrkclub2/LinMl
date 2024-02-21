import json

from django.test import TestCase
from rest_framework.test import APITestCase, APIClient
from django.urls import reverse


class ModelTests(APITestCase):
    def test_detection(self):
        url = reverse("plate-reader")
        data = {'upload': open('alpr/assets/car1.jpg', 'rb')}

        response = self.client.post(url, data)
        print('adding response' + str(json.loads(response.content)['results'][0]))
