import os
import cv2
import uuid
import numpy as np
import argparse
import copy
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse
from pyzm.ml.detect_sequence import DetectSequence
import pyzm.helpers.utils as pyzmutils
import ast
import pyzm.api as zmapi

class Detect(APIView):
    def post(self, request, *args, **kwargs):
        DetectSequence()
        return JsonResponse({'status': 'Persian Plate Detection'}, status=status.HTTP_200_OK)
