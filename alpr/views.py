import re
import uuid

from django.core.files.base import ContentFile

from LinMl.settings import LABEL_STUDIO_URL, LABEL_STUDIO_API_KEY
from alpr.serializers import AlprDetectionSerializer
from label_studio_sdk import Client
from django.core.files.storage import FileSystemStorage
from rest_framework.views import APIView
from rest_framework import status
from django.http import JsonResponse
from ultralytics import YOLO
from alpr.models import LicencePlate, NeedToTrain
import requests
import time
import math
import cv2
import numpy as np


class Detect(APIView):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.plate_format = ('[1-9][0-9](?:be|dal|ein|he|jim|lam|mim|nun|qaf|sad|sin|ta|te|vav|ye|zhe)[0-9][0-9][0-9]['
                             '0-9][0-9]')
        object_model = YOLO('alpr/assets/car-model.engine')
        # character_model = YOLO('model/best.pt')

        character_model = YOLO('alpr/assets/yolov8l.engine')
        self.ls = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

        self.chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'be', 'dal', 'ein',
                      'he', 'jim', 'lam', 'mim', 'nun', 'qaf', 'sad', 'sin', 'ta', 'te',
                      'vav', 'ye', 'zhe']
        # self.char_classnames = ['0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 's', '1', 'malul', 'n', 's',
        #                         'sad',
        #                         't',
        #                         'ta',
        #                         'v', 'y', '2'
        #     , '3', '4', '5', '6', '7', '8']
        self.char_classes = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', 'ب', 'د', 'ع', 'ه', 'ج', 'ل', 'م', 'ن',
                             'ق', 'ص', 'س', 'ط', 'ت', 'و', 'ی', 'معلول'
                             ]
        self.object_model = object_model
        self.character_model = character_model

    def post(self, request, *args, **kwargs):
        start_processing = time.time()
        source = request.FILES['upload']
        FileSystemStorage(location="/mnt/HDD/kokhaie/LinMl/Pictures").save(source.name, source)
        # FileSystemStorage(location="/tmp").save(source.name, source)
        licence_plate = LicencePlate.objects.create(initial_image=source)
        # source = '/tmp/{0}'.format(source.name)
        source = '/mnt/HDD/kokhaie/LinMl/Pictures/{0}'.format(source.name)

        img = cv2.imread(source)
        output = self.object_model(source)

        results = []
        # extract bounding box and class names

        for i in output:
            bbox = i.boxes

            for box in bbox:

                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                confs = math.ceil((box.conf[0] * 100)) / 100

                # convert detected object ids to int
                cls_names = int(box.cls[0])

                # check plate to recognize characters with YOLO model
                if cls_names == 1:

                    char_display = []
                    # crop plate from frame
                    plate_img = img[y1:y2, x1:x2]
                    ret, buf = cv2.imencode('.jpg', plate_img)
                    # detect characters of plate with YOLO model

                    plate_output = self.character_model(plate_img, conf=0.5)
                    licence_plate.plate_image.save(str(uuid.uuid4().hex) + '.jpg', ContentFile(buf.tobytes()),
                                                   save=False)

                    # extract bounding box and class names
                    bbox = plate_output[0].boxes.xyxy
                    cls = plate_output[0].boxes.cls

                    # make a dict and sort it from left to right to show the correct characters of plate
                    keys = cls.cpu().numpy().astype(int)
                    print('score: ' + str(plate_output[0].boxes.conf.tolist()))

                    print(keys)
                    values = bbox[:, 0].cpu().numpy().astype(int)
                    print(values)
                    dictionary = list(zip(keys, values))
                    print(dictionary)
                    sorted_list = sorted(dictionary, key=lambda x: x[1])
                    print(sorted_list)

                    # convert all characters to a string
                    for char in sorted_list:
                        char_class = char[0]
                        # char_display.append(plate_output[0].names[char_class])
                        char_display.append(self.chars[char_class])

                    char_result = (''.join(char_display))

                    if not re.match(self.plate_format, char_result):
                        end_processing = time.time()
                        licence_plate.plate_number = char_result
                        licence_plate.plate_image.save(str(uuid.uuid4().hex) + '.jpg', ContentFile(buf.tobytes()),
                                                       save=False)
                        licence_plate.processing_time = str(end_processing - start_processing)
                        licence_plate.save()

                        # sending unreadable plate to Label Studio
                        requests.post(LABEL_STUDIO_URL, headers={
                            "Authorization": "Token {0}".format(LABEL_STUDIO_API_KEY)},
                                      files={"file": licence_plate.plate_image})

                        # saving the initial and plate image on django admin
                        need2train = NeedToTrain.objects.create(initial_image=request.FILES['upload'])
                        need2train.plate_image.save(str(uuid.uuid4().hex) + '.jpg', ContentFile(buf.tobytes()),
                                                    save=False)

                        return JsonResponse({}, status=status.HTTP_200_OK)

                    # calculate average score of all characters
                    score = math.ceil(np.mean(plate_output[0].boxes.conf.tolist()) * 100) / 100
                    print('score: ' + str(plate_output[0].boxes.conf.tolist()))

                    # plate bounding box dictionary
                    box = {'xmin': x1, 'ymin': y1, 'xmax': x2, 'ymax': y2}
                    results.append({'box': box, 'plate': str(char_result),
                                    'score': score,
                                    'dscore': confs})

                    if score >= 0.7:
                        licence_plate.identified = True

                    # saving plate information to the Model
                    licence_plate.plate_number = char_result
                    licence_plate.score = str(score)
                    licence_plate.dscore = str(confs)

                    # save image to the Model
                    ret, buf = cv2.imencode('.jpg', img)
                    licence_plate.processed_image.save(str(uuid.uuid4().hex) + '.jpg', ContentFile(buf.tobytes()),
                                                       save=False)

        end_processing = time.time()
        licence_plate.processing_time = str(end_processing - start_processing)
        licence_plate.save()

        # serialize plate result for returning to ZM
        serializer = AlprDetectionSerializer(
            data={'results': results, 'processing_time': float(end_processing - start_processing)})
        if serializer.is_valid(raise_exception=True):
            print(serializer.data)
            return JsonResponse(serializer.data, status=status.HTTP_200_OK)


# simple health check for the alpr api
class AlprHealthCheck(APIView):
    # permission_classes = [IsAuthenticated]

    def get(self, request, *args, **kwargs):
        return JsonResponse({'status': 'healthy'}, status=status.HTTP_200_OK)
