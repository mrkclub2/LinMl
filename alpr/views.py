from rest_framework.views import APIView
from django.http import JsonResponse
from ultralytics import YOLO
import time
import math
import cv2


class Detect(APIView):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        start_time = time.time()

        object_model = YOLO('alpr/assets/best.pt')
        character_model = YOLO('alpr/assets/yolov8n_char_new.pt')
        char_classnames = ['0', '9', 'b', 'd', 'ein', 'ein', 'g', 'gh', 'h', 'n', 's', '1', 'malul', 'n', 's', 'sad',
                           't',
                           'ta',
                           'v', 'y', '2'
            , '3', '4', '5', '6', '7', '8']
        self.object_model = object_model
        self.character_model = character_model
        self.char_classnames = char_classnames
        end_time = time.time()
        print(end_time - start_time)

    def post(self, request, *args, **kwargs):
        source = 'alpr/assets/car1.jpg'
        output = self.object_model(source, show=False, conf=0.75)
        img = cv2.imread(source)

        # extract bounding box and class names
        for i in output:
            bbox = i.boxes
            for box in bbox:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
                confs = math.ceil((box.conf[0] * 100)) / 100
                cls_names = int(box.cls[0])
                if cls_names == 1:
                    cv2.putText(img, f'{confs}', (max(40, x2 + 5), max(40, y2 + 5)), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=0.5, color=(0, 20, 255), thickness=1, lineType=cv2.LINE_AA)
                elif cls_names == 0:
                    cv2.putText(img, f'{confs}', (max(40, x1), max(40, y1)), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=0.6, color=(0, 20, 255), thickness=1, lineType=cv2.LINE_AA)

                # check plate to recognize characters with yolov8n model
                if cls_names == 1:
                    char_display = []
                    # crop plate from frame
                    plate_img = img[y1:y2, x1:x2]
                    # detect characters of plate with yolov8n model
                    plate_output = self.character_model(plate_img, conf=0.4)

                    # extract bounding box and class names
                    bbox = plate_output[0].boxes.xyxy
                    cls = plate_output[0].boxes.cls
                    # make a dict and sort it from left to right to show the correct characters of plate
                    keys = cls.cpu().numpy().astype(int)
                    values = bbox[:, 0].cpu().numpy().astype(int)
                    dictionary = list(zip(keys, values))
                    sorted_list = sorted(dictionary, key=lambda x: x[1])
                    # convert all characters to a string
                    for i in sorted_list:
                        char_class = i[0]
                        # char_display.append(plate_output[0].names[char_class])
                        char_display.append(self.char_classnames[char_class])
                    char_result = (''.join(char_display))

                    # just show the correct characters in output
                    if len(char_display) == 8:
                        cv2.line(img, (max(40, x1 - 25), max(40, y1 - 10)), (x2 + 25, y1 - 10), (0, 0, 0), 20,
                                 lineType=cv2.LINE_AA)
                        cv2.putText(img, char_result, (max(40, x1 - 15), max(40, y1 - 5)),
                                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255), thickness=1,
                                    lineType=cv2.LINE_AA)

                        return JsonResponse({'plate number:': str(char_result)})
