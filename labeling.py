import math
import numpy as np
import cv2
import os
from ultralytics import YOLO


class Labeling():
    def __init__(self):
        self.car_plate_model = YOLO('alpr/assets/best.pt')
        self.char_model = YOLO('alpr/assets/bounding_box_model.pt')

        self.main()

    def main(self):
        image_path = 'alpr/assets/car1.jpg'

        for file in os.listdir("/home/kokhaie/Desktop/mr.soltani-v1/"):
            if file.endswith(".jpg"):
                plate_img = self.crop_plate(os.path.join("/home/kokhaie/Desktop/mr.soltani-v1/", file))
                self.bounding_box_detector(plate_img, file)

    # crop plate images from cars
    def crop_plate(self, image_path):
        source = cv2.imread(image_path)
        results = self.car_plate_model(image_path)

        for result in results:
            bbox = result.boxes

            for box in bbox:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2)

                # check if plate detected or not
                if box.cls[0] == 1:
                    plate_img = source[y1:y2, x1:x2]
                    return plate_img

    # finding character bounding box
    def bounding_box_detector(self, plate_image, image_name):
        results = self.char_model(plate_image, conf=0.6)
        # extract bounding box and class names
        bbox = results[0].boxes.xyxy

        # make a dict and sort it from left to right to show the correct characters of plate

        values = sorted(bbox[:, 0].tolist())

        chars = self.plate_number_exporter(image_name)

        bbox = results[0].boxes
        # print(bbox.index(value))
        if len(bbox) != 8:
            # send it to label studio
            raise ValueError('can not detect all bounding boxes or it is more than it should be')

        else:
            for i, value in enumerate(values):
                x1, y1, x2, y2 = bbox[results[0].boxes.xyxy[:, 0].tolist().index(value)].xyxy[0]
                x1, y1, x2, y2 = math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2)

                cv2.putText(plate_image, chars[i], ((x1 - 15), max(40, y1 - 5)),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255), thickness=1,
                            lineType=cv2.LINE_AA)

                cv2.rectangle(plate_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

            cv2.imshow('test', plate_image)
            cv2.waitKey(0)

    # export each char by sequence
    def plate_number_exporter(self, file_name):
        try:
            char_1 = file_name[15]
            char_2 = file_name[16]
            char_3 = file_name[21:].split('-')[0]
            char_4 = file_name[17]
            char_5 = file_name[18]
            char_6 = file_name[19]
            char_7 = file_name[21:].split('-')[1][0]
            char_8 = file_name[21:].split('-')[1][1]
            return [char_1, char_2, char_3, char_4, char_5, char_6, char_7, char_8]

        except (KeyError, IndexError) as e:
            print(e)

    def xyxy_to_xywh(self, xyxy):

        """
        Convert XYXY format (x,y top left and x,y bottom right) to XYWH format (x,y center point and width, height).
        :param xyxy: [X1, Y1, X2, Y2]
        :return: [X, Y, W, H]
        """
        if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
            raise ValueError('xyxy format: [x1, y1, x2, y2]')
        x_temp = (xyxy[0] + xyxy[2]) / 2
        y_temp = (xyxy[1] + xyxy[3]) / 2
        w_temp = abs(xyxy[0] - xyxy[2])
        h_temp = abs(xyxy[1] - xyxy[3])
        return np.array([int(x_temp), int(y_temp), int(w_temp), int(h_temp)])

    def export_labeled_image(self):
        # save image file ine images folder

        # save text file in labels folder
        pass


if __name__ == '__main__':
    Labeling()
