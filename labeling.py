import math
import numpy as np
import cv2
import os
from ultralytics import YOLO


class Labeling:
    def __init__(self):
        self.car_plate_model = YOLO('alpr/assets/best.pt')
        self.char_model = YOLO('alpr/assets/bounding_box_model.pt')
        self.characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'be', 'dal', 'ein',
                           'he', 'jim', 'lam', 'mim', 'non', 'ghaf', 'sad', 'sin', 'ta', 'te',
                           'vav', 'ye', 'zhe', 'alef', 'se', 'pe', 'ze', 'shin']

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
            label_text = ''
            for i, value in enumerate(values):
                x1, y1, x2, y2 = bbox[results[0].boxes.xyxy[:, 0].tolist().index(value)].xyxy[0]

                xywh_format = self.xyxy_to_xywh(
                    [x1.cpu().numpy().astype(float), y1.cpu().numpy().astype(float), x2.cpu().numpy().astype(float),
                     y2.cpu().numpy().astype(float)])

                # go to next line when it is not at first line
                if i != 0:
                    label_text += '\n'

                # convert to yolo label format (label x y w h)
                label_text += '{} {} {} {} {}'.format(self.characters.index(chars[i]),
                                                      xywh_format[0],
                                                      xywh_format[1],
                                                      xywh_format[2],
                                                      xywh_format[3])

                x1, y1, x2, y2 = math.ceil(x1), math.ceil(y1), math.ceil(x2), math.ceil(y2)

                cv2.putText(plate_image, chars[i], ((x1 - 15), max(40, y1 - 5)),
                            fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=0.5, color=(10, 50, 255), thickness=1,
                            lineType=cv2.LINE_AA)

                cv2.rectangle(plate_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
            self.export_label(image_name, label_text)
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

    # convert xyxy format to xywh (yolo train format)
    # x,y is center of the object
    # x1,y1,y2,x2 is corner of bounding boxes

    def xyxy_to_xywh(self, xyxy):

        if np.array(xyxy).ndim > 1 or len(xyxy) > 4:
            raise ValueError('xyxy format: [x1, y1, x2, y2]')
        x_temp = (xyxy[0] + xyxy[2]) / 2
        y_temp = (xyxy[1] + xyxy[3]) / 2
        w_temp = abs(xyxy[0] - xyxy[2])
        h_temp = abs(xyxy[1] - xyxy[3])
        return np.array([x_temp, y_temp, w_temp, h_temp])

    def export_image(self, plate_image):
        image = open('file.jpg', 'w')
        image.write('what ever')
        image.close()

    def export_label(self, file_name, text):

        # remove .jpg from file name
        if '.jpg' == file_name[-4:]:
            file_name = file_name[:-4]

        image = open('{0}.txt'.format(file_name), 'w')
        image.write(text)
        image.close()
    # save image file ine images folder

    # save text file in labels folder


if __name__ == '__main__':
    Labeling()
