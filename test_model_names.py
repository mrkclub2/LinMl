from django.core.files.storage import FileSystemStorage
from ultralytics import YOLO


def test():
    object_model = YOLO('alpr/assets/best.pt')

    FileSystemStorage(location="/home/kokhaie/Pictures").save(source.name, source)
    source = '/home/kokhaie/Pictures/{0}'.format(source.name)
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
                plate_output = self.character_model(plate_img)


if __name__ == '__main__':
    test()
