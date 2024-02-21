from ultralytics import YOLO


def test():
    # model_1 = YOLO('model/best.pt')
    model_1 = YOLO('alpr/assets/bestV2m.pt')

    predict_1 = model_1('/home/kokhaie/PycharmProjects/LinMl/media/plates/181ed8ca43924293ae9a5bedd704aa39.jpg')

    char_display = []
    bbox = predict_1[0].boxes.xyxy
    cls = predict_1[0].boxes.cls
    # make a dict and sort it from left to right to show the correct characters of plate
    keys = cls.cpu().numpy().astype(int)
    values = bbox[:, 0].cpu().numpy().astype(int)
    dictionary = list(zip(keys, values))
    sorted_list = sorted(dictionary, key=lambda x: x[1])
    print(sorted_list)

    # convert all characters to a string
    for char in sorted_list:
        char_class = char[0]
        # char_display.append(plate_output[0].names[char_class])
        char_display.append(model_1.names[char_class])
    char_result = (''.join(char_display))

    print(char_result)


if __name__ == '__main__':
    test()
