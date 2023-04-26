import cv2


def save_frame(frame, index, indexlength, output_path):
    img_name = (str(output_path) + "/img" + str(index).zfill(indexlength) + ".png")
    cv2.imwrite(img_name, frame)
