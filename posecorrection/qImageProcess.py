from PyQt5.QtGui import QImage, QPixmap


def qt_image_process(image):
    image = QImage(image.data, image.shape[1], image.shape[0],
                   QImage.Format_RGB888).rgbSwapped()
    return QPixmap.fromImage(image)
