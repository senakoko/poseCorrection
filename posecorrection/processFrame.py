import cv2


def process_frame(image, screen_height, screen_width):
    """
    Resizes the image before it is plotted in the GUI
    :param image: the image
    :param screen_height: the height of the computer screen
    :param screen_width: the width of the computer screen
    :return:
    """
    height, width = image.shape[:2]
    if screen_height < height or screen_width < width:
        scale_factor = 0.5
    else:
        scale_factor = 1
    height = int(height * scale_factor)
    width = int(width * scale_factor)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image
