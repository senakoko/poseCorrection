from easydict import EasyDict as eDict


def set_run_parameters(parameters=None):

    if isinstance(parameters, dict):
        parameters = eDict(parameters)
    else:
        parameters = eDict()

    font_small = 15

    font_mid = 20

    line_edit_width = 100

    dot_size = 10

    scale_factor = 0.5 # the scale factor to resize the image. O.5 is recommended

    if 'font_small' not in parameters.keys():
        parameters.font_small = font_small

    if 'font_mid' not in parameters.keys():
        parameters.font_mid = font_mid

    if 'line_edit_width' not in parameters.keys():
        parameters.line_edit_width = line_edit_width

    if 'scale_factor' not in parameters.keys():
        parameters.scale_factor = scale_factor

    if 'dot_size' not in parameters.keys():
        parameters.dot_size = dot_size

    return parameters
