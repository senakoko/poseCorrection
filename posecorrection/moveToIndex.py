import numpy as np
from pathlib import Path


def move_to_index(h5_path, current_frame_number):
    """
    Move to the next index based on the file loaded
    :param h5_path: path to the video file
    :param current_frame_number: the current frame number is GUI is on
    :return:
    """

    destination_name = Path(h5_path).stem
    destination_name = destination_name[:destination_name.find('CNN')]
    destination_name += 'bad_tracking.npy'
    destination_file = f'{str(Path(h5_path).parent)}/{destination_name}'

    data = np.load(destination_file)

    for i, dt in enumerate(data):
        if dt > current_frame_number:
            percentage = np.round((i / data.shape[0]) * 100)
            return dt, percentage
