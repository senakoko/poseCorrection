from pathlib import Path
import yaml


def save_last_frame_number(frame_number, video_file):
    """
    Saves the frame number for the video when the GUI is cancelled. It allows the GUI to continue from where it stopped
    in the previous session
    :param frame_number: the last frame number when the GUI is closed
    :param video_file: the name of the video file
    :return:
    """
    destination_file = Path('.') / 'last_video_frame.yaml'

    video_frame = {f"{video_file}": frame_number}

    if not destination_file.exists():
        with open(destination_file, 'w') as fw:
            yaml.dump(video_frame, fw, default_flow_style=False, sort_keys=False)
    else:
        with open(destination_file, 'r') as fr:
            data = yaml.load(fr, Loader=yaml.FullLoader)
            data[f'{video_file}'] = frame_number
        with open(destination_file, 'w') as fw:
            yaml.dump(data, fw, default_flow_style=False, sort_keys=False)
