import pandas as pd
import numpy as np


def update_h5file(new_points, h5, frame_number, h5_filename, scale_factor):
    """
    Update the H5 file with the adjusted relabeled body points
    :param new_points: the adjusted newly tracked body points
    :param h5: the H5 data (not the filepath)
    :param frame_number: the frame number for the image that was relabeled
    :param h5_filename: the filepath for the H5 file
    :param scale_factor: the scale_factor to adjust the points
    :return: Saves the newly adjusted tracked points (overwrites the current H5 file)
    """
    with pd.HDFStore(h5_filename, 'r') as df:
        animal_key = df.keys()[0]

    scorer = h5.columns.get_level_values('scorer').unique().item()
    bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()
    individuals = h5.columns.get_level_values('individuals').unique().to_list()

    for individual in individuals:
        for bpt in bodyparts:
            new_pts_value = np.array(new_points[individual][bpt]) * (1/scale_factor)
            # print(new_pts_value)
            h5[scorer][individual].loc[frame_number, bpt] = new_pts_value

    h5.to_hdf(h5_filename, animal_key)
