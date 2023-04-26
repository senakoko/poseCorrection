import pandas as pd
import numpy as np
from pathlib import Path


def cal_animal_area(h5_data=None, scorer="Stacked_Autoencoder", individual='ind1'):
    """
    Calculate the area of the animal assuming it has an ellipsoid shape

    Parameters
    ----------
    h5_data: h5 data or path to h5 data
    scorer: the annotator/scorer of h5 file
    individual: which individual to calculate its area
    """
    if not isinstance(h5_data, pd.DataFrame):
        h5 = pd.read_hdf(h5_data)
    else:
        h5 = h5_data

    nose = h5[scorer][individual].loc[:, 'Nose']
    left_mid = h5[scorer][individual].loc[:, 'leftMidWaist']
    bodypart = h5[scorer][individual].loc[:, ['leftMidWaist', 'rightMidWaist']]
    center = bodypart.leftMidWaist.add(bodypart.rightMidWaist).divide(2)
    a_len = nose.sub(center)
    b_len = center.sub(left_mid)
    a_dist = np.linalg.norm(a_len, axis=1)
    b_dist = np.linalg.norm(b_len, axis=1)
    area = np.pi * a_dist * b_dist
    area_df = pd.DataFrame({'area': area})

    return area_df


def cal_bodyparts_dist(h5_data=None, body_part1='Nose', body_part2='betweenEars',
                       scorer="Stacked_Autoencoder", individual='ind1'):
    """
    Calculate the distance between body parts

    Parameters
    ----------
    h5_data: h5 data or path to h5 data
    body_part1: the first body point to use
    body_part2: the second boyd point to use
    scorer: the annotator/scorer of h5 file
    individual: which individual to calculate its area
    """
    if not isinstance(h5_data, pd.DataFrame):
        h5 = pd.read_hdf(h5_data)
    else:
        h5 = h5_data

    bpt1 = h5[scorer][individual].loc[:, body_part1]
    bpt2 = h5[scorer][individual].loc[:, body_part2]
    bpts_diff = bpt1.sub(bpt2)
    bpts_dist = np.linalg.norm(bpts_diff, axis=1)
    bpts_df = pd.DataFrame({'bpts_dist': bpts_dist})

    return bpts_df


# noinspection PyTypeChecker
def find_bad_tracking(file):
    mad_multiplier = 2.75

    h5 = pd.read_hdf(file)
    scorer = h5.columns.get_level_values('scorer').unique().item()
    individuals = h5.columns.get_level_values('individuals').unique().to_list()

    body_parts_list = [['Nose', 'betweenEars'], ['tailStart', 'midHip']]

    bad_tracking_list = []
    for ind in individuals:

        area = cal_animal_area(h5, scorer=scorer, individual=ind)
        area_thresh = (area.median() * 0.5).item()
        bad_area1 = np.where(area.values < area.median().item() - area_thresh)[0]
        bad_area2 = np.where(area.values > area.median().item() + area_thresh)[0]
        bad_area = np.concatenate((bad_area1, bad_area2))
        bad_area = np.unique(bad_area)

        bad_tracking_list.extend(bad_area.tolist())
        for bpts in body_parts_list:
            bpts_dist = cal_bodyparts_dist(h5, body_part1=bpts[0], body_part2=bpts[1],
                                           scorer=scorer, individual=ind)
            mad = (bpts_dist - bpts_dist.mean()).abs().median()
            bpts_dist_thresh = (mad_multiplier * mad).item()
            bad_bpts_dist1 = np.where(bpts_dist.values < bpts_dist.median().item() - bpts_dist_thresh)[0]
            bad_bpts_dist2 = np.where(bpts_dist.values > bpts_dist.median().item() + bpts_dist_thresh)[0]
            bad_bpts_dist = np.concatenate((bad_bpts_dist1, bad_bpts_dist2))
            bad_bpts_dist = np.unique(bad_bpts_dist)

            bad_tracking_list.extend(bad_bpts_dist.tolist())
    bad_tracking_list = np.array(bad_tracking_list)

    destination_name = Path(file).stem
    destination_name = destination_name[:destination_name.find('CNN')]
    destination_name += 'bad_tracking.npy'
    destination_file = f'{str(Path(file).parent)}/{destination_name}'
    np.save(destination_file, bad_tracking_list)

    return
