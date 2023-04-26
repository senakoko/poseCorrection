def plot_tracked_points(h5, scale_factor, frame_number):
    """
    plot body points from h5
    :param h5: the h5 file
    :param scale_factor: how to resize the points
    :param frame_number: the frame number
    :return:
    """

    scorer = h5.columns.get_level_values('scorer').unique().item()
    bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()
    individuals = h5.columns.get_level_values('individuals').unique().to_list()

    individual_dict = {}
    for j, ind in enumerate(individuals):

        individual = h5[scorer][ind]
        bodypart_dict = {}

        for bpt in bodyparts:
            bpt_values = individual[bpt]
            bpt_value = bpt_values.iloc[frame_number] * scale_factor
            bodypart_dict[bpt] = bpt_value.astype('int').to_numpy()

        individual_dict[ind] = bodypart_dict

    return individual_dict
