import pandas as pd


def propagate_frame(h5, frame_number, h5_filename, forward_backward='forward', steps=1, animal_ident='both'):
    """
    Propagate rightly tracked body points forward or backward. Hence, update the next or previous N number of frames
    from the current one. The "N" is defined by the steps
    :param h5: the H5 data (not the filepath)
    :param frame_number: the frame number for the current image
    :param h5_filename: the filepath for the H5 file
    :param forward_backward: propagate forward or backward
    :param steps: the number of frames to update from the current one
    :param animal_ident: the animal identity or identities to use to propagate frames
    :return:
    """

    with pd.HDFStore(h5_filename, 'r') as df:
        animal_key = df.keys()[0]

    scorer = h5.columns.get_level_values('scorer').unique().item()
    bodyparts = h5.columns.get_level_values('bodyparts').unique().to_list()
    individuals = h5.columns.get_level_values('individuals').unique().to_list()

    data_df = pd.DataFrame()
    for i in range(len(individuals)):
        data = h5[scorer][individuals[i]].values
        if animal_ident == 'both':
            if forward_backward == 'backward':
                data[frame_number - steps: frame_number, :] = data[frame_number, :]
            else:
                data[frame_number + 1: frame_number + steps, :] = data[frame_number, :]
        else:
            v = int(animal_ident[-1:]) - 1
            if i == v:
                if forward_backward == 'backward':
                    data[frame_number - steps: frame_number, :] = data[frame_number, :]
                else:
                    data[frame_number + 1: frame_number + steps, :] = data[frame_number, :]

        df = pd.DataFrame(data)
        data_df = pd.concat((data_df, df), axis=1, ignore_index=True)

    col = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, ['x', 'y']],
                                     names=['scorer', 'individuals', 'bodyparts', 'coords'])
    data_ind = h5.index
    dataframe = pd.DataFrame(data_df.values, index=data_ind, columns=col)
    dataframe.to_hdf(h5_filename, animal_key)
