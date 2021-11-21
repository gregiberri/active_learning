import os
import pickle


def get_datapart(datapart_name, data_path):
    """
    Get the datapart according to the datapart name.
    First loads the datafile and index pairs to read from the dataparts folder from the corresponding pickle file,
    then loads the datapoints, data_part names and labels.

    :param datapart_name: the name of the data_part
    :param data_path: path for the dataset
    :return: [data, data_files, labels]
    """
    data, data_parts, labels = [], [], []
    datapart_file = os.path.join('data/data_parts', datapart_name)
    assert os.path.exists(datapart_file), f'The required data_part file {datapart_file} does not exists'

    # get the data_part dict containing the data files as keys and the required data indices inside
    with open(datapart_file, 'rb') as f:
        data_part = pickle.load(f, encoding='bytes')

    # read from all the required data file all the required data points
    data, indices, data_files, labels = [], [], [], []
    for data_file, data_indices in data_part.items():
        with open(os.path.join(data_path, data_file), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        # load all the required datapoint from the file
        for i in data_indices:
            data.append(dict[b'data'][i])
            indices.append(i)
            data_files.append(data_file)
            labels.append(dict[b'labels'][i])

    return data, indices, data_files, labels