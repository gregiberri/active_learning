import os
import pickle


def make_base_data_part_files(data_path):
    """
    Make pickle files containing the dictionary of data files and the used data in those files for the base data parts:
    `full`, `labelled`, `unlabelled`, `val`

    :param data_path:
    """
    # train
    dict = {'data_batch_1': list(range(10000)),
            'data_batch_2': list(range(10000)),
            'data_batch_3': list(range(10000)),
            'data_batch_4': list(range(10000)),
            'data_batch_5': list(range(10000))}
    write_data_part_to_pickle('train', dict)

    # labelled
    dict = {'data_batch_1': list(range(1000))}
    write_data_part_to_pickle('labelled', dict)

    # unlabelled
    dict = {'data_batch_1': list(range(1000, 10000)),
            'data_batch_2': list(range(10000)),
            'data_batch_3': list(range(10000)),
            'data_batch_4': list(range(10000)),
            'data_batch_5': list(range(10000))}
    write_data_part_to_pickle('unlabelled', dict)

    # val
    dict = {'test_batch': list(range(10000))}
    write_data_part_to_pickle('val', dict)


def write_data_part_to_pickle(data_part_name, data_part_dict):
    with open(data_part_name, 'wb+') as f:
        pickle.dump(data_part_dict, f)


if __name__ == '__main__':
    make_base_data_part_files('cifar-10-batches-py')
