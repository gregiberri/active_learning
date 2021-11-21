import os
import pickle
import random
import sys

import numpy as np

# set the working dir to the parent working dir
current_working_dir = os.getcwd()
print(f'Previous working dir: {current_working_dir}')
new_working_dir = '/'.join(current_working_dir.split('/')[:-1])
os.chdir(new_working_dir)
sys.path.insert(0, new_working_dir)
print(f'Current working dir: {os.getcwd()}')

random.seed(0)
np.random.seed(0)


def random_sampling(pred_path, sample_number):
    with open(pred_path, 'rb') as fo:
        pred_dict = pickle.load(fo, encoding='bytes')

    filenames = np.array(pred_dict['data_files'])
    indices = np.array(pred_dict['indices'].cpu())

    sample_indices = np.random.choice(filenames.shape[0], sample_number, replace=False)

    sample_filenames = filenames[sample_indices]
    sample_indices = indices[sample_indices]

    data_part = {key: [] for key in sorted(set(sample_filenames))}
    for filename, indice in zip(sample_filenames, sample_indices):
        data_part[filename].append(indice)

    data_part_path = os.path.join('data', 'data_parts', 'random')
    print(f'saving file to: {data_part_path}')
    with open(data_part_path, 'wb+') as fo:
        pickle.dump(data_part, fo)


if __name__ == '__main__':
    pred_path = "results/labelled_hpo/hpo_outputs/run_experiment_2021-11-20_19-56-42/run_experiment_{'result_dir': 'results', 'random_seed': 0, 'epochs': 50, 'save_preds': False}_8b7edf0d_42_base_config=labelled,nam_2021-11-20_20-20-31/best_epoch_preds"
    random_sampling(pred_path, 5000)
