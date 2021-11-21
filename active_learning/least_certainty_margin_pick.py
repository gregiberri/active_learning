import os
import pickle
import sys

import numpy as np
import torch
import torch.nn.functional as F

# set the working dir to the parent working dir
current_working_dir = os.getcwd()
print(f'Previous working dir: {current_working_dir}')
new_working_dir = '/'.join(current_working_dir.split('/')[:-1])
os.chdir(new_working_dir)
sys.path.insert(0, new_working_dir)
print(f'Current working dir: {os.getcwd()}')


def uncertainty_sampling(pred_path, sample_number):
    with open(pred_path, 'rb') as fo:
        pred_dict = pickle.load(fo, encoding='bytes')
    pred_softmaxes = F.softmax(pred_dict['preds'], -1)
    top2_pred_softmaxes, _ = torch.topk(pred_softmaxes, k=2, dim=-1)
    pred_softmax_diffs = torch.abs(top2_pred_softmaxes[..., 0] - top2_pred_softmaxes[..., 1])
    _, sample_indices = torch.topk(-pred_softmax_diffs.cpu(), k=sample_number, dim=0)

    sample_filenames = np.array(pred_dict['data_files'])[sample_indices]
    sample_indices = np.array(pred_dict['indices'].cpu())[sample_indices]

    data_part = {key: [] for key in sorted(set(sample_filenames))}
    for filename, indice in zip(sample_filenames, sample_indices):
        data_part[filename].append(indice)

    data_part_path = os.path.join('data', 'data_parts', 'least_certainty_margin')
    print(f'saving file to: {data_part_path}')
    with open(data_part_path, 'wb+') as fo:
        pickle.dump(data_part, fo)


if __name__ == '__main__':
    pred_path = "results/labelled_hpo/hpo_outputs/run_experiment_2021-11-20_19-56-42/run_experiment_{'result_dir': 'results', 'random_seed': 0, 'epochs': 50, 'save_preds': False}_8b7edf0d_42_base_config=labelled,nam_2021-11-20_20-20-31/best_epoch_preds"
    uncertainty_sampling(pred_path, 5000)
