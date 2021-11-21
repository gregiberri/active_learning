import logging
import os
import pickle

import torch
from tensorboardX import SummaryWriter

from ml.metrics.metrics import Metrics
from utils.device import DEVICE


class IOHandler:
    def __init__(self, args, solver):
        self.args = args
        self.phase = args.mode
        self.solver = solver
        self.config = self.solver.config

        self.init_results_dir()
        self.init_metrics()
        self.init_tensorboard()
        self.load_checkpoint()
        self.reset_results()

    def train(self):
        """
        Set the iohandler to use the train metric.
        """
        self.metric = self.train_metric

    def val(self):
        """
        Set the iohandler to use the train metric.
        """
        self.metric = self.val_metric

    def reset_results(self):
        """
        Reset the results to be empty before starting an epoch.
        """
        self.results = {'data_files': [], 'indices': [], 'preds': [], 'labels': []}

    def get_max_metric(self):
        """
        Get the validation goal_metrics of the best model.
        """
        return max(self.val_metric.epoch_results[self.config.metrics.goal_metric]) if self.phase == 'train' else None

    def init_results_dir(self):
        """
        Making results dir.
        """
        logging.info("Making result dir.")
        result_name = os.path.join(self.config.id, self.args.id_tag) if self.args.id_tag else self.config.id
        self.result_dir = os.path.join(self.config.env.result_dir, result_name)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)
        logging.info(f"Results dir is made. Results will be saved at: {self.result_dir}")

    def init_metrics(self):
        """
        Initialize the metrics to follow the performance during training and validation.
        """
        logging.info("Initializing metrics.")
        self.val_metric = Metrics(self.result_dir, 'val', self.config.metrics)
        if self.phase == 'train':
            self.train_metric = Metrics(self.result_dir, 'train', self.config.metrics)

    def init_tensorboard(self):
        """
        Initialize the tensorboard.
        """
        logging.info("Initializing lr policy.")
        self.writer = SummaryWriter(os.path.join(self.result_dir, 'tensorboard'))

    def append_results(self, minibatch, output):
        """
        Save for calculating the full epoch metrics (the dataset is small so it can fit into the memory)
        the calculated metrics are `macro` average to be less sensitive to the class imbalance
        with larger datasets running metrics are suggested
        large datasets usually less affected by class imbalance too.

        :param minibatch: minibatch diractory containing the indices (and during train or val the labels)
        :param output: predictions of the network
        """
        self.results['data_files'].append(minibatch['data_files'])
        self.results['indices'].append(minibatch['indices'])
        self.results['preds'].append(output)
        self.results['labels'].append(minibatch['labels'])

    def calculate_iteration_metrics(self, minibatch, output, loss, pbar, preproc_time, train_time, idx):
        """
        Calculate the metrics during an iteration inside an epoch.
        """
        self.metric.compute_metric(output, minibatch['labels'])
        if self.phase == 'train': self.update_bar_description(pbar, idx, preproc_time, train_time, loss)

        # write to tensorboard
        writer_iteration = self.solver.epoch * len(self.solver.loader) + idx
        # image denormalization is hardcoded here which is not nice but at least works :P
        self.writer.add_image(f'{self.solver.current_mode}/image', minibatch['input_images'][0].cpu() * \
                              torch.Tensor([[[0.2023]], [[0.1994]], [[0.2010]]]) + \
                              torch.Tensor([[[0.4914]], [[0.4822]], [[0.4465]]]),
                              writer_iteration)
        if self.solver.current_mode == 'train':
            self.writer.add_scalar('loss', loss, writer_iteration)

    def update_bar_description(self, pbar, idx, preproc_time, train_time, loss):
        """
        Update the current log bar with the latest result.

        :param pbar: pbar object
        :param idx: iteration number in the epoch
        :param preproc_time: time spent with preprocessing
        :param train_time: time spent with training
        :param loss: loss value
        """
        print_str = f'[{self.solver.current_mode}] epoch {self.solver.epoch}/{self.solver.epochs} ' \
                    + f'iter {idx + 1}/{len(self.solver.loader)}:' \
                    + f'lr:{self.solver.optimizer.param_groups[0]["lr"]:.5f}|' \
                    + f'loss: {loss:.3f}|' \
                    + self.metric.get_snapshot_info() \
                    + f'|t_prep: {preproc_time:.3f}s|' \
                    + f't_train: {train_time:.3f}s'
        pbar.set_description(print_str, refresh=False)

    def compute_epoch_metric(self):
        """
        Calculate the metrics of a whole epoch.
        """
        self.metric.compute_epoch_metric(torch.cat(self.results['preds'], 0),
                                         torch.cat(self.results['labels'], 0),
                                         self.writer,
                                         self.solver.epoch)
        for key, value in self.metric.current_metric.items():
            self.writer.add_scalar(f'{self.solver.current_mode}/{key}', value, self.solver.epoch)

    def save_preds(self):
        # save preds for new label picking
        path = os.path.join(self.result_dir, 'best_epoch_preds')

        preds = {'data_files': [element for sublist in self.results['data_files'] for element in sublist],
                 'indices': torch.cat(self.results['indices'], 0),
                 'preds': torch.cat(self.results['preds'], 0)}

        with open(path, 'wb+') as f:
            pickle.dump(preds, f)
        logging.info(f"Saved preds to file {path}\n")

    def save_best_checkpoint(self):
        """
        Save the model if the last epoch result is the best.
        """
        epoch_results = self.val_metric.epoch_results[self.config.metrics.goal_metric]

        if not max(epoch_results) == epoch_results[-1]:
            return

        path = os.path.join(self.result_dir, 'model_best.pth.tar')

        state_dict = {'epoch': self.solver.epoch,
                      'optimizer': self.solver.optimizer.state_dict(),
                      'lr_policy': self.solver.lr_policy.state_dict(),
                      'train_metric': self.train_metric,
                      'val_metric': self.val_metric,
                      'config': self.solver.config,
                      'model': self.solver.model.state_dict()}

        torch.save(state_dict, path)
        del state_dict
        logging.info(f"Saved checkpoint to file {path}\n")

    def load_checkpoint(self):
        """
        If a saved model in the result folder exists load the model
        and the hyperparameters from a trained model checkpoint.
        """
        path = os.path.join(self.result_dir, 'model_best.pth.tar')
        if not os.path.exists(path):
            assert self.phase == 'train', f'No model file found to load at: {path}'
            return

        logging.info(f"Loading the checkpoint from: {path}")
        continue_state_object = torch.load(path, map_location=torch.device("cpu"))

        # load the things from the checkpoint
        if self.phase == 'train':
            self.solver.optimizer.load_state_dict(continue_state_object['optimizer'])
            self.solver.lr_policy.load_state_dict(continue_state_object['lr_policy'])
            self.train_metric = continue_state_object['train_metric']
        self.val_metric = continue_state_object['val_metric']

        self.solver.epoch = continue_state_object['epoch']
        self.solver.config = continue_state_object['config']
        self.solver.model.load_state_dict(continue_state_object['model'])
        if DEVICE == torch.device('cuda'): self.solver.model.cuda()

        del continue_state_object
