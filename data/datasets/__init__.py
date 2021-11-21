from torch.utils.data import DataLoader

from data.datasets.cifar_10_dataset import Cifar10Dataloader


def get_dataloader(data_config, mode):
    # get the iterator object
    if data_config.name == 'cifar10':
        dataset = Cifar10Dataloader(data_config.params, mode)
    else:
        raise ValueError(f'Wrong dataset name: {data_config.name}')

    # make the torch dataloader object
    loader = DataLoader(dataset,
                        batch_size=int(data_config.params.batch_size),
                        num_workers=data_config.params.workers,
                        drop_last=False,
                        shuffle='train' in mode,
                        pin_memory=data_config.params.load_into_memory)

    return loader
