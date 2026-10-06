import os
import torch

from .defaults import get_default_config
from .config_node import configurable


def update_config(config):
    if config.dataset.name in ['CIFAR10', 'CIFAR100']:
        dataset_dir = f'~/.torch/datasets/{config.dataset.name}'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 32
        config.dataset.n_channels = 3
        config.dataset.n_classes = int(config.dataset.name[5:])
    elif config.dataset.name in ['MNIST', 'FashionMNIST', 'KMNIST']:
        dataset_dir = '~/.torch/datasets'
        config.dataset.dataset_dir = dataset_dir
        config.dataset.image_size = 28
        config.dataset.n_channels = 1
        config.dataset.n_classes = 10
    elif config.dataset.name == 'tas_instance':
        # NOTE: the dataset-specific if/elif chain in defaults.py only runs
        # once at module IMPORT time, using whatever `config.dataset.name`
        # default was hardcoded there -- it does NOT re-trigger when
        # `dataset.name` is overridden afterwards via --config/CLI (as the
        # README instructs for gtea/50salads/breakfast too, where this is a
        # pre-existing latent issue). Re-apply the tas_instance values here,
        # since update_config() runs after merge_from_file/merge_from_list.
        mapping_file = os.path.join(os.path.expanduser(config.dataset.dataset_dir), 'mapping.txt')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                config.dataset.n_classes = len([l for l in f.readlines() if l.strip()])
        elif not hasattr(config.dataset, 'n_classes') or not config.dataset.n_classes:
            config.dataset.n_classes = 5
        config.dataset.sample_rate = 1
        if not hasattr(config.dataset, 'noise_weight') or config.dataset.noise_weight is None:
            config.dataset.noise_weight = 0.08
        config.dataset.num_query = 150
        config.dataset.guassian_sigma = None
        if not hasattr(config.dataset, 'boundary_sigma') or config.dataset.boundary_sigma is None:
            config.dataset.boundary_sigma = 1.5
        if not hasattr(config.dataset, 'pos_weight') or config.dataset.pos_weight is None or config.dataset.pos_weight == 20:
            config.dataset.pos_weight = 6.0
        if not hasattr(config.dataset, 'threshold') or config.dataset.threshold is None:
            config.dataset.threshold = 0.15
        if not hasattr(config.dataset, 'peak_prominence') or config.dataset.peak_prominence is None:
            config.dataset.peak_prominence = 0.035
        if not hasattr(config.dataset, 'min_distance') or config.dataset.min_distance is None:
            config.dataset.min_distance = 7
        if not hasattr(config.dataset, 'min_duration') or config.dataset.min_duration is None:
            config.dataset.min_duration = 7
        if not hasattr(config.dataset, 'theta_t') or config.dataset.theta_t is None:
            config.dataset.theta_t = 7
        if not hasattr(config.model, 'contra_weight') or config.model.contra_weight is None:
            config.model.contra_weight = 0.20
        if not hasattr(config.model, 'repulse_weight') or config.model.repulse_weight is None:
            config.model.repulse_weight = 0.15
        if not hasattr(config.model, 'note') or not config.model.note:
            config.model.note = 'final_exp14'
        if not hasattr(config.train, 'base_lr') or not config.train.base_lr:
            config.train.base_lr = 0.0005
        if not hasattr(config.scheduler, 'milestones') or not config.scheduler.milestones:
            config.scheduler.milestones = [100, 180, 260]
        if not hasattr(config.dataset, 'feat_folders') or not config.dataset.feat_folders:
            config.dataset.feat_folders = ['features_dinov2', 'features_videomae']
        if not hasattr(config.dataset, 'n_channels') or config.dataset.n_channels is None:
            config.dataset.n_channels = getattr(config.model.action_seg.frame_decoder, 'input_dim', 1536)

    if not torch.cuda.is_available():
        config.device = 'cpu'

    return config
