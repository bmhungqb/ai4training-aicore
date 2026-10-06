import math
import torch

from .combined_scheduler import CombinedScheduler
from .components import (
    ConstantScheduler,
    CosineScheduler,
    ExponentialScheduler,
    LinearScheduler,
)
from .multistep_scheduler import MultistepScheduler
from .sgdr import SGDRScheduler


def _create_warmup(config, warmup_steps):
    warmup_type = config.scheduler.warmup.type
    if warmup_type == 'none' or warmup_steps == 0:
        return None

    warmup_start_factor = config.scheduler.warmup.start_factor

    if warmup_type == 'linear':
        lr_end = 1
        lr_start = warmup_start_factor
        scheduler = LinearScheduler(warmup_steps, lr_start, lr_end)
    elif warmup_type == 'exponential':
        scheduler = ExponentialScheduler(warmup_steps, config.train.base_lr,
                                         config.scheduler.warmup.exponent,
                                         warmup_start_factor)
    else:
        raise ValueError()

    return scheduler


def _create_main_scheduler(config, main_steps):
    scheduler_type = config.scheduler.type

    if scheduler_type == 'constant':
        scheduler = ConstantScheduler(main_steps, 1)
    elif scheduler_type == 'multistep':
        lr_decay = config.scheduler.lr_decay
        scheduler = MultistepScheduler(main_steps, 1, lr_decay,
                                       config.scheduler.milestones)
    elif scheduler_type == 'linear':
        lr_start = 1
        lr_end = config.scheduler.lr_min_factor
        scheduler = LinearScheduler(main_steps, lr_start, lr_end)
    elif scheduler_type == 'cosine':
        scheduler = CosineScheduler(main_steps, 1,
                                    config.scheduler.lr_min_factor)
    elif scheduler_type == 'sgdr':
        scheduler = SGDRScheduler(main_steps, 1, config.scheduler.T0,
                                  config.scheduler.T_mul,
                                  config.scheduler.lr_min_factor)
    else:
        raise ValueError()

    return scheduler


def create_scheduler(config, optimizer, steps_per_epoch):
    sched_config = config.scheduler
    sched_type = getattr(sched_config, 'type', 'multistep')
    if sched_type == 'cosine':
        # Pillar 6 (proposal_imprv_baformer14): Cosine Annealing Scheduler with Warmup
        # Smoothly decays learning rate from base_lr down to eta_min without sharp cliff drops,
        # perfectly matching the peak generalization epoch window (ep 50-70).
        eta_min = getattr(sched_config, 'eta_min', 1e-6)
        warmup_epochs = getattr(sched_config, 'warmup_epochs', 10)
        total_epochs = getattr(sched_config, 'epochs', 300)
        base_lr = float(config.train.base_lr)
        if warmup_epochs > 0:
            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                return max(eta_min / base_lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=eta_min)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched_config.milestones, gamma=sched_config.lr_decay)
    return scheduler

def create_set_scheduler(config, optimizer, steps_per_epoch):
    #------------ simple
    sched_config = config.scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=sched_config.milestones_set, gamma=sched_config.lr_decay_set)

    return scheduler