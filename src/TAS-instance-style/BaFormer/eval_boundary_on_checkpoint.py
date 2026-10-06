#!/usr/bin/env python3
"""Evaluate a trained checkpoint's boundary Precision/Recall/F1 at the
+/-0.1s / +/-0.25s / +/-0.5s time tolerances (see main.py::validate()),
without retraining.

Usage:
    python eval_boundary_on_checkpoint.py \
        --config configs/dataset_clean_dinov2.yaml \
        --checkpoint experiments/tas_instance/bk_fde_tde/dataset_clean_dinov2/1/checkpoint_best.pth
"""
import argparse
import pathlib

import torch

from action_segmentation import (
    create_dataloader,
    create_model,
    get_default_config,
    update_config,
)
from action_segmentation.utils import DummyWriter, create_logger, set_seed, setup_cudnn

from main import validate, Record_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--checkpoint', required=True, type=str)
    parser.add_argument('options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    config = get_default_config()
    config.merge_from_file(args.config, allow_unsafe=True)
    if args.options:
        config.merge_from_list(args.options)
    if not torch.cuda.is_available():
        config.device = 'cpu'
        config.train.dataloader.pin_memory = False
    config = update_config(config)
    config.freeze()

    set_seed(config)
    setup_cudnn(config)

    # Unique log filename per (config, checkpoint) pair, so results from
    # different backbones don't get appended into the same shared log file.
    tag = f"{pathlib.Path(args.config).stem}__{pathlib.Path(args.checkpoint).parent.parent.name}"
    log_filename = f'eval_boundary_log_{tag}.txt'
    logger = create_logger(name=__name__, distributed_rank=0,
                            output_dir=pathlib.Path('.'), filename=log_filename)

    val_loader = create_dataloader(config, is_train=False)
    model = create_model(config)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(torch.device(config.device))

    val_record = Record_dict()
    _, val_metrics = validate(epoch=1, config=config, model=model,
                               val_loader=val_loader, val_record=val_record,
                               logger=logger, tensorboard_writer=DummyWriter())

    print(f"\n=== Boundary metrics ({args.checkpoint}) === [log: {log_filename}]")
    for tol_s in (0.1, 0.25, 0.5):
        f1 = val_metrics['bd_f1_by_tol'][tol_s]
        prec = val_metrics['bd_prec_by_tol'][tol_s]
        rec = val_metrics['bd_rec_by_tol'][tol_s]
        print(f"  Tol +/-{tol_s}s: F1={f1:.2f}%  Prec={prec:.2f}%  Rec={rec:.2f}%")
    print(f"  Acc={val_metrics['acc']:.2f}%  Edit={val_metrics['edit']:.2f}  "
          f"F1@10/25/50={val_metrics['f1_10']:.2f}/{val_metrics['f1_25']:.2f}/{val_metrics['f1_50']:.2f}")


if __name__ == '__main__':
    main()
