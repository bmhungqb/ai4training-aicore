import copy
import os

from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler, ConcatDataset
from torch.utils.data.dataloader import default_collate

from utils.sampler import DistBalancedBatchSampler
from .dataset import GEBDDataset, TAPOSDataset

# train
# True: train data / train
# False: val data / val
def build_dataloader(cfg, args, dataset_splits, train):
    assert len(dataset_splits) >= 1

    def build_dataset(dataset):
        name, split = dataset.rsplit('_', 1)

        def _resolve(p):
            if not p:
                return p
            if os.path.exists(p):
                return p
            alt = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', p))
            return alt if os.path.exists(alt) else p

        ROOT = {
            'GEBD': _resolve(os.getenv('GEBD_ROOT', args.gebd_data_dir)),
            'TAPOS': _resolve(os.getenv('TAPOS_ROOT', args.tapos_data_dir)),
            'SEWING': _resolve(os.getenv('SEWING_ROOT', args.sewing_data_dir)),
            # Chunked variant (tools/chunk_diff_gebd_dataset.py) lives under the
            # same dataset dir (images/{split}/<vid>__chunk<i>/ symlinks).
            'SEWING_CHUNKED': _resolve(os.getenv('SEWING_ROOT', args.sewing_data_dir)),
        }

        datasets = {
            'GEBD': {
                'train': 'train',
                'val': 'val',
            },
            'TAPOS': {
                'train': 'rgb',
                'val': 'rgb'
            },
            'SEWING': {
                'train': 'train',
                'val': 'val',
            },
            'SEWING_CHUNKED': {
                'train': 'train',
                'val': 'val',
            },
        }

        assert name in datasets, f'Dataset {name} not exists!'
        root = os.path.join(ROOT[name], datasets[name][split])

        if name == 'GEBD':
            template = 'img_{:05d}.jpg'
            dataset = GEBDDataset(cfg, root=root,
                                name=name,
                                split=split,
                                template=template,
                                train=train)

        elif name in ('SEWING', 'SEWING_CHUNKED'):
            # Industrial sewing step-segmentation dataset, built by
            # tools/prepare_diff_gebd_dataset.py (+ tools/chunk_diff_gebd_dataset.py
            # for the chunked variant). Uses the same Kinetics-GEBD annotation
            # schema/loader as 'GEBD', but frames are named frame<N>.jpg
            # (1-indexed) instead of img_%05d.jpg.
            template = 'frame{:d}.jpg'
            dataset = GEBDDataset(cfg, root=root,
                                name=name,
                                split=split,
                                template=template,
                                train=train)

        elif name == 'TAPOS':
            template = 'img_{:05d}.jpg'
            dataset = TAPOSDataset(cfg, root=root,
                                name=name,
                                split=split,
                                template=template,
                                train=train)
            
        return dataset

    datasets = []
    for split in dataset_splits:
        datasets.append(build_dataset(split))

    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        annotations = []
        for dataset in datasets:
            annotations.extend(copy.deepcopy(dataset.annotations))
        dataset = ConcatDataset(datasets)
        dataset.annotations = annotations

    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=train)
    else:
        sampler = RandomSampler(dataset) if train else SequentialSampler(dataset)
        

    # collate_fn = (lambda x: x) if cfg.INPUT.END_TO_END else default_collate
    collate_fn = default_collate
    loader = DataLoader(dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                        sampler=sampler,
                        drop_last=False,
                        collate_fn=collate_fn,
                        num_workers=cfg.SOLVER.NUM_WORKERS)
    return loader
