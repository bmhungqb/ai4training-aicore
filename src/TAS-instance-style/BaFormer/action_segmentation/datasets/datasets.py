import os
import os.path
import pathlib
import json
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset


import yacs.config
from tqdm import tqdm, trange


def make_dataset(
    directory: str,
    config: yacs.config.CfgNode,
    state: str,
    class_to_idx: Dict[str, int],
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
) -> List[Tuple[str, int]]:
    instances = []
    directory = os.path.expanduser(directory)

    if state=='train':
        split_file = 'train.split' + str(config.split) + ".bundle"
        file = os.path.join(directory, config.name, "splits", split_file)
    elif state=='val':
        split_file = 'test.split' + str(config.split) + ".bundle"
        file = os.path.join(directory, config.name, "splits", split_file)

    f = open(file, 'r')
    lines = f.readlines()
    for line in lines:
        gtfname = line.split('\n')[0]
        fname = gtfname.split('.')[0] + '.npy'
        startname = gtfname.split('.')[0] + '.txt'
        fpath = os.path.join(directory, config.name, 'features', fname)
        gtpath = os.path.join(directory, config.name, 'groundTruth', gtfname)
        # startpath = os.path.join(directory, config.name, 'actStart', startname)
        # item = fpath, gtpath, startpath, str(gtfname.split('.')[0])
        item = fpath, gtpath, str(gtfname.split('.')[0])
        instances.append(item)
    f.close()
    return instances


class DatasetFolder(VisionDataset):
    """A data loader where the samples are arranged in this way: ::

    Args:
        root (string): Root directory path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            config: yacs.config.CfgNode,
            state: str,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        # self.with_noise = config.with_noise
        self.roll = config.roll
        action_to_idx = self._find_classes(os.path.join(self.root, config.name))
        samples = make_dataset(self.root, config, state, action_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.action_to_idx = action_to_idx
        self.samples = samples
        # if state == 'train':
        #     self.samples = samples[ :1]
        # elif state == 'val':
        #     self.samples = samples[ :1]
        self.sample_rate = config.sample_rate
        self.state = state

    def _find_classes(self, dir: str) ->  Dict[str, int]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        f = open(os.path.join(dir, 'mapping.txt'), 'r')
        lines = f.readlines()
        action_to_idx = {}
        for line in lines:
            action_to_idx[line.split()[1]] = int(line.split()[0])
        f.close()
        return action_to_idx

    def loader(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        content = [line.split('\n')[0] for line in lines]
        len_frame = len(content)
        act_sequence = np.zeros(len_frame).astype(np.int32)
        for i in range(len_frame):
            act_sequence[i] = self.action_to_idx[content[i]]
        return act_sequence

    def load_start(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        content = [int(line.split('\n')[0]) for line in lines]
        return content

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.state=='train':
            # act_path, cls_path, start_path, fname = self.samples[index]
            act_path, cls_path,  fname = self.samples[index]

            sample = np.load(act_path)[:, ::self.sample_rate]
            target = self.loader(cls_path)[::self.sample_rate]
            # start = self.load_start(start_path)


            ## generate noise
            noise = np.random.randn(*sample.shape)# add noise herer
            # if self.with_noise is False:
            #     noise = 0

            ## roll for the boundary
            if self.roll is not None:
                roll_dist = int(torch.randint(low=-self.roll, high=self.roll+1, size=(1,)))
                target_roll = np.roll(target, shift=roll_dist, axis=0)
                if roll_dist > 0:
                    target = np.pad(target_roll[roll_dist:], (roll_dist, 0), 'edge')
                elif roll_dist <0:
                    target = np.pad(target_roll[:roll_dist], (0, -roll_dist), 'edge')
            #     a = 1
                ## roll over the act, so need known the boundary, pre-processing
                # change target and sample after roll

            return sample, target, fname, noise


        elif self.state=='val':
            # act_path, cls_path, start_path, fname = self.samples[index]
            act_path, cls_path, fname = self.samples[index]

            sample = np.load(act_path)[:, ::self.sample_rate]
            target = self.loader(cls_path)

            # roll_dist = torch.randint(low=-5, high=6, size=(1,))
            # sample_roll = np.roll(sample, shift=int(roll_dist), axis=1) # 1-->, -1 <--
            # if roll_dist > 0:
            #     sample = np.pad(sample_roll[:, roll_dist: ], ((0, 0), (roll_dist, 0)), 'edge')
            # elif roll_dist <0:
            #     sample = np.pad(sample_roll[:, :roll_dist], ((0, 0), (0, -roll_dist)), 'edge')

        # if self.transform is not None:
        #     sample = self.transform(sample)
        # if self.target_transform is not None:
        #     target = self.target_transform(target)

            return sample, target, fname

    def __len__(self) -> int:
        return len(self.samples)


def create_dataset(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'gtea', '50salads', 'breakfast'
    ]:
        if is_train:
            dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
            # train_transform = create_transform(config, is_train=True)
            # val_transform = create_transform(config, is_train=False)
            train_transform = None
            val_transform = None
            train_dataset = DatasetFolder(dataset_dir,config.dataset, 'train', transform=train_transform)
            val_dataset = DatasetFolder(dataset_dir,config.dataset, 'val', transform=val_transform)
            return train_dataset, val_dataset
        else:
            dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
            # val_transform = create_transform(config, is_train=False)
            val_transform = None
            val_dataset = DatasetFolder(dataset_dir,config.dataset, 'val', transform=val_transform)
            return val_dataset

    else:
        raise ValueError()

# ============================================= all ===========================================
class DatasetFolderAll(VisionDataset):
    """A data loader where the samples are arranged in this way: ::

    Args:
        root (string): Root directory path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
            self,
            root: str,
            config: yacs.config.CfgNode,
            state: str,
            extensions: Optional[Tuple[str, ...]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(DatasetFolderAll, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        action_to_idx = self._find_classes(os.path.join(self.root, config.name))
        samples = make_dataset(self.root, config, state, action_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.action_to_idx = action_to_idx
        self.sample_rate = config.sample_rate
        self.state = state

        self.features = []
        self.gt = []
        self.fname = []

        # for i in tqdm(range(100)):
        for i in tqdm(range(len(samples))):
            act_path = samples[i][0]
            cls_path = samples[i][1]
            fname = samples[i][2]

            sample = np.load(act_path)[:, ::config.sample_rate]
            if state == "train":
                target = self.loader(cls_path)[::config.sample_rate]
            elif state == "val":
                target = self.loader(cls_path)
            self.features.append(sample)
            self.gt.append(target)
            self.fname.append(fname)

    def _find_classes(self, dir: str) ->  Dict[str, int]:
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """

        f = open(os.path.join(dir, 'mapping.txt'), 'r')
        lines = f.readlines()
        action_to_idx = {}
        for line in lines:
            action_to_idx[line.split()[1]] = int(line.split()[0])
        f.close()
        return action_to_idx

    def loader(self, path):
        f = open(path, 'r')
        lines = f.readlines()
        content = [line.split('\n')[0] for line in lines]
        len_frame = len(content)
        act_sequence = np.zeros(len_frame).astype(np.int32)
        for i in range(len_frame):
            act_sequence[i] = self.action_to_idx[content[i]]
        return act_sequence


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample= self.features[index]
        target= self.gt[index]
        fname = self.fname[index]
        return sample, target, fname

    def __len__(self) -> int:
        return len(self.features)

def create_dataset_all(config: yacs.config.CfgNode,
                   is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    if config.dataset.name in [
            'gtea', '50salads', 'breakfast'
    ]:
        dataset_dir = pathlib.Path(config.dataset.dataset_dir).expanduser()
        train_transform = None
        val_transform = None
        train_dataset = DatasetFolderAll(dataset_dir,config.dataset, 'train', transform=train_transform)
        val_dataset = DatasetFolderAll(dataset_dir,config.dataset, 'val', transform=train_transform)
        return train_dataset, val_dataset
    else:
        raise ValueError()


# ============================================= instance-level (tas_instance) ===========================================
class InstanceDatasetFolder(VisionDataset):
    """Loader for the instance-level TAS dataset described in
    `src/TAS-instance-style/problem_definition.md` (section 17), produced by
    `tools/prepare_tas_instance_dataset.py` into `dataset_tas_instance/`:

        <root>/
          mapping.txt
          features/<video_id>.npy        (C, T) float32
          annotations/<video_id>.json    {fps, num_frames, instances:[...], boundaries:[...]}
          splits/{train,val}.bundle      "<video_id>.txt" per line

    Unlike `DatasetFolder` (which only has access to a frame-wise class
    sequence and must *re-derive* segments by run-length splitting on class
    change -- unable to separate two adjacent same-class instances), this
    loader reads the real `instances` list directly from
    `annotations/<video_id>.json`, so adjacent/non-adjacent same-class
    instances are always kept separate, exactly as required by
    problem_definition.md.

    Returns:
        'train': sample (C, T') float32, target (T',) int64, instances (N, 3)
                 int64 [start_frame, end_frame, class_id] -- ALL at the SAME
                 subsampled resolution T' = ceil(T / sample_rate), fname, and
                 noise (C, T') float32.
        'val':   sample (C, T') float32 (subsampled), target (T,) int64 and
                 instances (N, 3) int64 both at FULL resolution T, fname.
                 (mirrors `DatasetFolder`: main.py::validate() upsamples model
                 outputs back to full resolution before computing metrics)
    """

    def __init__(
            self,
            root: str,
            config: yacs.config.CfgNode,
            state: str,
    ) -> None:
        super(InstanceDatasetFolder, self).__init__(root)
        self.sample_rate = config.sample_rate
        self.state = state

        root = os.path.expanduser(root)
        self.class_to_idx = self._find_classes(root)

        bundle_name = 'train.bundle' if state == 'train' else 'val.bundle'
        bundle_path = os.path.join(root, 'splits', bundle_name)
        with open(bundle_path, 'r') as f:
            video_ids = [line.strip()[:-4] for line in f.readlines() if line.strip()]

        # `config` passed to InstanceDatasetFolder is already `config.dataset`
        feat_folders = (
            getattr(config, 'feat_folders', None)
            or getattr(config, 'feat_folder', None)
            or (getattr(config.dataset, 'feat_folders', None) if hasattr(config, 'dataset') else None)
        )
        if feat_folders is None:
            feat_folders = ['features']
        elif isinstance(feat_folders, str):
            cleaned = feat_folders.strip("[] \t\r\n")
            if ',' in cleaned:
                feat_folders = [s.strip().strip("'\"") for s in cleaned.split(',')]
            elif '+' in cleaned:
                feat_folders = [s.strip().strip("'\"") for s in cleaned.split('+')]
            else:
                feat_folders = [cleaned.strip("'\"")]
        self.feat_folders = list(feat_folders)

        self.samples = []
        for vid in video_ids:
            feat_paths = [os.path.join(root, folder, vid + '.npy') for folder in self.feat_folders]
            ann_path = os.path.join(root, 'annotations', vid + '.json')
            self.samples.append((feat_paths, ann_path, vid))

        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 samples for state={state} in {root}")

    def _find_classes(self, root: str) -> Dict[str, int]:
        with open(os.path.join(root, 'mapping.txt'), 'r') as f:
            lines = f.readlines()
        class_to_idx = {}
        for line in lines:
            idx, name = line.split(maxsplit=1)
            class_to_idx[name.strip()] = int(idx)
        return class_to_idx

    def _load_annotation(self, ann_path: str) -> dict:
        with open(ann_path, 'r') as f:
            return json.load(f)

    def _instances_to_array(self, instances: List[dict]) -> np.ndarray:
        if len(instances) == 0:
            return np.zeros((0, 3), dtype=np.int64)
        return np.array(
            [[inst['start_frame'], inst['end_frame'], inst['class_id']] for inst in instances],
            dtype=np.int64,
        )

    def _instances_to_frame_labels(self, instances: List[dict], n_frames: int) -> np.ndarray:
        labels = np.zeros(n_frames, dtype=np.int64)
        for inst in instances:
            labels[inst['start_frame']: inst['end_frame']] = inst['class_id']
        return labels

    def __getitem__(self, index: int):
        feat_paths, ann_path, fname = self.samples[index]
        if len(feat_paths) == 1:
            sample_full = np.load(feat_paths[0])
        else:
            feats = []
            for fp in feat_paths:
                if not os.path.exists(fp):
                    raise FileNotFoundError(
                        f"Feature file '{fp}' not found. Please ensure feature extraction is complete for all videos."
                    )
                feats.append(np.load(fp))
            # align temporal dimension if any minor discrepancy
            min_t = min(f.shape[1] for f in feats)
            feats = [f[:, :min_t] for f in feats]
            sample_full = np.concatenate(feats, axis=0)

        ann = self._load_annotation(ann_path)
        n_frames = ann['num_frames']
        target_full = self._instances_to_frame_labels(ann['instances'], n_frames)  # full resolution
        if sample_full.shape[1] != target_full.shape[0]:
            min_len = min(sample_full.shape[1], target_full.shape[0])
            sample_full = sample_full[:, :min_len]
            target_full = target_full[:min_len]

        if self.state == 'train':
            sample = sample_full[:, ::self.sample_rate]
            # subsample target too, matching DatasetFolder's train behaviour
            # (data and target share the same resolution during training;
            # only val keeps the target at full resolution -- see below)
            target = target_full[::self.sample_rate]

            if self.sample_rate != 1:
                instances = []
                for inst in ann['instances']:
                    s = inst['start_frame'] // self.sample_rate
                    e = max(s + 1, -(-inst['end_frame'] // self.sample_rate))  # ceil div, at least width 1
                    instances.append({'start_frame': s, 'end_frame': e, 'class_id': inst['class_id']})
            else:
                instances = ann['instances']
            instances_arr = self._instances_to_array(instances)

            noise = np.random.randn(*sample.shape).astype(np.float32)
            return sample.astype(np.float32), target, instances_arr, fname, noise

        else:  # val: sample subsampled, target/instances kept at FULL resolution
            # (mirrors DatasetFolder: main.py::validate() upsamples model
            # outputs back to full resolution via repeat_interleave/interpolate
            # before computing metrics against this full-res target)
            sample = sample_full[:, ::self.sample_rate]
            instances_arr = self._instances_to_array(ann['instances'])  # full resolution, matches `target_full`
            return sample.astype(np.float32), target_full, instances_arr, fname

    def __len__(self) -> int:
        return len(self.samples)


def create_instance_dataset(config: yacs.config.CfgNode,
                             is_train: bool) -> Union[Tuple[Dataset, Dataset], Dataset]:
    # Unlike DatasetFolder's convention (dataset_dir/<config.dataset.name>/),
    # `dataset.dataset_dir` IS the dataset root directly for tas_instance
    # (i.e. point it at .../dataset_tas_instance), since the on-disk folder
    # name doesn't match the dataset.name='tas_instance' config key.
    dataset_root = pathlib.Path(config.dataset.dataset_dir).expanduser()
    if is_train:
        train_dataset = InstanceDatasetFolder(dataset_root, config.dataset, 'train')
        val_dataset = InstanceDatasetFolder(dataset_root, config.dataset, 'val')
        return train_dataset, val_dataset
    else:
        val_dataset = InstanceDatasetFolder(dataset_root, config.dataset, 'val')
        return val_dataset
