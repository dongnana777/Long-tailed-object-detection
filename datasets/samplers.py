# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from codes in torch.utils.data.distributed
# ------------------------------------------------------------------------

import os
import math
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from util.misc import get_rank, get_world_size
from collections import defaultdict
import numpy as np
from torch.utils.data.sampler import BatchSampler
import bisect
import itertools
from datasets.lvis_classes import *

class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class NodeDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, local_rank=None, local_size=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        if local_size is None:
            local_size = int(os.environ.get('LOCAL_SIZE', 1))
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.num_parts = local_size
        self.rank = rank
        self.local_rank = local_rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.total_size_parts = self.num_samples * self.num_replicas // self.num_parts

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()
        indices = [i for i in indices if i % self.num_parts == self.local_rank]

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size_parts - len(indices))]
        assert len(indices) == self.total_size_parts

        # subsample
        indices = indices[self.rank // self.num_parts:self.total_size_parts:self.num_replicas // self.num_parts]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedRepeatFactorReSampler_teacher(Sampler):
    """ Suitable for long-tail distribution datasets.
    Refer to `LVIS <https://arxiv.org/abs/1908.03195>`_ paper
    """
    def __init__(self, dataset, t=0.00001, ri_mode='ceil', pn=0.5,
                 ri_if_empty=1, num_replicas=None, static_size=False, rank=None, shuffle=True):
        """
        Arguments:
            - dataset (:obj:`Dataset`): dataset used for sampling.
            - t (:obj:`float`):  thresh- old that intuitively controls the point at which oversampling kicks in (0.001)
            - ri_mode (:obj:`str`): choices={floor, round, random_round, ceil, c_ceil_r_f_floor}, method to compute
              repeat factor for one image
            - pn (:obj:`float`): power number (0.5)
            - num_replicas (int): number of processes participating in distributed training, optional.
            - rank (int): rank of the current process within num_replicas, optional.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.original_num_samples = self.num_samples
        self.t = t
        self.ri_mode = ri_mode
        self.ri_if_empty = int(ri_if_empty)
        self.pn = pn
        self.static_size = static_size
        self._prepare()
        self.shuffle = shuffle

    def _prepare(self):
        # prepare re-sampling factor for category
        rc = defaultdict(int)
        img_num_per_class = defaultdict(int)
        for cls, img_num in sorted(self.dataset.num_images_per_class.items()):
            if cls in freq_classes_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            elif cls in common_30_100_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            elif cls in common_10_30_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            else:
                img_num_per_class[cls] = img_num
                rc[cls] = 1


        self.rc = rc

    def _compute_ri(self, img_index):
        classes = self.dataset.get_image_classes(img_index)
        ris = [self.rc[cls] for cls in classes]
        if len(ris) == 0:
            return self.ri_if_empty
        if self.ri_mode == 'floor':
            ri = int(max(ris))
        elif self.ri_mode == 'round':
            ri = round(max(ris))
        elif self.ri_mode == 'random_round':
            ri_max = max(ris)
            p = ri_max - int(ri_max)
            if np.random.rand() < p:
                ri = math.ceil(ri_max)
            else:
                ri = int(ri_max)
        elif self.ri_mode == 'ceil':
            ri = math.ceil(max(ris))
        elif self.ri_mode == 'c_ceil_r_f_floor':
            max_ind = np.argmax(ris)
            assert hasattr(self.dataset, 'lvis'), 'Only lvis dataset supportted for c_ceil_r_f_floor mode'
            img_id = self.dataset.img_ids[img_index]
            meta_annos = self.dataset.lvis.img_ann_map[img_id]
            f = self.dataset.lvis.cats[meta_annos[max_ind]['category_id']]['frequency']
            assert f in ['f', 'c', 'r']
            if f in ['r', 'f']:
                ri = int(max(ris))
            else:
                ri = math.ceil(max(ris))
        else:
            raise NotImplementedError
        return ri

    def _get_new_indices(self):
        indices = []
        for idx in range(len(self.dataset)):
            ri = self._compute_ri(idx)
            indices += [idx] * ri

        return indices

    def __iter__(self):
        # deterministically shuffle based on epoch

        # generate a perm based using class-aware balance for this epoch
        indices = self._get_new_indices()

        # override num_sample total size
        self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        indices = np.random.RandomState(seed=self.epoch).permutation(np.array(indices))
        indices = list(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        # convert to int because this array will be converted to torch.tensor,
        # but torch.as_tensor dosen't support numpy.int64
        # a = torch.tensor(np.float64(1))  # works
        # b = torch.tensor(np.int64(1))  # fails
        indices = list(map(lambda x: int(x), indices))
        return iter(indices)

    def __len__(self):
        return self.original_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedRepeatFactorReSampler_0(Sampler):
    """ Suitable for long-tail distribution datasets.
    Refer to `LVIS <https://arxiv.org/abs/1908.03195>`_ paper
    """
    def __init__(self, dataset, t=0.00001, ri_mode='ceil', pn=0.5,
                 ri_if_empty=1, num_replicas=None, static_size=False, rank=None, shuffle=True):
        """
        Arguments:
            - dataset (:obj:`Dataset`): dataset used for sampling.
            - t (:obj:`float`):  thresh- old that intuitively controls the point at which oversampling kicks in (0.001)
            - ri_mode (:obj:`str`): choices={floor, round, random_round, ceil, c_ceil_r_f_floor}, method to compute
              repeat factor for one image
            - pn (:obj:`float`): power number (0.5)
            - num_replicas (int): number of processes participating in distributed training, optional.
            - rank (int): rank of the current process within num_replicas, optional.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.original_num_samples = self.num_samples
        self.t = t
        self.ri_mode = ri_mode
        self.ri_if_empty = int(ri_if_empty)
        self.pn = pn
        self.static_size = static_size
        self._prepare()
        self.shuffle = shuffle

    def _prepare(self):
        # prepare re-sampling factor for category
        rc = defaultdict(int)
        img_num_per_class = defaultdict(int)
        for cls, img_num in sorted(self.dataset.num_images_per_class.items()):
            if cls in freq_classes_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            elif cls in common_30_100_v5:
                t = 0.003
                pn = 0.3
                f = img_num / len(self.dataset)
                img_num_per_class[cls] = img_num
                rc[cls] = max(1, math.pow(t / f, pn))
            elif cls in common_10_30_v5:
                t = 0.003
                pn = 0.5
                f = img_num / len(self.dataset)
                img_num_per_class[cls] = img_num
                rc[cls] = max(1, math.pow(t / f, pn))
            else:
                t = 0.003
                pn = 0.5
                f = img_num / len(self.dataset)
                img_num_per_class[cls] = img_num
                rc[cls] = max(1, math.pow(t / f, pn))
        self.rc = rc

    def _compute_ri(self, img_index):
        classes = self.dataset.get_image_classes(img_index)
        ris = [self.rc[cls] for cls in classes]
        if len(ris) == 0:
            return self.ri_if_empty
        if self.ri_mode == 'floor':
            ri = int(max(ris))
        elif self.ri_mode == 'round':
            ri = round(max(ris))
        elif self.ri_mode == 'random_round':
            ri_max = max(ris)
            p = ri_max - int(ri_max)
            if np.random.rand() < p:
                ri = math.ceil(ri_max)
            else:
                ri = int(ri_max)
        elif self.ri_mode == 'ceil':
            ri = math.ceil(max(ris))
        elif self.ri_mode == 'c_ceil_r_f_floor':
            max_ind = np.argmax(ris)
            assert hasattr(self.dataset, 'lvis'), 'Only lvis dataset supportted for c_ceil_r_f_floor mode'
            img_id = self.dataset.img_ids[img_index]
            meta_annos = self.dataset.lvis.img_ann_map[img_id]
            f = self.dataset.lvis.cats[meta_annos[max_ind]['category_id']]['frequency']
            assert f in ['f', 'c', 'r']
            if f in ['r', 'f']:
                ri = int(max(ris))
            else:
                ri = math.ceil(max(ris))
        else:
            raise NotImplementedError
        return ri

    def _get_new_indices(self):
        indices = []
        for idx in range(len(self.dataset)):
            ri = self._compute_ri(idx)
            indices += [idx] * ri

        return indices

    def __iter__(self):
        # deterministically shuffle based on epoch

        # generate a perm based using class-aware balance for this epoch
        indices = self._get_new_indices()

        # override num_sample total size
        self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        indices = np.random.RandomState(seed=self.epoch).permutation(np.array(indices))
        indices = list(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        # convert to int because this array will be converted to torch.tensor,
        # but torch.as_tensor dosen't support numpy.int64
        # a = torch.tensor(np.float64(1))  # works
        # b = torch.tensor(np.int64(1))  # fails
        indices = list(map(lambda x: int(x), indices))
        return iter(indices)

    def __len__(self):
        return self.original_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedRepeatFactorReSampler_1(Sampler):
    """ Suitable for long-tail distribution datasets.
    Refer to `LVIS <https://arxiv.org/abs/1908.03195>`_ paper
    """
    def __init__(self, dataset, t=0.00001, ri_mode='ceil', pn=0.5,
                 ri_if_empty=1, num_replicas=None, static_size=False, rank=None, shuffle=True):
        """
        Arguments:
            - dataset (:obj:`Dataset`): dataset used for sampling.
            - t (:obj:`float`):  thresh- old that intuitively controls the point at which oversampling kicks in (0.001)
            - ri_mode (:obj:`str`): choices={floor, round, random_round, ceil, c_ceil_r_f_floor}, method to compute
              repeat factor for one image
            - pn (:obj:`float`): power number (0.5)
            - num_replicas (int): number of processes participating in distributed training, optional.
            - rank (int): rank of the current process within num_replicas, optional.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.original_num_samples = self.num_samples
        self.t = t
        self.ri_mode = ri_mode
        self.ri_if_empty = int(ri_if_empty)
        self.pn = pn
        self.static_size = static_size
        self._prepare()
        self.shuffle = shuffle

    def _prepare(self):
        # prepare re-sampling factor for category
        rc = defaultdict(int)
        img_num_per_class = defaultdict(int)
        for cls, img_num in sorted(self.dataset.num_images_per_class.items()):
            if cls in freq_classes_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            elif cls in common_30_100_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            elif cls in common_10_30_v5:
                t = 0.0015
                pn = 1
                f = img_num / len(self.dataset)
                img_num_per_class[cls] = img_num
                rc[cls] = max(1, math.pow(t / f, pn))
            else:
                t = 0.0015
                pn = 1
                f = img_num / len(self.dataset)
                img_num_per_class[cls] = img_num
                rc[cls] = max(1, math.pow(t / f, pn))

        self.rc = rc

    def _compute_ri(self, img_index):
        classes = self.dataset.get_image_classes(img_index)
        ris = [self.rc[cls] for cls in classes]
        if len(ris) == 0:
            return self.ri_if_empty
        if self.ri_mode == 'floor':
            ri = int(max(ris))
        elif self.ri_mode == 'round':
            ri = round(max(ris))
        elif self.ri_mode == 'random_round':
            ri_max = max(ris)
            p = ri_max - int(ri_max)
            if np.random.rand() < p:
                ri = math.ceil(ri_max)
            else:
                ri = int(ri_max)
        elif self.ri_mode == 'ceil':
            ri = math.ceil(max(ris))
        elif self.ri_mode == 'c_ceil_r_f_floor':
            max_ind = np.argmax(ris)
            assert hasattr(self.dataset, 'lvis'), 'Only lvis dataset supportted for c_ceil_r_f_floor mode'
            img_id = self.dataset.img_ids[img_index]
            meta_annos = self.dataset.lvis.img_ann_map[img_id]
            f = self.dataset.lvis.cats[meta_annos[max_ind]['category_id']]['frequency']
            assert f in ['f', 'c', 'r']
            if f in ['r', 'f']:
                ri = int(max(ris))
            else:
                ri = math.ceil(max(ris))
        else:
            raise NotImplementedError
        return ri

    def _get_new_indices(self):
        indices = []
        for idx in range(len(self.dataset)):
            ri = self._compute_ri(idx)
            indices += [idx] * ri

        return indices

    def __iter__(self):
        # deterministically shuffle based on epoch

        # generate a perm based using class-aware balance for this epoch
        indices = self._get_new_indices()

        # override num_sample total size
        self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        indices = np.random.RandomState(seed=self.epoch).permutation(np.array(indices))
        indices = list(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        # convert to int because this array will be converted to torch.tensor,
        # but torch.as_tensor dosen't support numpy.int64
        # a = torch.tensor(np.float64(1))  # works
        # b = torch.tensor(np.int64(1))  # fails
        indices = list(map(lambda x: int(x), indices))
        return iter(indices)

    def __len__(self):
        return self.original_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DistributedRepeatFactorReSampler_2(Sampler):
    """ Suitable for long-tail distribution datasets.
    Refer to `LVIS <https://arxiv.org/abs/1908.03195>`_ paper
    """
    def __init__(self, dataset, t=0.00001, ri_mode='ceil', pn=0.5,
                 ri_if_empty=1, num_replicas=None, static_size=False, rank=None, shuffle=True):
        """
        Arguments:
            - dataset (:obj:`Dataset`): dataset used for sampling.
            - t (:obj:`float`):  thresh- old that intuitively controls the point at which oversampling kicks in (0.001)
            - ri_mode (:obj:`str`): choices={floor, round, random_round, ceil, c_ceil_r_f_floor}, method to compute
              repeat factor for one image
            - pn (:obj:`float`): power number (0.5)
            - num_replicas (int): number of processes participating in distributed training, optional.
            - rank (int): rank of the current process within num_replicas, optional.
        """
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.original_num_samples = self.num_samples
        self.t = t
        self.ri_mode = ri_mode
        self.ri_if_empty = int(ri_if_empty)
        self.pn = pn
        self.static_size = static_size
        self._prepare()
        self.shuffle = shuffle

    def _prepare(self):
        # prepare re-sampling factor for category
        rc = defaultdict(int)
        img_num_per_class = defaultdict(int)
        for cls, img_num in sorted(self.dataset.num_images_per_class.items()):
            # if cls in freq_classes_v5:
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = 1
            # elif cls in common_30_100_v5:
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = 1
            # elif cls in common_10_30_v5:
            #     t = 0.0015
            #     pn = 1
            #     f = img_num / len(self.dataset)
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = max(1, math.pow(t / f, pn))
            # else:
            #     t = 0.0015
            #     pn = 1
            #     f = img_num / len(self.dataset)
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = max(1, math.pow(t / f, pn))
        ###
            # if cls in freq_classes_v5:
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = 1
            # elif cls in common_30_100_v5:
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = 1
            # elif cls in common_10_30_v5:
            #     t = 0.0015
            #     pn = 0.5
            #     f = img_num / len(self.dataset)
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = max(1, math.pow(t / f, pn))
            # else:
            #     t = 0.0015
            #     pn = 0.5
            #     f = img_num / len(self.dataset)
            #     img_num_per_class[cls] = img_num
            #     rc[cls] = max(1, math.pow(t / f, pn))
        ###
            if cls in freq_classes_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            elif cls in common_30_100_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            elif cls in common_10_30_v5:
                img_num_per_class[cls] = img_num
                rc[cls] = 1
            else:
                t = 0.0015
                pn = 0.5
                f = img_num / len(self.dataset)
                img_num_per_class[cls] = img_num
                rc[cls] = max(1, math.pow(t / f, pn))
        self.rc = rc

    def _compute_ri(self, img_index):
        classes = self.dataset.get_image_classes(img_index)
        ris = [self.rc[cls] for cls in classes]
        if len(ris) == 0:
            return self.ri_if_empty
        if self.ri_mode == 'floor':
            ri = int(max(ris))
        elif self.ri_mode == 'round':
            ri = round(max(ris))
        elif self.ri_mode == 'random_round':
            ri_max = max(ris)
            p = ri_max - int(ri_max)
            if np.random.rand() < p:
                ri = math.ceil(ri_max)
            else:
                ri = int(ri_max)
        elif self.ri_mode == 'ceil':
            ri = math.ceil(max(ris))
        elif self.ri_mode == 'c_ceil_r_f_floor':
            max_ind = np.argmax(ris)
            assert hasattr(self.dataset, 'lvis'), 'Only lvis dataset supportted for c_ceil_r_f_floor mode'
            img_id = self.dataset.img_ids[img_index]
            meta_annos = self.dataset.lvis.img_ann_map[img_id]
            f = self.dataset.lvis.cats[meta_annos[max_ind]['category_id']]['frequency']
            assert f in ['f', 'c', 'r']
            if f in ['r', 'f']:
                ri = int(max(ris))
            else:
                ri = math.ceil(max(ris))
        else:
            raise NotImplementedError
        return ri

    def _get_new_indices(self):
        indices = []
        for idx in range(len(self.dataset)):
            ri = self._compute_ri(idx)
            indices += [idx] * ri

        return indices

    def __iter__(self):
        # deterministically shuffle based on epoch

        # generate a perm based using class-aware balance for this epoch
        indices = self._get_new_indices()

        # override num_sample total size
        self.num_samples = int(math.ceil(len(indices) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        indices = np.random.RandomState(seed=self.epoch).permutation(np.array(indices))
        indices = list(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        # convert to int because this array will be converted to torch.tensor,
        # but torch.as_tensor dosen't support numpy.int64
        # a = torch.tensor(np.float64(1))  # works
        # b = torch.tensor(np.int64(1))  # fails
        indices = list(map(lambda x: int(x), indices))
        return iter(indices)

    def __len__(self):
        return self.original_num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class AspectRatioGroupedBatchSampler(BatchSampler):
    """Wraps another sampler to yield a mini-batch of indices.
    It enforces that elements from the same group should appear in groups of batch_size.
    It also tries to provide mini-batches which follows an ordering which is
    as close as possible to the ordering from the original sampler.

    """

    def __init__(self, sampler, batch_size, aspect_grouping, drop_last=False, training=True):
        """
        Arguments:
             - sampler (:obj:`sampler`): instance of sampler object
             - batch_size (:obj:`int`)
             - aspect_grouping (:obj:`list` of `float`): split point of aspect ration
        """
        super(AspectRatioGroupedBatchSampler, self).__init__(sampler, batch_size, drop_last)
        self.group_ids = self._get_group_ids(sampler.dataset, aspect_grouping)
        assert self.group_ids.dim() == 1
        self.training = training

        self.groups = torch.unique(self.group_ids).sort(0)[0]
        self._can_reuse_batches = False

    def _get_group_ids(self, dataset, aspect_grouping):
        assert isinstance(aspect_grouping, (list, tuple)), "aspect_grouping must be a list or tuple"
        assert hasattr(dataset, 'aspect_ratios'), 'Image aspect ratios are required'

        # devide images into groups by aspect ratios
        aspect_ratios = dataset.aspect_ratios
        aspect_grouping = sorted(aspect_grouping)
        group_ids = list(map(lambda y: bisect.bisect_right(aspect_grouping, y), aspect_ratios))
        return torch.as_tensor(group_ids)

    def _prepare_batches(self):
        sampled_ids = torch.as_tensor(list(self.sampler))
        sampled_group_ids = self.group_ids[sampled_ids]
        clusters = [sampled_ids[sampled_group_ids == i] for i in self.groups]
        target_batch_num = int(np.ceil(len(sampled_ids) / float(self.batch_size)))

        splits = [c.split(self.batch_size) for c in clusters if len(c) > 0]
        merged = tuple(itertools.chain.from_iterable(splits))

        # re-permuate the batches by the order that
        # the first element of each batch occurs in the original sampled_ids
        first_element_of_batch = [t[0].item() for t in merged]
        inv_sampled_ids_map = {v: k for k, v in enumerate(sampled_ids.tolist())}
        first_index_of_batch = torch.as_tensor(
            [inv_sampled_ids_map[s] for s in first_element_of_batch])
        permutation_order = first_index_of_batch.sort(0)[1].tolist()
        batches = [merged[i].tolist() for i in permutation_order]

        if self.training:
            # ensure number of batches in different gpus are the same
            if len(batches) > target_batch_num:
                batches = batches[:target_batch_num]
            assert len(batches) == target_batch_num, "Error, uncorrect target_batch_num!"
        return batches

    def __iter__(self):
        if self._can_reuse_batches:
            batches = self._batches
            self._can_reuse_batches = False
        else:
            batches = self._prepare_batches()
        self._batches = batches
        return iter(batches)

    def __len__(self):
        if getattr(self.sampler, 'static_size', False):
            return int(np.ceil(len(self.sampler) / float(self.batch_size)))
        if not hasattr(self, "_batches"):
            self._batches = self._prepare_batches()
            self._can_reuse_batches = True
        return len(self._batches)