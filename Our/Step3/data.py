import math
import os
import random
from typing import Dict

import numpy as np
import torch
from torch.utils.data import IterableDataset

class ShuffleIterableDataset(IterableDataset):
    def __init__(self, dataset, buffer_size: int) -> None:
        super().__init__()
        assert buffer_size > 0
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buf = []
        for x in self.dataset:
            if len(buf) == self.buffer_size:
                idx = random.randint(0, self.buffer_size - 1)
                yield buf[idx]
                buf[idx] = x
            else:
                buf.append(x)
        random.shuffle(buf)
        while buf:
            yield buf.pop()

class IndividualForecastDataIter(IterableDataset):
    def __init__(self, dataset, transforms: torch.nn.Module, output_transforms: torch.nn.Module, win_size, region_info = None):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.output_transforms = output_transforms
        self.region_info = region_info
        self.win_size = win_size

    def __iter__(self):
        for (inp, out, variables, out_variables) in self.dataset:
            assert inp.shape[0] == out.shape[0]
            for i in range(0,inp.shape[0]-self.win_size, self.win_size):
                if self.region_info is not None:
                    yield self.transforms(inp[i:i+self.win_size]), self.output_transforms(out[i:i+self.win_size]), variables, out_variables, self.region_info
                else:
                    yield self.transforms(inp[i:i+self.win_size]), self.output_transforms(out[i:i+self.win_size]), variables, out_variables


class NpyReader(IterableDataset):
    def __init__(
        self,
        file_list,
        variables,
        out_variables,
        shuffle: bool = False
    ) -> None:
        super().__init__()
        self.file_list_inp = [f for f in file_list if 'inp' in f]
        self.file_list_out = [f for f in file_list if 'out' in f]
        assert len(self.file_list_inp) == len(self.file_list_out)
        self.variables = variables
        self.out_variables = out_variables if out_variables is not None else variables
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            index_shuf = list(range(len(self.file_list_inp)))
            random.shuffle(index_shuf)
            file_list_inp_shuf = []
            file_list_out_shuf = []
            for i in index_shuf:
                file_list_inp_shuf.append(self.file_list_inp[i])
                file_list_out_shuf.append(self.file_list_out[i])
            self.file_list_inp = file_list_inp_shuf
            self.file_list_out = file_list_out_shuf
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = 0
            iter_end = len(self.file_list_inp)
        else:
            if not torch.distributed.is_initialized():
                rank = 0
                world_size = 1
            else:
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            num_workers_per_ddp = worker_info.num_workers
            num_shards = num_workers_per_ddp * world_size
            per_worker = int(math.floor(len(self.file_list_inp) / float(num_shards)))
            worker_id = rank * num_workers_per_ddp + worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker

        for idx in range(iter_start, iter_end):
            path_inp = self.file_list_inp[idx]
            path_out = self.file_list_out[idx]
            data_inp = np.load(path_inp)
            data_out = np.load(path_out)
            yield (
                {k: data_inp[k] for k in self.variables},
                {k: data_out[k] for k in self.out_variables},
                self.variables, self.out_variables
            )


class Downscaling(IterableDataset):
    def __init__(self, dataset: NpyReader) -> None:
        super().__init__()
        self.dataset = dataset

    def __iter__(self):
        for data_inp, data_out, variables, out_variables in self.dataset:
            inp = np.concatenate([data_inp[k].astype(np.float32) for k in data_inp.keys()], axis=1)
            inp = torch.from_numpy(inp)
            out = np.concatenate([data_out[k].astype(np.float32) for k in data_out.keys()], axis=1)
            out = torch.from_numpy(out)

            yield inp, out, variables, out_variables