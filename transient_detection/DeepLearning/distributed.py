import os
import functools

import torch
import numpy as np
import torch.distributed as dist

from collections import defaultdict, deque
import time, datetime
from pathlib import Path

from transient_detection.DeepLearning.utilities import print_with_rank_index

"""
Mostly copy-pasted from torchvision references or other public repos like DETR:
https://github.com/facebookresearch/detr/blob/master/util/misc.py
"""

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_shared_folder(args) -> Path:
    p = Path(args["PATHS"]["out"])
    if p.is_dir():
        return p
    raise RuntimeError(f"No valid out folder available, got {args['PATHS']['out']}.")

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, iterable, print_freq, header=None, delimiter="\t", unit_of_byte_size = 1024 * 1024):
        self.meters = defaultdict(SmoothedValue)
        self.iterable = iterable
        self.print_freq = print_freq
        self.delimiter = delimiter
        self.header = header if not header else ''
        self.unit_of_byte_size = unit_of_byte_size
        self.print = functools.partial(print_with_rank_index, int(os.environ.get("SLURM_LOCALID")) )

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter
    
    def __next__(self):
        if self.i > 0:
            self.iter_time.update(time.time() - self.end)
            if self.i % self.print_freq == 0 or self.i == len(self.iterable) - 1:
                eta_seconds = self.iter_time.global_avg * (len(self.iterable) - self.skipped - self.i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.print(self.log_msg.format(
                        self.i, len(self.iterable) - self.skipped, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time),
                        memory=torch.cuda.max_memory_allocated() / self.unit_of_byte_size,
                        time_of_msg=datetime.datetime.now()))
                else:
                    self.print(self.log_msg.format(
                        self.i, len(self.iterable) - self.skipped, eta=eta_string,
                        meters=str(self),
                        time=str(self.iter_time), data=str(self.data_time),
                        time_of_msg=datetime.datetime.now()))
        
        self.end = time.time()

        while True:
            try:
                obj = next(self.iterator)
            except StopIteration:
                total_time = time.time() - self.start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                self.print('{} Total time: {} ({:.6f} s / it)'.format(
                    self.header, total_time_str, total_time / len(self.iterable)))
                raise StopIteration
            
            #########################
            # check if input data is too big
            MB = 1024 * 1024
            if obj.x.element_size() * obj.x.nelement() <= 90 * MB:
                break
            self.skipped += 1
            # self.print(f"Skipped since too big. Counting total {skipped} skipped files.")

        self.data_time.update(time.time() - self.end)
        ###########################
        # self.print("Iteration n°: ", self.i)
        ###########################
        self.i += 1
        return obj
    
    def __iter__(self):
        self.i = 0
        self.skipped = 0
    
        self.start_time = time.time()
        self.end = time.time()
        self.iter_time = SmoothedValue(fmt='{avg:.6f}')
        self.data_time = SmoothedValue(fmt='{avg:.6f}')

        space_fmt = ':' + str(len(str(len(self.iterable)))) + 'd'
        if torch.cuda.is_available():
            self.log_msg = self.delimiter.join([
                self.header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}',
                'time of msg: {time_of_msg}'
            ])
        else:
            self.log_msg = self.delimiter.join([
                self.header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'time of msg: {time_of_msg}'
            ])
        self.iterator = iter(self.iterable)
        return self


    # def log_every(self, iterable, print_freq, header=None):
    #     i = 0
    #     if not header:
    #         header = ''
    #     start_time = time.time()
    #     end = time.time()
    #     iter_time = SmoothedValue(fmt='{avg:.6f}')
    #     data_time = SmoothedValue(fmt='{avg:.6f}')
    #     space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
    #     if torch.cuda.is_available():
    #         log_msg = self.delimiter.join([
    #             header,
    #             '[{0' + space_fmt + '}/{1}]',
    #             'eta: {eta}',
    #             '{meters}',
    #             'time: {time}',
    #             'data: {data}',
    #             'max mem: {memory:.0f}'
    #         ])
    #     else:
    #         log_msg = self.delimiter.join([
    #             header,
    #             '[{0' + space_fmt + '}/{1}]',
    #             'eta: {eta}',
    #             '{meters}',
    #             'time: {time}',
    #             'data: {data}'
    #         ])
    #     MB = 1024.0 * 1024.0
    #     for obj in iterable:
    #         ###########################
    #         print("Iteration n°: ",i)
    #         ###########################
    #         data_time.update(time.time() - end)
    #         yield obj
    #         iter_time.update(time.time() - end)
    #         if i % print_freq == 0 or i == len(iterable) - 1:
    #             eta_seconds = iter_time.global_avg * (len(iterable) - i)
    #             eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
    #             if torch.cuda.is_available():
    #                 print(log_msg.format(
    #                     i, len(iterable), eta=eta_string,
    #                     meters=str(self),
    #                     time=str(iter_time), data=str(data_time),
    #                     memory=torch.cuda.max_memory_allocated() / MB))
    #             else:
    #                 print(log_msg.format(
    #                     i, len(iterable), eta=eta_string,
    #                     meters=str(self),
    #                     time=str(iter_time), data=str(data_time)))
    #         i += 1
    #         end = time.time()
    #     total_time = time.time() - start_time
    #     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    #     print('{} Total time: {} ({:.6f} s / it)'.format(
    #         header, total_time_str, total_time / len(iterable)))
