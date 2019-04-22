import random
import torch
import numpy
import mmap


class AverageMeter(object):
    """Computes and stores the average and current value."""
    # adapted from: https://github.com/facebookresearch/DrQA
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_environment(seed, set_cuda=False):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and set_cuda:
        torch.cuda.manual_seed_all(seed)


def count_lines(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines

