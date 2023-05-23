import logging
import torch
from torch.utils.data import Sampler



logger = logging.getLogger(__name__)

class ShuffleWarpSampler(Sampler):
    def __init__(self, data_source, n_repeats=5):
        super().__init__(data_source)
        self.data_source = data_source
        self.n_repeats = n_repeats
        logger.info("[ShuffleWarpSampler] Expanded data size: %s", len(self))

    def __iter__(self):
        shuffle_idx = []
        for i in range(self.n_repeats):
            sub_epoch = torch.randperm(len(self.data_source)).tolist()
            shuffle_idx = shuffle_idx + sub_epoch
        return iter(shuffle_idx)

    def __len__(self):
        return len(self.data_source) * self.n_repeats
