import logging
import os

import torch

log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=logging.getLevelName(log_level))
CUDA_DEVICE = "cuda:0"
device = torch.device(CUDA_DEVICE if torch.cuda.is_available() and os.getenv("USE_GPU") else "cpu")
