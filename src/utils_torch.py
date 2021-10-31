import os
import random
from typing import Tuple

import numpy as np
import torch
from torchinfo import summary


def seed_everything(seed: int = 42, deterministic: bool = False, benchmark: bool = True) -> None:
    """
    Args:
        deterministic: True -> Reproducibility, False -> Performance
        benchmark: False -> Reproducibility, True -> Performance
    Returns:
    See Also:
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark


def summary_model(
    model,
    input_size: Tuple[int, int, int, int] = (32, 3, 224, 224),
    path: str = "path/to/output/file.txt",
    verbose: int = 0,
):
    """
    Args:
        input_size (Tuple[int, int, int, int]): (batch_size, num_channels, height, width)
    """
    model_stats = summary(model, input_size=input_size, verbose=verbose)
    model_str = str(model_stats)
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(model_str)
