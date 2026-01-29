import torch
from torch import Tensor
import numpy as np

from typing import Optional
from jaxtyping import Bool, Float, Int
import os
import typing
import numpy.typing as npt

from collections.abc import Callable, Iterable
import math

from einops import rearrange, einsum

def MyGetBatch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = dataset.shape[0]
    ix = np.random.randint(0, data_len - context_length, size=(batch_size,))
    x = np.stack([dataset[i : i + context_length] for i in ix])
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in ix])
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)
    return x_tensor, y_tensor

def MySaveCheckpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)

def MyLoadCheckpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration