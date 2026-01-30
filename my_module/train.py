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

import tqdm
import pickle

from einops import rearrange, einsum

def MyGetBatch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = dataset.shape[0]
    ix = np.random.randint(0, data_len - context_length - 1, size=batch_size)

    x = np.empty((batch_size, context_length), dtype=np.int64)
    y = np.empty_like(x)

    for k, i in enumerate(ix):
        x[k] = dataset[i : i + context_length]
        y[k] = dataset[i + 1 : i + context_length + 1]

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )

def MySaveCheckpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], loss: Optional[list[float]] = None) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "loss": loss if loss is not None else [],
    }
    torch.save(checkpoint, out)

def MyLoadCheckpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, list[float]]:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    loss = checkpoint.get("loss", [])
    return iteration, loss

# 定义单次训练：

from my_module.optim import MyCrossEntropy, MyGradientClipping, MyAdamW, MyRateScheduling
from my_module.nn import MyTransformer

def SingleTrain(model: torch.nn.Module, optimizer: torch.optim.Optimizer, data: tuple[Tensor, Tensor], T: int, lrarg: tuple[float, float, int, int]) -> float:
    lr = MyRateScheduling(T, *lrarg)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    
    model.train()
    # x_batch, y_batch = MyGetBatch(dataset, batch_size, context_length, device)
    x_batch, y_batch = data  # (batch_size, context_length), (batch_size, context_length)
    
    logits = model(x_batch)  # (batch_size, context_length, vocab_size)
    logits = rearrange(logits, "b c v -> (b c) v")  # (batch_size * context_length, vocab_size)
    targets = rearrange(y_batch, "b c -> (b c)")  # (batch_size * context_length,)

    # ========= 只改这里 =========
    loss = torch.nn.functional.cross_entropy(logits, targets)
    # ===========================

    optimizer.zero_grad()
    loss.backward()
    
    # MyGradientClipping(model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


def _innerTrain(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    start_step: int,
    lrarg: tuple[float, float, int, int],
    max_iters: int,
) -> float:
    
    # 进度条，从T进度回复
    progress = tqdm.tqdm(
        range(start_step, max_iters),
        initial=start_step,
        total=max_iters,
        desc="Training",
        unit="iter",
    )
    
    losses = []
    
    try:
        for T in progress:
            data = MyGetBatch(dataset, batch_size, context_length, device)
            loss = SingleTrain(model, optimizer, data, T, lrarg)
            losses.append(loss)
        
        success = True
    except BaseException as e:
        progress.close()
        print(f"Training interrupted at step {T} due to exception: {e}")
        success = False
    
    return success, T, losses

def Train(
    path: str,
    dataset_path: str,
    meta_path: str,
    # 初创时需要的参数
    **meta_kwargs,
):
    # 读取数据集
    dataset = np.memmap(
        dataset_path,
        dtype=np.uint16,
        mode="r",
    )
    
    # 读取元信息
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
    else:
        meta = {}
        meta.update(meta_kwargs)
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
    
    if "device" in meta_kwargs:
        meta["device"] = meta_kwargs["device"]
    
    print(meta)

    device = meta.get("device", "cpu")
    
    model = MyTransformer(
        num_embeddings = meta["vocab_size"],
        d_model = meta["d_model"],
        num_layers = meta["num_layers"],
        num_heads = meta["num_heads"],
        theta= meta["theta"],
        max_seq_len= meta["max_seq_len"],
        d_ff = meta["d_ff"],
        device=device,
    ).to(device)
    
    # 尝试加载检查点
    optimizer = MyAdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    start_step = 0
    if os.path.exists(path):
        start_step, prev_losses = MyLoadCheckpoint(path, model, optimizer)
        print(f"Resumed from checkpoint at step {start_step}.")
    else:
        prev_losses = []
    
    success, final_step, losses = _innerTrain(
        model=model,
        optimizer=optimizer,
        dataset=dataset,
        batch_size=meta["batch_size"],
        context_length=meta["max_seq_len"],
        device=device,
        start_step=start_step,
        lrarg=(
            meta["lr_initial"],
            meta["lr_final"],
            meta["lr_warmup_iters"],
            meta["max_iters"],
        ),
        max_iters=meta["max_iters"],
    )
    
    MySaveCheckpoint(model, optimizer, final_step, path, loss=prev_losses + losses)

if __name__ == "__main__":
    
    import cProfile
    import pstats
    
    pr = cProfile.Profile()
    pr.enable()

    Train(
        path="/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_module/checkpoint.pth",
        dataset_path="/home/nipporita/大模型/Week 1/lfs-data/owt_valid.npy",
        meta_path="/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_module/meta.pkl",
        # 结构参数
        vocab_size=32000,
        num_layers=4,
        d_model=16,
        num_heads=4,
        d_ff=16,
        # 网络超参数
        theta=0.7,
        max_seq_len=128,
        # 训练超参数
        batch_size=16,
        lr_initial=1e-3,
        lr_final=1e-5,
        lr_warmup_iters=1000,
        max_iters=10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(30)
    
    