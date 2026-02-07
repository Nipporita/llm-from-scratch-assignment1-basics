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
import time
import pickle

from my_module.shutdown_now import shutdown_now

from einops import rearrange, einsum

import wandb

def create_model(meta: dict, device: str) -> torch.nn.Module:
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
    model = torch.compile(model)
    return model

def load_meta(meta_path: str) -> dict:
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta

def MyGetBatch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    data_len = dataset.shape[0]
    ix = np.random.randint(0, data_len - context_length, size=batch_size)

    x = np.empty((batch_size, context_length), dtype=np.int64)
    y = np.empty_like(x)

    for k, i in enumerate(ix):
        x[k] = dataset[i : i + context_length]
        y[k] = dataset[i + 1 : i + context_length + 1]

    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )

def MySaveCheckpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    train_log: dict,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "train_log": train_log,
    }
    torch.save(checkpoint, out)


def MyLoadCheckpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[str] = None,
):
    checkpoint = torch.load(src, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    train_log = checkpoint.get("train_log", None)
    return iteration, train_log

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

def SingleValidate(model: torch.nn.Module, dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> float:
    model.eval()
    with torch.no_grad():
        data = MyGetBatch(dataset, batch_size, context_length, device)
        x_batch, y_batch = data
        logits = model(x_batch)
        logits = rearrange(logits, "b c v -> (b c) v")
        targets = rearrange(y_batch, "b c -> (b c)")
        loss = torch.nn.functional.cross_entropy(logits, targets)
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
    val_dataset: Optional[npt.NDArray] = None,
    train_log: Optional[dict] = None,
):
    if train_log is None:
        train_log = {
            "train": [],
            "valid": [],
        }
    
    # 进度条，从T进度回复
    progress = tqdm.tqdm(
        range(start_step, max_iters),
        initial=start_step,
        total=max_iters,
        desc="Training",
        unit="iter",
    )
    
    start_time = time.time()
    target_time = 7 * 60 * 60  # 7 hours
    
    ex = None
    try:
        for T in progress:
            data = MyGetBatch(dataset, batch_size, context_length, device)
            loss = SingleTrain(model, optimizer, data, T, lrarg)
            elapsed = time.time() - start_time

            train_log["train"].append({
                "step": T,
                "loss": loss,
                "time": elapsed,
            })
            
            wandb.log({
                "train/loss": loss,
                "train/step": T,
                "train/time": elapsed,
            }, step=T)
            
            if val_dataset is not None and (T + 1) % 100 == 0:
                val_loss = SingleValidate(model, val_dataset, batch_size, context_length, device)
                train_log["valid"].append({
                    "step": T,
                    "loss": val_loss,
                    "time": elapsed,
                })
                wandb.log({
                    "valid/loss": val_loss,
                    "valid/step": T,
                    "valid/time": elapsed,
                }, step=T)
                
                progress.set_postfix({"loss": loss, "val_loss": val_loss})
            
            if time.time() - start_time > target_time:
                print(f"Reached target training time of {target_time} seconds at step {T}.")
                break
        
        success = True
    except BaseException as e:
        ex = e
        print(f"Training interrupted at step {T} due to exception: {e}")
        success = False
    finally:
        progress.close()
    
    return success, T, train_log, ex

def Train(
    path: str,
    dataset_path: str,
    meta_path: str,
    val_dataset_path: Optional[str] = None,
    # 初创时需要的参数
    **meta_kwargs,
):
    # 读取数据集
    dataset = np.memmap(
        dataset_path,
        dtype=np.uint16,
        mode="r",
    )
    
    if val_dataset_path is not None:
        val_dataset = np.memmap(
            val_dataset_path,
            dtype=np.uint16,
            mode="r",
        )
    else:
        val_dataset = None
    
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
    
    wandb.init(
        project="llm-from-scratch",
        config=meta,
    )

    device = meta.get("device", "cpu")
    
    model = create_model(meta, device)
    
    # 尝试加载检查点
    optimizer = MyAdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    start_step = 0
    if os.path.exists(path):
        start_step, train_log = MyLoadCheckpoint(path, model, optimizer, device)
        if train_log is None:
            train_log = {
                "meta": meta,
                "train": [],
                "valid": [],
            }
        print(f"Resumed from checkpoint at step {start_step}.")
    else:
        train_log = {
            "meta": meta,
            "train": [],
            "valid": [],
        }
    
    success, final_step, train_log, e = _innerTrain(
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
        val_dataset=val_dataset,
    )
    
    MySaveCheckpoint(
        model=model,
        optimizer=optimizer,
        iteration=final_step,
        out=path,
        train_log=train_log,
    )
    
    return not isinstance(e, KeyboardInterrupt)

if __name__ == "__main__":
    
    import cProfile
    import pstats
    
    pr = cProfile.Profile()
    pr.enable()

    
    import os
    main_path = "/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/"
    model_path = os.path.join(main_path, "my_model")
    dataset_path = "/home/nipporita/大模型/Week 1/lfs-data/TinyStoriesV2-GPT4-train.npy"
    valid_dataset_path = "/home/nipporita/大模型/Week 1/lfs-data/TinyStoriesV2-GPT4-valid.npy"
    
    for learning_rate in [1e-4, 1e-3, 1e-2, 1e-1]:
        print(f"Learning Rate: {learning_rate}")
    
        if_shutdown = Train(
            path=os.path.join(model_path, f"checkpoint_{learning_rate:.0e}.pth"),
            dataset_path=dataset_path,
            meta_path=os.path.join(model_path, f"meta_{learning_rate:.0e}.pkl"),
            valid_dataset_path=os.path.join(model_path, "valid_dataset.npy"),
            # 结构参数
            vocab_size=10000,
            num_layers=4,
            d_model=512,
            num_heads=16,
            d_ff=1344,
            # 网络超参数
            theta=10000,
            max_seq_len=256,
            # 训练超参数
            batch_size=128,
            lr_initial=learning_rate,
            lr_final=learning_rate * 0.01,
            lr_warmup_iters=1000,
            max_iters=10000,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    pr.disable()

    stats = pstats.Stats(pr)
    stats.strip_dirs()
    stats.sort_stats("cumtime")
    stats.print_stats(30)
    
    if if_shutdown:
        shutdown_now()