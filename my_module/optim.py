import torch
from torch import Tensor

from typing import Optional
from jaxtyping import Bool, Float, Int

from collections.abc import Callable, Iterable
import math

from einops import rearrange, einsum

from my_module.nn import MyLogSoftMax

def MyCrossEntropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    ce = -MyLogSoftMax(inputs) # (batch_size, vocab_size)
    
    batch_idx = torch.arange(inputs.shape[0], device=inputs.device)
    
    result = ce[batch_idx, targets] # (batch_size,)
    
    result = torch.mean(result, dim=0, keepdim=True)
    
    return result

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                # 获取与 p 相关联的状态。
                t = state.get("t", 0)
                # 从状态中获取迭代次数，若不存在则使用初始值。
                grad = p.grad.data
                # 获取损失函数对 p 的梯度。
                p.data -= lr / math.sqrt(t + 1) * grad
                # 原地更新权重张量。
                state["t"] = t + 1
                # 递增迭代次数。
        return loss


class MyAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta_1, beta_2 = group["betas"]
            eps = group["eps"]
            lamb = group["weight_decay"]
            # 获取学习率。
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                
                grad = p.grad.data
                m = beta_1 * m + (1 - beta_1) * grad
                v = beta_2 * v + (1 - beta_2) * (grad ** 2)
                
                lr_t = lr * math.sqrt(1 - beta_2 ** (t + 1)) / (1 - beta_1 ** (t + 1))
                
                p.data -= lr_t * m / (torch.sqrt(v) + eps) + lr * lamb * p.data
                
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1
                
        return loss

def MyRateScheduling(
    T: int,
    lr_max: float,
    lr_min: float,
    T_wormup: int,
    T_anneal: int,
):
    if T < T_wormup:
        return lr_max * (T) / T_wormup
    elif T <= T_anneal:
        return lr_min + (1+math.cos(math.pi * (T - T_wormup) / (T_anneal - T_wormup))) * 0.5 * (lr_max - lr_min)
    else:
        return lr_min

def MyGradientClipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> Float[Tensor, ""]:
    s = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        s += torch.norm(parameter.grad.data, p=2) ** 2
    grad_norm = torch.sqrt(s)
    
    if grad_norm > max_norm:
        for parameter in parameters:
            if parameter.grad is None:
                continue
            clip_coef = max_norm / (grad_norm + eps)
            parameter.grad.data.mul_(clip_coef)

if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    N = 10
    t = np.arange(0, N, 1)
    y = []
    lrs = [1,1e1,1e2]
    for lr in lrs:
        result = []
        weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
        opt = SGD([weights], lr=lr)
        for _ in range(N):
            opt.zero_grad()
            # Reset the gradients for all learnable parameters.
            loss = (weights**2).mean() # Compute a scalar loss value.
            result.append(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients.
            opt.step() # Run optimizer step.
        y.append(result)
    
    y = np.array(y)
    
    plt.plot(t, y[0], label="lr=1")
    plt.plot(t, y[1], label="lr=10")
    plt.plot(t, y[2], label="lr=100")
    # plt.plot(t, y[3], label="lr=1000")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("SGD with different learning rates")
    plt.show()