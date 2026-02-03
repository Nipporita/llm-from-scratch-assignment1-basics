import torch

from typing import Optional
from jaxtyping import Bool, Float, Int

from einops import rearrange, einsum

class MyLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), device=self.device, dtype=self.dtype))
        
        self.weight = torch.nn.init.trunc_normal_(self.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # y = Wx
        y = einsum(self.weight, x, 'o i, ... i -> ... o')
        return y

class MyEmbedding(torch.nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        
        self.weight = torch.nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))
        
        self.weight = torch.nn.init.trunc_normal_(self.weight, std=0.02)
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]

class MyRMSNorm(torch.nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        
        self.weight = torch.nn.Parameter(torch.ones((d_model,), device=self.device, dtype=self.dtype))
        
        self.weight = torch.nn.init.trunc_normal_(self.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.sum(x * x, dim=-1, keepdim=True) / self.d_model + self.eps)
        
        result = x / rms * self.weight
        
        return result.to(in_dtype)


def MySiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class MySwiGLU(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        
        self.weight_w1 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.weight_w1 = torch.nn.init.trunc_normal_(self.weight_w1, std=0.02)
        
        self.weight_w2 = torch.nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.weight_w2 = torch.nn.init.trunc_normal_(self.weight_w2, std=0.02)
        
        self.weight_w3 = torch.nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.weight_w3 = torch.nn.init.trunc_normal_(self.weight_w3, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # SiLU(x) = x · σ(x)
        # SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)
        
        w1x = einsum(self.weight_w1, x, 'd_ff d_model, ... d_model -> ... d_ff')
        siluw1x = torch.sigmoid(w1x) * w1x
        
        w3x = einsum(self.weight_w3, x, 'd_ff d_model, ... d_model -> ... d_ff')
        
        result = einsum(self.weight_w2, siluw1x * w3x, 'd_model d_ff, ... d_ff -> ... d_model')
        
        return result

class MyRoPE(torch.nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.dtype = dtype if dtype is not None else torch.float32
        
        pos = torch.arange(max_seq_len)
        dim = torch.arange(0, d_k, 2)

        freqs = pos[:, None] / (theta ** (dim / d_k))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        
        # x shape: (..., d_k)
        x = x.view(*x.shape[:-1], self.d_k//2, 2)

        x1 = x[..., 0]
        x2 = x[..., 1]

        out1 =  cos * x1 - sin * x2
        out2 =  sin * x1 + cos * x2

        out = torch.stack([out1, out2], dim=-1).view(*x.shape[:-2], self.d_k)
        
        return out
        
def MySoftMax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x_exp / x_exp_sum

def MyLogSoftMax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_exp_sum = torch.sum(x_exp, dim=dim, keepdim=True)
    return x - x_max - torch.log(x_exp_sum)

def MyAttention(
    Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... values d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
    preserve_head: bool = False,
) -> Float[torch.Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    
    if not preserve_head:
        QtK = einsum(Q, K, '... q d_k, ... k d_k -> ... q k')
    else:
        QtK = einsum(Q, K, '... h q d_k, ... h k d_k -> ... h q k')
        
    scores = QtK / torch.sqrt(torch.tensor(d_k, dtype=Q.dtype))
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    attn_weights = MySoftMax(scores, dim=-1)
    
    if not preserve_head:
        result = einsum(attn_weights, V, '... q k, ... k d_v -> ... q d_v')
    else:
        result = einsum(attn_weights, V, '... h q k, ... h k d_v -> ... h q d_v')
    
    return result

class MyMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_k = self.d_v = d_model // num_heads
        
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.weight_WQ = torch.nn.Parameter(torch.empty((self.num_heads*self.d_k, self.d_model), device=device, dtype=dtype))
        self.weight_WQ = torch.nn.init.trunc_normal_(self.weight_WQ, std=0.02)
        
        self.weight_WK = torch.nn.Parameter(torch.empty((self.num_heads*self.d_k, self.d_model), device=device, dtype=dtype))
        self.weight_WK = torch.nn.init.trunc_normal_(self.weight_WK, std=0.02)
        
        self.weight_WV = torch.nn.Parameter(torch.empty((self.num_heads*self.d_v, self.d_model), device=device, dtype=dtype))
        self.weight_WV = torch.nn.init.trunc_normal_(self.weight_WV, std=0.02)
        
        self.weight_WO = torch.nn.Parameter(torch.empty((self.d_model, self.num_heads*self.d_v), device=device, dtype=dtype))
        self.weight_WO = torch.nn.init.trunc_normal_(self.weight_WO, std=0.02)
        
        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_l = x.shape[-2]
        
        Q = einsum(self.weight_WQ, x, "... hd_k d_in, ... s_l d_in -> ... s_l hd_k") \
            .view(*x.shape[:-2], s_l, self.num_heads, self.d_k).transpose(-3,-2)
        K = einsum(self.weight_WK, x, "... hd_k d_in, ... s_l d_in -> ... s_l hd_k") \
            .view(*x.shape[:-2], s_l, self.num_heads, self.d_k).transpose(-3,-2)
        V = einsum(self.weight_WV, x, "... hd_v d_in, ... s_l d_in -> ... s_l hd_v") \
            .view(*x.shape[:-2], s_l, self.num_heads, self.d_v).transpose(-3,-2)
        O = self.weight_WO
        
        mask = ~(torch.triu(torch.ones(s_l, s_l, device=self.device), diagonal=1).bool())
        for _ in range(len(x.shape[:-2]) + 1):
            mask = mask.unsqueeze(0)
        mask = mask.expand(*x.shape[:-2], self.num_heads, s_l, s_l)
        
        result = MyAttention(Q, K, V, mask=mask, preserve_head=True).transpose(-3, -2).reshape(*x.shape[:-2], s_l, self.num_heads* self.d_k)
            
        result = einsum(O, result, "... d_model hd_v, ... s_l hd_v -> ... s_l d_model")
        
        return result
    
class MyMultiHeadSelfAttentionWithRoPE(MyMultiHeadSelfAttention):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        
        super().__init__(d_model, num_heads, device=device, dtype=dtype)
        
        self._RoPE = MyRoPE(theta, self.d_k, max_seq_len, dtype=dtype)
        self.theta = theta
        self.max_seq_len = max_seq_len
    
    def forward(self, x: torch.Tensor, token_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        s_l = x.shape[-2]
        
        Q = einsum(self.weight_WQ, x, "... hd_k d_in, ... s_l d_in -> ... s_l hd_k") \
            .view(*x.shape[:-2], s_l, self.num_heads, self.d_k).transpose(-3,-2)
        K = einsum(self.weight_WK, x, "... hd_k d_in, ... s_l d_in -> ... s_l hd_k") \
            .view(*x.shape[:-2], s_l, self.num_heads, self.d_k).transpose(-3,-2)
        V = einsum(self.weight_WV, x, "... hd_v d_in, ... s_l d_in -> ... s_l hd_v") \
            .view(*x.shape[:-2], s_l, self.num_heads, self.d_v).transpose(-3,-2)
        O = self.weight_WO
        
        mask = ~(torch.triu(torch.ones(s_l, s_l, device=self.device), diagonal=1).bool())
        for _ in range(len(x.shape[:-2]) + 1):
            mask = mask.unsqueeze(0)
        mask = mask.expand(*x.shape[:-2], self.num_heads, s_l, s_l)
        
        if token_positions is None:
            token_positions = torch.arange(0, s_l, dtype=torch.int, device=self.device)
        
        Q = self._RoPE.forward(Q, token_positions)
        K = self._RoPE.forward(K, token_positions)
        
        result = MyAttention(Q, K, V, mask=mask, preserve_head=True).transpose(-3, -2).reshape(*x.shape[:-2], s_l, self.num_heads* self.d_k)
        
        result = einsum(O, result, "... d_model hd_v, ... s_l hd_v -> ... s_l d_model")
        
        return result

class MyPreNormTransformerBlock(torch.nn.Module):
    def __init__(self, 
                 d_model: int, num_heads: int, theta: float, max_seq_len: int,
                 d_ff: int,
                 eps_1: float = 1e-5, eps_2: float = 1e-5,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):

        super().__init__()

        self.device = device if device is not None else torch.device('cpu')
        self.dtype = dtype if dtype is not None else torch.float32

        # --- Attention with RoPE ---
        self.attn = MyMultiHeadSelfAttentionWithRoPE(
            d_model, num_heads, theta, max_seq_len, device=device, dtype=dtype
        )

        # --- SwiGLU ---
        self.ffn = MySwiGLU(d_model, d_ff, device=device, dtype=dtype)

        # --- RMSNorms ---
        self.ln1 = MyRMSNorm(d_model, eps_1, device=device, dtype=dtype)
        self.ln2 = MyRMSNorm(d_model, eps_2, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn.forward(self.ln1.forward(x))
        
        x = x + self.ffn.forward(self.ln2.forward(x))
        
        return x
    
    @property
    def max_seq_len(self) -> int:
        return self.attn.max_seq_len

class MyTransformer(torch.nn.Module):
    def __init__(self,
                 # embedding
                 num_embeddings: int, d_model: int, 
                 # transform
                 num_layers: int,
                 num_heads: int, theta: float, max_seq_len: int,
                 # SwiGLU
                 d_ff: int,
                 # RMSNorm
                 eps_1: float = 1e-5, eps_2: float = 1e-5,
                 eps_final: float = 1e-5,
                 device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        
        super().__init__()
        
        self.token_embeddings = MyEmbedding(num_embeddings, d_model, device, dtype)
        self.layers = torch.nn.ModuleList([
            MyPreNormTransformerBlock(
                d_model, num_heads, theta, max_seq_len,
                d_ff,
                eps_1, eps_2,
                device, dtype
            ) for _ in range(num_layers)
        ])
        self.ln_final = MyRMSNorm(d_model, eps_final, device, dtype)
        self.lm_head = MyLinear(d_model, num_embeddings, device, dtype)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: Int[torch.Tensor, " batch_size seq_len"]
        embed = self.token_embeddings.forward(token_ids)
        # embed: Float[torch.Tensor, " batch_size seq_len embedding_dim"]
        for layer in self.layers:
            embed = layer.forward(embed)
        
        normed = self.ln_final.forward(embed)
        logits = self.lm_head.forward(normed)
        
        # possibility = MySoftMax(logits, dim=-1)
        
        # return possibility
        return logits
    
    @property
    def max_seq_len(self) -> int:
        return self.layers[0].max_seq_len
            