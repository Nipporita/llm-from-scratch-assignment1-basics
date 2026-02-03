from typing import Optional, Iterable, Iterator

import time
import regex as re
import os
import tqdm
import multiprocessing as mp
import numpy as np

import torch

from my_module.train import load_meta, create_model, MyLoadCheckpoint
from my_module.Tokenizer import Tokenizer

def load_model(
    meta_path: str,
    checkpoint_path: str,
    device: str = torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
):
    meta = load_meta(meta_path)
    model = create_model(meta, device)
    _, _ = MyLoadCheckpoint(checkpoint_path, model, device=device)
    return model, meta

def get_Tokenizer(vocab_path: str, merges_path: str, special_tokens: Optional[list[bytes]] = None) -> Tokenizer:
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=special_tokens)
    return tokenizer

class MyNaiveGPT:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Tokenizer,
        device: str = torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
        seed: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        if seed is None:
            seed = int(time.time())
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        self._seed = seed
        
        self._eof = tokenizer.encode("<|endoftext|>")[0]
        
        self.max_content_length = self.model.max_seq_len
    
    @classmethod
    def from_paths(
        cls,
        meta_path: str,
        checkpoint_path: str,
        vocab_path: str,
        merges_path: str,
        special_tokens: Optional[list[bytes]] = None,
        device: str = torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
    ):
        model, _ = load_model(meta_path, checkpoint_path, device)
        tokenizer = get_Tokenizer(vocab_path, merges_path, special_tokens=special_tokens)
        return cls(model, tokenizer, device)
    
    def _get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(input_ids)
        return logits
    
    def _softmax_with_temperature(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature < 0:
            raise ValueError("Temperature must be positive.")
        elif temperature == 0:
            logits_max = torch.max(logits, dim=-1, keepdim=True).values
            probs = torch.zeros_like(logits)
            probs[logits == logits_max] = 1.0
            probs /= torch.sum(probs, dim=-1, keepdim=True)
        else:
            scaled_logits = logits / temperature
            exp_logits = torch.exp(scaled_logits - torch.max(scaled_logits, dim=-1, keepdim=True).values)
            probs = exp_logits / torch.sum(exp_logits, dim=-1, keepdim=True)
        return probs
    
    def _top_p_sample(self, probs, top_p=0.9):
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        if cumulative_probs[-1] < top_p:
            cutoff = cumulative_probs.numel() - 1
        else:
            cutoff = torch.searchsorted(cumulative_probs, top_p)

        filtered_probs = sorted_probs[:cutoff + 1]
        filtered_indices = sorted_indices[:cutoff + 1]

        filtered_probs = filtered_probs / filtered_probs.sum()

        next_idx = torch.multinomial(filtered_probs, 1)
        return filtered_indices[next_idx].item()
    
    def _next_token(
        self,
        input_ids: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> int:
        logits = self._get_logits(input_ids)
        last_logits = logits[:, -1, :]
        probs = self._softmax_with_temperature(last_logits, temperature=temperature).squeeze(0)
        next_token = self._top_p_sample(probs, top_p=top_p)
        return next_token
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> str:
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        generated_ids = input_ids.copy()
        
        progress = tqdm.trange(max_new_tokens, desc="Generating", unit="token")
        for _ in progress:
            next_token = self._next_token(input_tensor, temperature=temperature, top_p=top_p)
            if next_token == self._eof:
                progress.write("End-of-Text token generated. Stopping generation.")
                break
            generated_ids.append(next_token)
            
            if len(generated_ids) > self.max_content_length:
                input_tensor = torch.tensor([generated_ids[-self.max_content_length:]], dtype=torch.long).to(self.device)
            else:
                input_tensor = torch.tensor([generated_ids], dtype=torch.long).to(self.device)
        
        generated_text = self.tokenizer.decode(generated_ids)
        return generated_text

if __name__ == "__main__":
    meta_path = "/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_model/meta.pkl"
    model_path = "/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_model/checkpoint.pth"
    vocab_path = "/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_module/owt_bpe_vocab.pkl"
    merge_path = "/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_module/owt_bpe_merges.pkl"
    
    special_tokens = [b"<|endoftext|>"]
    
    my_gpt = MyNaiveGPT.from_paths(
        meta_path=meta_path,
        checkpoint_path=model_path,
        vocab_path=vocab_path,
        merges_path=merge_path,
        special_tokens=special_tokens,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    test_prompt = "I love you."
    generated_text = my_gpt.generate(
        prompt=test_prompt,
        max_new_tokens=1000,
        temperature=0.8,
        top_p=0.9,
    )
    print(generated_text)