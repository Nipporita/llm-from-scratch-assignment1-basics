from typing import Optional, Iterable, Iterator

import time
import regex as re
import os
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

def split_with_special(text, special_tokens):
    if not special_tokens:
        return [text]
    pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    return [t for t in re.split(pattern, text) if t]

def apply_merges(word_bytes, merges_set, vocab_to_id):
    word_bytes = list(word_bytes)
    while True:
        min_token_id = float('inf')
        best_pair_idx = -1
        merged = None
        
        for i in range(len(word_bytes) - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            if pair in merges_set:
                combined = pair[0] + pair[1]
                token_id = vocab_to_id.get(combined)
                if token_id is not None and token_id < min_token_id:
                    min_token_id = token_id
                    best_pair_idx = i
                    merged = combined
                
        if best_pair_idx == -1:
            break
        # Apply best merge
        word_bytes = (word_bytes[:best_pair_idx]
                      + [merged]
        + word_bytes[best_pair_idx + 2:])
        
    return tuple(word_bytes)

class Tokenizer:
    PRE_TOKENIZE_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=Optional[list[bytes]]):
        self.vocab = vocab
        self.rev_vocab = {v: k for k, v in vocab.items()}
        self.merges = set(merges)
        
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens.sort(key=len, reverse=True)
        
        self.special_token_strs = [st.decode("utf-8") for st in self.special_tokens]
        
    @property
    def _max_special_token_length(self) -> int:
        if not self.special_tokens:
            return 0
        return max(len(st) for st in self.special_tokens)
    
    @classmethod
    def from_files(cls, vocab_path: str, merges_path: str, special_tokens: Optional[list[bytes]] = None):
        """
        从 pickle 文件加载 vocab 和 merges，并构造 Tokenizer
        此部分由于没有调用，并且是__init__级别函数，直接用GPT生成了
        """
        import pickle
        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_path, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """
        将输入文本编码为 token id 列表
        """
        # pre tokenize, assume that len(text) is small enough to fit in memory
        
        pre_texts = split_with_special(text, self.special_token_strs)
        
        pre_tokens = []
        for text_part in pre_texts:
            if text_part in self.special_token_strs:
                pre_tokens.append(text_part)
            else:
                pre_tokens += self.PRE_TOKENIZE_PATTERN.findall(text_part)
        
        return self._encode(pre_tokens)
    
    def _encode(self, pre_tokens: list[str]) -> list[int]:
        token_ids = []
        
        for pre_token in pre_tokens:
            if pre_token.encode("utf-8") in self.special_tokens:
                token_ids.append(self.rev_vocab[pre_token.encode("utf-8")])
                continue
            # apply bpe merges
            
            pre_token_bytes = [bytes([x]) for x in pre_token.encode("utf-8")]
            
            pre_token_bytes_ = apply_merges(pre_token_bytes, self.merges, self.rev_vocab)
            
            # for merge in self.merges:
            #     i = 0
            #     while i < len(pre_token_bytes) - 1:
            #         if pre_token_bytes[i] == merge[0] and pre_token_bytes[i + 1] == merge[1]:
            #             pre_token_bytes.pop(i + 1)
            #             pre_token_bytes[i] = merge[0] + merge[1]
            #         else:
            #             i += 1
            
            for byte_seq in pre_token_bytes_:
                token_ids.append(self.rev_vocab[byte_seq])
                
        return token_ids
    
    def decode(self, ids: list[int]) -> str:
        """
        将 token id 列表解码为文本
        感觉没有挑战这一块，直接copilot写了
        """
        byte_seq = b"".join([self.vocab[token_id] for token_id in ids])
        return byte_seq.decode("utf-8", errors="ignore")
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Iterable for char
        
        import tqdm
        
        # char 进度条
        tq = tqdm.tqdm(unit=" tokens", desc="Encoding")

        read_str = ""
        read_str_list = []
        max_special_token_length = self._max_special_token_length
        
        CHUNK_TARGET_SIZE = 1024 * 32
        CHUNK_READ_SIZE = CHUNK_TARGET_SIZE + max_special_token_length
        chunk_size = 0
        
        previous_text = None
        
        str_iter = iter(iterable)

        # read_str = iterable.read()
        
        while True:
            read_str_list = [read_str]
            for text in str_iter:
                if chunk_size < CHUNK_READ_SIZE:
                    read_str_list.append(text)
                    chunk_size += 1
                else:
                    read_str = "".join(read_str_list)
                    read_str_list = []
                    break
            else:
                read_str = "".join(read_str_list)
                break
            
            # 处理上一个 chunk
            
            text_chunk = split_with_special(read_str, self.special_token_strs)
            
            if len(text_chunk) > 1:
                length = 0
                idx = None
                for i in range(len(text_chunk) - 1):
                    if length + len(text_chunk[i]) >= CHUNK_TARGET_SIZE:
                        idx = i
                        break
                    length += len(text_chunk[i])
                for chunk in text_chunk[:idx+1]:
                    tokens = self.encode(chunk)
                    for token in tokens:
                        tq.update(1)
                        yield token
                        
                read_str = "".join(text_chunk[idx+1:])
                chunk_size = len(read_str)
            else:
                # 没有 special token
                
                length = 0
                idx = None
                
                pre_tokens = self.PRE_TOKENIZE_PATTERN.findall(read_str)
                
                for i in range(len(pre_tokens) - 1):
                    if length + len(pre_tokens[i]) >= CHUNK_TARGET_SIZE:
                        idx = i
                        break
                    length += len(pre_tokens[i])
                
                pre_token_chunks = pre_tokens[:idx+1]
                tokens = self._encode(pre_token_chunks)
                for token in tokens:
                    tq.update(1)
                    yield token
                
                read_str = "".join(pre_tokens[idx+1:])
                chunk_size = len(read_str)
        
        # #处理最后的遗留
        
        # pre_texts = split_with_special(read_str, self.special_token_strs)
        
        # for text_part in pre_texts:
        #     if text_part in self.special_token_strs:
        #         tq.update(1)
        #         yield self.rev_vocab[text_part.encode("utf-8")]
        #     else:
        #         for token in self.PRE_TOKENIZE_PATTERN.findall(text_part):
        #             for token_id in self._encode([token]):
        #                 tq.update(1)
        #                 yield token_id
        
        
        tokens = self.encode(read_str)
        for token in tokens:
            tq.update(1)
            yield token
            
        # raise NotImplementedError

# 全局变量（每个子进程各一份）
_TOKENIZER = None

def init_worker(vocab_path, merges_path, special_tokens):
    global _TOKENIZER
    _TOKENIZER = Tokenizer.from_files(
        vocab_path,
        merges_path,
        special_tokens
    )

def TokenizeChunk(args):
    idx, input_path, start, end, save_path = args
    
    with open(input_path, "rb") as f:
        f.seek(start)
        data = f.read(end - start)

    text = data.decode("utf-8", errors="ignore")
    token_ids = _TOKENIZER.encode(text)

    output_path = save_path + f".part{idx}"
    
    arr = np.memmap(
        output_path,
        dtype=np.uint16,
        mode="w+",
        shape=(len(token_ids),),
    )
    arr[:] = np.array(token_ids, dtype=np.uint16)[:]
    arr.flush()
    
    return idx, None

def TokenizeData(input_path: str,
                 output_path: str = "/root/autodl-tmp/",
                 vocab_path: str = "/root/llm-from-scratch-assignment1-basics-main/my_module/owt_bpe_vocab.pkl",
                 merges_path: str = "/root/llm-from-scratch-assignment1-basics-main/my_module/owt_bpe_merges.pkl",
                 special_tokens: Optional[list[str]] = None,
                 ):
    num_workers = min(16, os.cpu_count())
    num_chunks = num_workers * 32   # 关键：chunk 数要明显多于 worker

    from cs336_basics.pretokenization_example import find_chunk_boundaries

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f, num_chunks, b"<|endoftext|>"
        )
    
    # np.memmap 存储
    input_filename = os.path.basename(input_path)
    pure_name, _ = os.path.splitext(input_filename)
    save_path = os.path.join(output_path, f"{pure_name}")

    max_size = max(
        boundaries[i+1] - boundaries[i]
        for i in range(len(boundaries)-1)
    )

    while max_size > 2**30:  # 单 chunk 不超过 1GB
        num_chunks *= 2
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(
                f, num_chunks, b"<|endoftext|>"
            )
        max_size = max(
            boundaries[i+1] - boundaries[i]
            for i in range(len(boundaries)-1)
        )

    tasks = [
        (idx, input_path, start, end, save_path)
        for idx, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        )
    ]

    results = [None] * len(tasks)

    ctx = mp.get_context("spawn")  # 强烈建议，避免 fork 坑

    with ctx.Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            vocab_path,
            merges_path,
            special_tokens,
        ),
    ) as pool:
        for idx, token_ids in tqdm(
            pool.imap_unordered(TokenizeChunk, tasks),
            total=len(tasks),
        ):
            results[idx] = token_ids

    # 合并结果
    t = time.time()
    shape = 0
    
    output_filename = f"{pure_name}.npy"
    
    for idx in range(len(results)):
        part_path = save_path + f".part{idx}"
        part_arr = np.memmap(
            part_path,
            dtype=np.uint16,
            mode="r",
        )
        shape += part_arr.shape[0]
        part_arr._mmap.close()
    
    arr = np.memmap(
        os.path.join(output_path, output_filename),
        dtype=np.uint16,
        mode="w+",
        shape=shape,
    )
    
    start_idx = 0
    
    for idx in range(len(results)):
        part_path = save_path + f".part{idx}"
        part_arr = np.memmap(
            part_path,
            dtype=np.uint16,
            mode="r",
        )
        part_size = part_arr.shape[0]
        arr[start_idx:start_idx + part_size] = part_arr[:]
        start_idx += part_size
        part_arr._mmap.close()
    
    arr.flush()
    
    for idx in range(len(results)):
        part_path = save_path + f".part{idx}"
        os.remove(part_path)

    print(f"Tokenization completed in {time.time() - t:.2f} seconds.")

if __name__ == "__main__":
    # temp = "brea   amjioap ! jewpo ! amopjio  ! a mnpjip iojajiopm i12345678"
    # a = Tokenizer(None, None, None)
    # a.encode_iterable(temp)
    
    TokenizeData(
        "/home/nipporita/大模型/Week 1/lfs-data/owt_valid.txt",
        "/home/nipporita/大模型/Week 1/lfs-data",
        "/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_module/owt_bpe_vocab.pkl",
        "/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_module/owt_bpe_merges.pkl",
        special_tokens=[
            b"<|endoftext|>",
        ],
    )
    
    # TokenizeData(
    #     "/root/assignment1-data/owt_train.txt",
    #     special_tokens=[
    #         b"<|endoftext|>",
    #     ],
    # )
    pass