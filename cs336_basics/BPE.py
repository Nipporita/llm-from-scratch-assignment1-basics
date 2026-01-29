import regex as re

import multiprocessing as mp
from tqdm import tqdm

import time
from tqdm import tqdm, trange
import heapq

import os


PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT = re.compile(PATTERN)

def split_with_special(text, special_tokens):
    if not special_tokens:
        return [text]
    pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    return [t for t in re.split(pattern, text) if t]

def read_chunk(args):
    idx, filename, start, end, special_tokens, progress_queue = args

    with open(filename, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
        # ✅ 汇报：这个 chunk 完成
    progress_queue.put(("read_done", 1))

    progress_queue.put(("part_report", len(chunk)))
    
    split_parts = split_with_special(chunk, special_tokens)

    pre_token_freq_small = {}  

    for part in split_parts:
        if part in special_tokens:
            progress_queue.put(("part_len", len(part)))
            continue
        
        # ✅ 汇报：我将处理一个 part（长度已知）

        for pattern in PAT.finditer(part):
            token = pattern.group(0)
            byte_token = tuple(token.encode("utf-8"))
            pre_token_freq_small[byte_token] = (
                pre_token_freq_small.get(byte_token, 0) + 1
            )
        
        progress_queue.put(("part_len", len(part)))
    
    progress_queue.put(("chunk_done", 1))

    return idx, pre_token_freq_small

def run_parallel(tasks, n_workers):
    manager = mp.Manager()
    progress_queue = manager.Queue()

    tasks_with_queue = [
        (*task, progress_queue) for task in tasks
    ]

    n_chunks = len(tasks)

    with mp.Pool(processes=n_workers) as pool:
        async_result = pool.map_async(read_chunk, tasks_with_queue)

        chunk_pbar = tqdm(
            total=n_chunks,
            desc="Reads",
            position=0,
            leave=True,
        )

        part_pbar = tqdm(
            total=0,                 # 动态 total
            desc="Parts",
            position=1,
            leave=True,
            mininterval=0.5,
            unit="chars",
        )
        
        chunk_done_pbar = tqdm(
            total=n_chunks,
            desc="Chunks Done",
            position=2,
            leave=True,
        )

        chunks_done = 0

        while chunks_done < n_chunks:
            tag, value = progress_queue.get()

            if tag == "read_done":
                chunk_pbar.update(value)

            elif tag == "part_report":
                # 只改 total，不 refresh
                part_pbar.total += value

            elif tag == "part_len":
                # 只 update，让 tqdm 自己决定是否 redraw
                part_pbar.update(value)
            
            elif tag == "chunk_done":
                chunks_done += value
                chunk_done_pbar.update(value)
            

        chunk_pbar.close()
        part_pbar.close()

        results = async_result.get()

    return results

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    
    now = time.time()
    
    # Pre-tokenization
    
    MAX_SPECIAL_TOKEN_LENGTH = max(len(token) for token in special_tokens) if special_tokens else 0
    
    special_tokens.sort(key=len, reverse=True)
    
    # FILE_READ_CHUNK_SIZE = max(1024, MAX_SPECIAL_TOKEN_LENGTH * 4)
    FILE_READ_CHUNK_SIZE = max(1024*32, MAX_SPECIAL_TOKEN_LENGTH * 4)
    
    pre_token_freq = {}
    
    reading_encodes = ["utf-8", "gbk"]
    
    text_chunk = ""
    
    num_processes = 4
    num_chunks = 4
    max_chunk_size = os.path.getsize(input_path) // num_chunks
    
    from cs336_basics.pretokenization_example import find_chunk_boundaries
    
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    max_size = max(boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1))
    
    while max_size > 2**30: # 1GB
        num_chunks *= 2
        with open(input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")
    
        max_size = max(boundaries[i+1] - boundaries[i] for i in range(len(boundaries)-1))
    
    tasks = [
        (idx, input_path, start, end, special_tokens)
        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
    ]
    
    num_tasks = len(tasks)
    
    for i in range(0, num_tasks, num_processes):
        
        task_chunk = tasks[i:min(i+num_processes, num_tasks)]
        
        results = run_parallel(
            task_chunk,
            len(task_chunk)
        )
        
        for idx, pre_token_freq_small in results:
            for byte_token, freq in pre_token_freq_small.items():
                if byte_token in pre_token_freq:
                    pre_token_freq[byte_token] += freq
                else:
                    pre_token_freq[byte_token] = freq
                
    # for start, end in zip(boundaries[:-1], boundaries[1:]):
    #     f.seek(start)
    
    # chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
    print(f"Pre-tokenization done in {time.time() - now:.2f} seconds.")
    now = time.time()
    
    # BPE training
    
    vocab_list = [i for i in range(256)]
    new_vocab = 0
    
    vocab = {}
    
    for i in range(256):
        vocab[new_vocab] = bytes([i])
        new_vocab += 1
    
    merges = []
    
    pair_freq = {}
    pair_occurence = {}
    
    pre_token_quasi_linked_list = {}
    
    for pre_token in pre_token_freq.keys():
        quasi_linked_list = {
            "index": [i for i in range(len(pre_token))],
            "char": [pre_token[i] for i in range(len(pre_token))],
            "next": [(i + 1) if i + 1 < len(pre_token) else None for i in range(len(pre_token))],
            "prev": [(i - 1) if i - 1 >= 0 else None for i in range(len(pre_token)) ],
            "len": len(pre_token),
        }
        pre_token_quasi_linked_list[pre_token] = quasi_linked_list
    
    # init
    for pre_token, freq in pre_token_freq.items():
        idx = 0
        next_idx = pre_token_quasi_linked_list[pre_token]["next"][idx]
        
        while next_idx is not None:
            pair = (pre_token_quasi_linked_list[pre_token]["char"][idx], pre_token_quasi_linked_list[pre_token]["char"][next_idx])
            if pair in pair_freq:
                pair_freq[pair] += freq
                pair_occurence[pair].add((pre_token, idx))
            else:
                pair_freq[pair] = freq
                pair_occurence[pair] = set()
                pair_occurence[pair].add((pre_token, idx))
            idx = next_idx
            next_idx = pre_token_quasi_linked_list[pre_token]["next"][idx]
    
    def _sort_func(x):
        return (vocab[x[1][0]], vocab[x[1][1]])
    
    def _valid(_pair):
        minus_freq, pair = _pair
        freq = - minus_freq
        
        if pair in pair_freq and pair_freq[pair] == freq:
            return True
        else:
            return False
    
    pair_freq_heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(pair_freq_heap)
    
    for _ in trange(vocab_size - len(special_tokens) - 256):
        most_frequent_pair_freq_pair = heapq.heappop(pair_freq_heap)
        while not _valid(most_frequent_pair_freq_pair):
            if pair_freq_heap:
                most_frequent_pair_freq_pair = heapq.heappop(pair_freq_heap)
            else:
                break
        if not _valid(most_frequent_pair_freq_pair):
            break
        
        most_frequent_pair_freq_pairs = [most_frequent_pair_freq_pair]
        while pair_freq_heap and pair_freq_heap[0][0] == most_frequent_pair_freq_pair[0]:
            candidate = heapq.heappop(pair_freq_heap)
            if _valid(candidate):
                most_frequent_pair_freq_pairs.append(candidate)
        
        if len(most_frequent_pair_freq_pairs) != 1:
            most_frequent_pair_freq_pair = max(most_frequent_pair_freq_pairs, key=_sort_func)
            most_frequent_pair_freq_pairs.remove(most_frequent_pair_freq_pair)
            
            most_frequent_pair_freq_pairs = set(most_frequent_pair_freq_pairs)
            for _pair in most_frequent_pair_freq_pairs:
                heapq.heappush(pair_freq_heap, _pair)
        
        most_frequent_pair = most_frequent_pair_freq_pair[1]
        
        if pair_freq[most_frequent_pair] == 0:
            break
        
        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))
        vocab[new_vocab] = bytes(vocab[most_frequent_pair[0]] + vocab[most_frequent_pair[1]])
        vocab_list.append(new_vocab)
        
        # update pre_token_quasi_linked_list, pair_freq, pair_occurence
        
        changed_pair_freq = set()
        
        most_frequent_pair_occurrences = pair_occurence[most_frequent_pair].copy()
        for occurrence in most_frequent_pair_occurrences:
            pre_token = occurrence[0]
            idx = occurrence[1]
            
            next_idx = pre_token_quasi_linked_list[pre_token]["next"][idx]
            
            if next_idx is None:
                continue
            # next_idx is not None
            next_next_idx = pre_token_quasi_linked_list[pre_token]["next"][next_idx]
            
            prev_idx = pre_token_quasi_linked_list[pre_token]["prev"][idx]
            
            # considering previous pair
            if prev_idx is not None:
                # delete old pair
                prev_pair = (pre_token_quasi_linked_list[pre_token]["char"][prev_idx], pre_token_quasi_linked_list[pre_token]["char"][idx])
                pair_freq[prev_pair] -= pre_token_freq[pre_token]
                if prev_pair not in changed_pair_freq:
                    changed_pair_freq.add(prev_pair)
                # if pair_freq[prev_pair] == 0:
                #     del pair_freq[prev_pair]
                pair_occurence[prev_pair].discard((pre_token, prev_idx))
                
                # create new pair
                new_pair = (pre_token_quasi_linked_list[pre_token]["char"][prev_idx], new_vocab)
                if new_pair in pair_freq:
                    pair_freq[new_pair] += pre_token_freq[pre_token]
                    pair_occurence[new_pair].add((pre_token, prev_idx))
                else:
                    pair_freq[new_pair] = pre_token_freq[pre_token]
                    pair_occurence[new_pair] = set()
                    pair_occurence[new_pair].add((pre_token, prev_idx))
                
                if new_pair not in changed_pair_freq:
                    changed_pair_freq.add(new_pair)
            
            # considering next pair
            if next_next_idx is not None:
                # delete old pair
                next_pair = (pre_token_quasi_linked_list[pre_token]["char"][next_idx], pre_token_quasi_linked_list[pre_token]["char"][next_next_idx])
                
                pair_freq[next_pair] -= pre_token_freq[pre_token]
                # if pair_freq[next_pair] == 0:
                #     del pair_freq[next_pair]
                if next_pair not in changed_pair_freq:
                    changed_pair_freq.add(next_pair)
                pair_occurence[next_pair].discard((pre_token, next_idx))
                # create new pair
                new_pair = (new_vocab, pre_token_quasi_linked_list[pre_token]["char"][next_next_idx])
                if new_pair in pair_freq:
                    pair_freq[new_pair] += pre_token_freq[pre_token]
                    pair_occurence[new_pair].add((pre_token, idx))
                else:
                    pair_freq[new_pair] = pre_token_freq[pre_token]
                    pair_occurence[new_pair] = set()
                    pair_occurence[new_pair].add((pre_token, idx))
                
                if new_pair not in changed_pair_freq:
                    changed_pair_freq.add(new_pair)
            
            # handling this pair
            pre_token_quasi_linked_list[pre_token]["char"][idx] = new_vocab
            
            pre_token_quasi_linked_list[pre_token]["next"][idx] = next_next_idx
            if next_next_idx is not None:
                pre_token_quasi_linked_list[pre_token]["prev"][next_next_idx] = idx
            
            pre_token_quasi_linked_list[pre_token]["next"][next_idx] = None
            pre_token_quasi_linked_list[pre_token]["prev"][next_idx] = None
            
            # pair_freq[most_frequent_pair] -= pre_token_freq[pre_token]
        
        pair_freq[most_frequent_pair] = 0
        
        for changed_pair in changed_pair_freq:
            if pair_freq[changed_pair] > 0:
                heapq.heappush(pair_freq_heap, (-pair_freq[changed_pair], changed_pair))
        
        new_vocab += 1
    
    # 算法问题
    new_vocab = {}
    for i in range(len(special_tokens)):
        new_vocab[i] = bytes(special_tokens[i].encode('utf-8'))
    for i in range(len(vocab)):
        new_vocab[i + len(special_tokens)] = vocab[i]
    
    return new_vocab, merges

if __name__ == "__main__":
    
    import pickle
    
    vocab, merges = train_bpe(
        r"/home/nipporita/大模型/Week 1/lfs-data/owt_train.txt",
        32000,
        ["<|endoftext|>"]
    )
    
    # 序列化存储
    with open("/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/cs336_basics/owt_bpe_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    with open("/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/cs336_basics/owt_bpe_merges.pkl", "wb") as f:
        pickle.dump(merges, f)
    
    # vocab, merges = train_bpe(
    #     r"/home/nipporita/大模型/Week 1/lfs-data/TinyStoriesV2-GPT4-train.txt",
    #     10000,
    #     ["<|endoftext|>"]
    # )
    
    # with open("/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/cs336_basics/tinystories_bpe_vocab.pkl", "wb") as f:
    #     pickle.dump(vocab, f)
    
    # with open("/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/cs336_basics/tinystories_bpe_merges.pkl", "wb") as f:
    #     pickle.dump(merges, f)
    
    