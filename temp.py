import pickle

# /home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/cs336_basics/tinystories_bpe_vocab.pkl

with open('/home/nipporita/大模型/Week 1/llm-from-scratch-assignment1-basics/my_module/owt_bpe_vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)