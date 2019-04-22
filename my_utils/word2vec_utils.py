'''
GloVe embedding functions
Created June, 2017
Author: xiaodl@microsoft.com
'''

import numpy as np
import tqdm

from .tokenizer import normalize_text
from .utils import count_lines


def load_emb_vocab(path, dim=300, fast_vec_format=False):
    vocab = set()
    num_lines = count_lines(path)
    with open(path, encoding='utf-8') as f:
        line_count = 0
        for line in tqdm.tqdm(f, total=num_lines):
            if fast_vec_format and line_count == 0:
                continue
            line_count += 1
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-dim]))
            vocab.add(token)
    return vocab


def build_embedding(path, vocab, dim=300, fast_vec_format=False):
    """Support fasttext format"""
    vocab_size = len(vocab)
    emb = np.zeros((vocab_size, dim))
    emb[0] = 0
    num_lines = count_lines(path)
    with open(path, encoding='utf-8') as f:
        line_count = 0
        for line in tqdm.tqdm(f, total=num_lines):
            if fast_vec_format and line_count == 0:
                items = [int(i) for i in line.split()]
                assert len(items) == 2
                assert items[1] == dim
                continue
            line_count += 1
            elems = line.split()
            token = normalize_text(' '.join(elems[0:-dim]))
            if token in vocab:
                emb[vocab[token]] = [float(v) for v in elems[-dim:]]
    return emb
