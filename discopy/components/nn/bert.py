from typing import List

import numpy as np
import torch

from discopy.data.token import Token

simple_map = {
    "''": '"',
    "``": '"',
    "-LRB-": "(",
    "-RRB-": ")",
    "-LCB-": "{",
    "-RCB-": "}",
    "n't": "not"
}


def get_sentence_embeddings(tokens: List[Token], tokenizer, model, device='cpu'):
    subtokens = [tokenizer.tokenize(simple_map.get(t.surface, t.surface)) for t in tokens]
    lengths = [len(s) for s in subtokens]
    tokens_ids = tokenizer.convert_tokens_to_ids([ts for t in subtokens for ts in t])
    tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)
    # tokens_pt = torch.tensor([tokens_ids]).to(device)
    outputs, pooled = model(np.array([tokens_ids]))
    embeddings = np.zeros((len(lengths), outputs.shape[-1]), np.float32)
    len_left = 1
    outputs = outputs.cpu().numpy()[0]
    for i, length in enumerate(lengths):
        embeddings[i] = outputs[len_left:len_left + length].mean(axis=0)
    return outputs
