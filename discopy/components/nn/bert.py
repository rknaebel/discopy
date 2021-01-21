from typing import List

import numpy as np

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


def get_sentence_embeddings(tokens: List[Token], tokenizer, model):
    subtokens = [tokenizer.tokenize(simple_map.get(t.surface, t.surface)) for t in tokens]
    lengths = [len(s) for s in subtokens]
    tokens_ids = tokenizer.convert_tokens_to_ids([ts for t in subtokens for ts in t])
    tokens_ids = tokenizer.build_inputs_with_special_tokens(tokens_ids)
    outputs = model(np.array([tokens_ids]))
    last_hidden_state = outputs.last_hidden_state.numpy()[0]
    embeddings = np.zeros((len(lengths), last_hidden_state.shape[-1]), np.float32)
    len_left = 1
    for i, length in enumerate(lengths):
        embeddings[i] = np.concatenate([last_hidden_state[:1],
                                        last_hidden_state[len_left:len_left + length],
                                        last_hidden_state[-1:]]).mean(axis=0)
        len_left += length
    return embeddings

# def get_doc_sentence_embeddings(sent_tokens: List[List[Token]], tokenizer, model):
#     lengths = []
#     inputs = []
#     for tokens in sent_tokens:
#         subtokens = [tokenizer.tokenize(simple_map.get(t.surface, t.surface)) for t in tokens]
#         lengths.append([len(s) for s in subtokens])
#         tokens_ids = tokenizer.convert_tokens_to_ids([ts for t in subtokens for ts in t])
#         inputs.append(tokenizer.build_inputs_with_special_tokens(tokens_ids))
#     outputs = model(np.array(inputs))
#     last_hidden_states = outputs.last_hidden_state.numpy()
#     embeddings = np.zeros((len(lengths), last_hidden_states[0].shape[-1]), np.float32)
#     e_i = 0
#     for o_i, last_hidden_state in enumerate(last_hidden_states):
#         len_left = 1
#         for i, length in enumerate(lengths[o_i]):
#             embeddings[i] = np.concatenate([last_hidden_state[:1],
#                                             last_hidden_state[len_left:len_left + length],
#                                             last_hidden_state[-1:]]).mean(axis=0)
#             if len_left + length >= len(last_hidden_state):
#                 print("ALERT", last_hidden_state.shape, len_left, lengths)
#             len_left += length
#     return embeddings
