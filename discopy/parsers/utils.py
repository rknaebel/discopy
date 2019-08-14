def get_token_list(doc_words, tokens, sent_id, sent_off):
    return [[doc_words[sent_off + t][1]['CharacterOffsetBegin'],
             doc_words[sent_off + t][1]['CharacterOffsetEnd'],
             sent_off + t, sent_id, t] for t in tokens]


def get_raw_tokens(doc_words, idxs, sent_id=None):
    if sent_id is not None:
        doc_words = [d for d in doc_words if d[1] == sent_id]
    return " ".join([doc_words[i[2]][0] for i in sorted(idxs)])


def get_raw_tokens2(doc_words, idxs, sent_id=None):
    if sent_id is not None:
        doc_words = [d for d in doc_words if d[1] == sent_id]
    # TODO fix whitespaces, depend on the character offsets
    return " ".join([doc_words[i][0][0] for i in sorted(idxs)])


def get_token_list2(doc_words, tokens, sent_id=None):
    """
    Returns the words corresponding to the global token ids.
    If sent_id is given, the tokens are handled as local ids corresponding to the sentence.
    """
    if sent_id is not None:
        offset = len([w for (w, s_i, w_i) in doc_words if s_i < sent_id])
        tokens = [i + offset for i in tokens]
    return [[int(doc_words[i][0][1]['CharacterOffsetBegin']),
             int(doc_words[i][0][1]['CharacterOffsetEnd']),
             int(i), int(doc_words[i][1]), int(doc_words[i][2])] for i in sorted(tokens)]
