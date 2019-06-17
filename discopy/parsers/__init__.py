from .bilstm import BiLSTMDiscourseParser1, BiLSTMDiscourseParser2, BiLSTMDiscourseParser3
from .gosh import GoshParser
from .lin import LinParser


def get_token_list(doc_words, tokens, sent_id, sent_off):
    return [[doc_words[sent_off + t][1]['CharacterOffsetBegin'],
             doc_words[sent_off + t][1]['CharacterOffsetEnd'],
             sent_off + t, sent_id, t] for t in tokens]


def get_raw_tokens(doc_words, idxs):
    return " ".join([doc_words[i[2]][0] for i in idxs])
