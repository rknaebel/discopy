from typing import List

import spacy
from discopy.data.doc import Document
from discopy.data.relation import Relation
from discopy.data.sentence import Sentence
from discopy.data.token import Token
from discopy.data.update import update_dataset_parses


def load_texts(texts: List[str], simple_tags=False, nlp=None) -> List[Document]:
    nlp = nlp or spacy.load('en')
    docs = []
    for raw_text in texts:
        words = []
        token_offset = 0
        sentences = []
        lines = [line.strip() for line in raw_text.splitlines(keepends=True)
                 if line.strip() and len(line.split(' ')) > 1]
        for doc in nlp.pipe(lines):
            offset = 0
            sents = list(doc.sents)
            for sent_i, sent in enumerate(sents):
                sent_words = [
                    Token(token_offset + w_i, sent_i, w_i, t.idx + offset, t.idx + len(t.text) + offset,
                          t.text,
                          t.pos_ if simple_tags else t.tag_)
                    for w_i, t in enumerate(sent)
                ]
                words.extend(sent_words)
                offset += len(sent.string)
                token_offset += len(sent_words)
                sentences.append(ParsedSentence(sent_words))
        docs.append(ParsedDocument(doc_id=hash(raw_text), sentences=sentences, relations=[]))
    update_dataset_parses(docs)
    return docs
