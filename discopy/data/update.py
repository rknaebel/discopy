import os
from typing import List

from discopy.data.doc import ParsedDocument
from discopy.data.sentence import ParsedSentence, DepRel


def get_constituent_parse(constituent_parser, inputs):
    try:
        ptree = constituent_parser.predict([inputs], prob=False, verbose=False).trees[0]
        if ptree is not None:
            parsetree = ptree._pformat_flat('', '()', False)
        else:
            print("empty tree")
            parsetree = ""
    except Exception as e:
        print(e)
        parsetree = ""
    return parsetree


def get_dependency_parse(dependency_parser, inputs, words):
    try:
        deps = dependency_parser.predict([inputs], prob=False, verbose=False).sentences[0]
        dependencies = [
            DepRel(rel=rel.lower(),
                   head=words[head - 1] if rel.lower() != 'root' else None,
                   dep=words[dep]
                   ) for dep, (head, rel) in enumerate(zip(deps.arcs, deps.rels))
        ]
    except Exception as e:
        print(e)
        dependencies = []
    return dependencies


def update_dataset_parses(docs: List[ParsedDocument], constituent_parser='crf-con-en',
                          dependency_parser='biaffine-dep-en'):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import supar
    cparser = supar.Parser.load(constituent_parser) if constituent_parser else None
    dparser = supar.Parser.load(dependency_parser) if dependency_parser else None
    for doc in docs:
        for sent_i, sent in enumerate(doc.sentences):
            inputs = [(t.surface, t.tag) for t in sent.tokens]
            parsetree = get_constituent_parse(cparser, inputs) if cparser else None
            dependencies = get_dependency_parse(dparser, inputs, sent.tokens) if dparser else None
            doc.sentences[sent_i] = ParsedSentence(
                tokens=sent.tokens,
                dependencies=dependencies if dependency_parser else sent.dependencies,
                parsetree=parsetree if constituent_parser else sent.parsetree)
