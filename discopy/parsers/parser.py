from tqdm import tqdm


class AbstractBaseParser(object):

    def fit(self, pdtb, parses, pdtb_val, parses_val):
        raise NotImplementedError()

    def score(self, pdtb, parses):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def parse_documents(self, documents, limit=0):
        doc_relations = {}
        for idx, (doc_id, doc) in tqdm(enumerate(documents.items()), total=len(documents)):
            if limit and idx > limit:
                break
            parsed_relations = self.parse_doc(doc)
            for p in parsed_relations:
                p['DocID'] = doc_id
                p['ID'] = hash(p['Connective']['RawText'] + p['Arg1']['RawText'] + p['Arg2']['RawText'])
            doc_relations[doc_id] = {
                'DocID': doc_id,
                'Relations': parsed_relations,
            }
        return doc_relations

    def parse_doc(self, doc):
        raise NotImplementedError()
