import json
import os
import pickle
import random
from collections import Counter

import nltk
from nltk.stem.porter import *

# helper function to remove duplicates from a list but keep the order of insertion
def unique(sequence):
    seen = set()
    return [x for x in sequence if not (x in seen or seen.add(x))]

# extracts all subtrees that are contained in the spans of an argument
def extract_arg_subtrees(tree, tokens):
    return [st for st in tree.subtrees(filter=lambda t: set(t.leaves()).issubset(tokens) and t.height() > 2)]

# gets all production rules as string from all subtrees in ptree which leaves are contained in tokens
def get_production_rules(ptree, tokens):
    return [",".join([t.label()] + [tt.label() for tt in t]) for t in extract_arg_subtrees(ptree, tokens)]

# extracts all production rules as a set from all trees in ptrees which subtrees are contained in tokens
def extract_productions(ptrees, tokens):
    prod_rules = {production for ptree in ptrees for production in get_production_rules(ptree, tokens)}
    return prod_rules

# extracts dependency rules as strings from a CoNLL-style list of dependencies of a sentence
def extract_dependencies(deps, tokens):
    # unpack
    deps = deps[0]
    # build dictionary mapping dependent to all labels of sourced pointing to it
    d_deps = {}
    for dep in deps:
        label = dep[0]
        dependent = dep[1].split('-')[0]
        source = dep[2].split('-')[0]
        if source in tokens and dependent in tokens:
            d = [] 
            if dependent in d_deps:
                d = d_deps[dependent]
            d.append(label) 
            d_deps[dependent] = d
    # build set containing string representations of dependency rules
    result = set()
    for d in d_deps.keys():
        r = d + ' <-'
        for s in d_deps[d]:
            r += ' <'+s+'>'
        result.add(r)
    return result

# returns all stemmed wordpairs of the cartesian product between tokens_arg1 and tokens_arg2
def extract_wordpairs(tokens_arg1, tokens_arg2):
    stemmer = PorterStemmer()
    stems_arg1 = {stemmer.stem(tk.lower()) for tk in tokens_arg1}
    stems_arg2 = {stemmer.stem(tk.lower()) for tk in tokens_arg2}

    return {s1+'_'+s2 for s1 in stems_arg1 for s2 in stems_arg2}

# returns true if span1 is embedded inside span2
def is_span_embedded(span1, span2):
    first = True if span2[0][0] < span1[0][0] else True if span2[0][0] == span1[0][0] and span2[0][1] <= span1[0][1] else False
    last = True if span2[0][0] > span1[0][0] else True if span2[0][0] == span1[0][0] and span2[0][1] >= span1[0][1] else False
    return first and last

# returns the spans first (sentence, word) to last (sentence, word)
def get_span_from_tokens(tokenList):
    return ( (tokenList[0][3], tokenList[0][4]), (tokenList[-1][3], tokenList[-1][4]) )

# extracts all context features from a relation
def extract_context_features(i, relations, parses):
    # get previous, current and next relation and extract their spans
    curr = relations[i]    
    prev = relations[i-1] if i > 0 and relations[i-1]['DocID'] == curr['DocID'] else None
    nxt  = relations[i+1] if i < len(relations)-1 and relations[i+1]['DocID'] == curr['DocID'] else None

    curr_span_arg1 = get_span_from_tokens(curr['Arg1']['TokenList'])
    curr_span_arg2 = get_span_from_tokens(curr['Arg2']['TokenList'])

    prev_span_arg1 = get_span_from_tokens(prev['Arg1']['TokenList']) if prev else None
    prev_span_arg2 = get_span_from_tokens(prev['Arg2']['TokenList']) if prev else None

    next_span_arg1 = get_span_from_tokens(nxt['Arg1']['TokenList']) if nxt else None
    next_span_arg2 = get_span_from_tokens(nxt['Arg2']['TokenList']) if nxt else None

    feats = {}
    # Fully embedded argument:
    #    - prev embedded in curr.Arg1
    feats['feat1'] = is_span_embedded( (prev_span_arg1[0], prev_span_arg2[1]), curr_span_arg1 ) if prev else False
    #    - next embedded in curr.Arg2
    feats['feat2'] = is_span_embedded( (next_span_arg1[0], next_span_arg2[1]), curr_span_arg2 ) if nxt  else False
    #    - curr embedded in pref.Arg2
    feats['feat3'] = is_span_embedded( (curr_span_arg1[0], curr_span_arg2[1]), prev_span_arg2 ) if prev else False
    #    - curr embedded in next.Arg1
    feats['feat4'] = is_span_embedded( (curr_span_arg1[0], curr_span_arg2[1]), next_span_arg1 ) if nxt  else False

    # Shared arguments:
    #    - prev.Arg2 = curr.Arg1
    feats['feat5'] = prev_span_arg2 == curr_span_arg1 if prev else False
    #    - curr.Arg2 = next.Arg1
    feats['feat6'] = curr_span_arg2 == next_span_arg1 if nxt  else False

    return feats
    
def get_features(parse_trees, deps, all_productions, all_dep_rules):
    productions = extract_productions(parse_trees)
    dep_rules = extract_dependencies(deps)

    feat = {}
    for p in all_productions:
        feat[p] = str(p in productions)
    for r in all_dep_rules:
        feat[r] = str(r in dep_rules)

    return feat


# {'AltLex': {'Comparison.Concession',
#             'Comparison.Contrast',
#             'Contingency.Cause.Reason',
#             'Contingency.Cause.Result',
#             'Contingency.Condition',
#             'Expansion',
#             'Expansion.Conjunction',
#             'Expansion.Exception',
#             'Expansion.Instantiation',
#             'Expansion.Restatement',
#             'Temporal.Asynchronous.Precedence',
#             'Temporal.Asynchronous.Succession',
#             'Temporal.Synchrony'},
#  'EntRel': {'EntRel'},
#  'Implicit': {'Comparison',
#               'Comparison.Concession',
#               'Comparison.Contrast',
#               'Contingency',
#               'Contingency.Cause',
#               'Contingency.Cause.Reason',
#               'Contingency.Cause.Result',
#               'Contingency.Condition',
#               'Expansion',
#               'Expansion.Alternative',
#               'Expansion.Alternative.Chosen alternative',
#               'Expansion.Conjunction',
#               'Expansion.Exception',
#               'Expansion.Instantiation',
#               'Expansion.Restatement',
#               'Temporal',
#               'Temporal.Asynchronous.Precedence',
#               'Temporal.Asynchronous.Succession',
#               'Temporal.Synchrony'}}

# returns true if the relation is in one of the type 2 classes that is ignored in the paper
#       - Condition
#       - Pragmatic Condition
#       - Pragmatic Contrast
#       - Pragmatic Concession
#       - Exception
def ignore_nonexp_relation(relation):
    ignore = {'Contingency.Condition',
              'Contingency.Pragmatic Condition',
              'Comparison.Pragmatic Contrast',
              'Comparison.Pragmatic Concession',
              'Expansion.Exception'}
    return relation['Sense'][0] in ignore

# generates the featureset used to train a classifier
#
# features are: Production Rules,
#               Dependency Rules,
#               Wordpairs,
#               Contextual Features
#
def generate_pdtb_features(pdtb, parses):
    # TODO: chop up function to smaller pieces
    extracted_productions = []
    extracted_dependency_rules = []
    extracted_context_features = []
    extracted_wordpairs = []

    for i, relation in enumerate(pdtb):
        # skip explicit relations
        if relation['Type'] != 'Implicit' or ignore_nonexp_relation(relation):
            continue

        doc = relation['DocID']
        sense = relation['Sense'][0]

        sentence_ids_arg1 = {t[3] for t in relation['Arg1']['TokenList']}
        sentence_ids_arg2 = {t[3] for t in relation['Arg2']['TokenList']}

        token_list_arg1 = relation['Arg1']['TokenList']
        token_list_arg2 = relation['Arg2']['TokenList']
        
        tokens_arg1 = set(parses[doc]['sentences'][t[3]]['words'][t[4]][0] for t in token_list_arg1)
        tokens_arg2 = set(parses[doc]['sentences'][t[3]]['words'][t[4]][0] for t in token_list_arg2)

        parse_trees_arg1 = [nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentence_id]['parsetree']) for sentence_id in sentence_ids_arg1]
        parse_trees_arg2 = [nltk.ParentedTree.fromstring(parses[doc]['sentences'][sentence_id]['parsetree']) for sentence_id in sentence_ids_arg2]

        p_arg1 = extract_productions(parse_trees_arg1, tokens_arg1)
        p_arg2 = extract_productions(parse_trees_arg2, tokens_arg2)

        extracted_productions.append((p_arg1, p_arg2, sense))

        dependencies_arg1 = [parses[doc]['sentences'][sentence_id]['dependencies'] for sentence_id in sentence_ids_arg1]
        dependencies_arg2 = [parses[doc]['sentences'][sentence_id]['dependencies'] for sentence_id in sentence_ids_arg2]

        d_arg1 = extract_dependencies(dependencies_arg1, tokens_arg1)
        d_arg2 = extract_dependencies(dependencies_arg2, tokens_arg2)

        extracted_dependency_rules.append((d_arg1, d_arg2, sense))

        extracted_context_features.append(extract_context_features(i, pdtb, parses))
        extracted_wordpairs.append(extract_wordpairs(tokens_arg1, tokens_arg2))

    prod_counter = Counter(p for feature in extracted_productions for p in feature[0].union(feature[1]))
    all_productions = list(p for p, c in prod_counter.items() if c >= 5)

    dep_counter = Counter(r for feature in extracted_dependency_rules for r in feature[0].union(feature[1]))
    all_dependency_rules = list(r for r, c in dep_counter.items() if c >= 5)

    wp_counter = Counter(p for feature in extracted_wordpairs for p in feature)
    all_wordpairs = list(p for p, c in wp_counter.items() if c >= 5)

    # info about the # of features. second row is # as it should be from the paper
    print('#Production Rules:\t', len(all_productions), '\t\t11113')
    print('#Dependency Rules:\t', len(all_dependency_rules), '\t\t5031')
    print('#Word Pairs:\t\t', len(all_wordpairs), '\t105783')

    feature_set = []
    for (p_arg1, p_arg2, sense), (d_arg1, d_arg2, _), c_feats, word_pairs in zip(extracted_productions, 
                                                                                 extracted_dependency_rules, 
                                                                                 extracted_context_features,
                                                                                 extracted_wordpairs):
        feat = {}
        for p in all_productions:
            feat[p + ':1' ] = str(p in p_arg1)
            feat[p + ':2' ] = str(p in p_arg2)
            feat[p + ':12'] = str(p in p_arg1 and p in p_arg2)
        for r in all_dependency_rules:
            feat[r + ':1' ] = str(r in d_arg1)
            feat[r + ':2' ] = str(r in d_arg2)
            feat[r + ':12'] = str(r in d_arg1 and r in d_arg2)
        for wp in all_wordpairs:
            feat[wp] = str(wp in word_pairs)

        feat.update(c_feats)
        feature_set.append((feat, sense))

    return all_productions, all_dependency_rules, feature_set

def parse_rules_file(fname, max_size):
    result = {}
    f = open(fname, 'r')
    for i in range(max_size):
        line = f.readline()
        tokens = line.split()
        rule = tokens[0]
        if tokens[-1] != '':
            value = float(tokens[-1])
            if not rule in result:
                result[rule] = value
    return result

# A classifier that can be trained on NonExplicit relations and predict their senses
class NonExplicitSenseClassifier:
    def __init__(self):
        self.model = None
        self.prod_rules = set()
        self.dep_rules = set()

    # loads the model from disk
    def load(self, path):
        self.model = pickle.load(open(os.path.join(path, 'non_explicit_clf.p'), 'rb'))
        self.prod_rules = pickle.load(open(os.path.join(path, 'non_explicit_prod_rules.p'), 'rb'))
        self.dep_rules = pickle.load(open(os.path.join(path, 'non_explicit_dep_rules.p'), 'rb'))

    # save the model to disk
    def save(self, path):
        pickle.dump(self.model, open(os.path.join(path, 'non_explicit_clf.p'), 'wb'))
        pickle.dump(self.prod_rules, open(os.path.join(path, 'non_explicit_prod_rules.p'), 'wb'))
        pickle.dump(self.dep_rules, open(os.path.join(path, 'non_explicit_dep_rules.p'), 'wb'))

    def fit(self, pdtb, parses, max_iter=5):
        self.prod_rules, self.dep_rules, features = generate_pdtb_features(pdtb, parses)
        self.fit_on_features(features, max_iter=max_iter)

    def fit_on_features(self, features, max_iter=5):
        self.model = nltk.MaxentClassifier.train(features, max_iter = max_iter)
        #self.model = nltk.NaiveBayesClassifier.train(features)

    def predict(self, X):
        pass

    def get_sense(self, sents):
        features = get_features(sents, self.prod_rules, self.dep_rules)
        return [self.model.classify(features)]

if __name__ == "__main__":
    trainpdtb = [json.loads(s) for s in open('/home/himbiss/uni/sdp/conll/en.train/relations.json', 'r').readlines()]
    trainparses = json.loads(open('/home/himbiss/uni/sdp/conll/en.train/parses.json').read())
    devpdtb = [json.loads(s) for s in open('/home/himbiss/uni/sdp/conll/en.dev/relations.json', 'r').readlines()]
    devparses = json.loads(open('/home/himbiss/uni/sdp/conll/en.dev/parses.json').read())
    testpdtb = [json.loads(s) for s in open('/home/himbiss/uni/sdp/conll/en.test/relations.json', 'r').readlines()]
    testparses = json.loads(open('/home/himbiss/uni/sdp/conll/en.test/parses.json').read())

    all_productions, all_dep_rules, train_data = generate_pdtb_features(trainpdtb, trainparses)
    #all_productions, all_dep_rules, train_data = generate_pdtb_features(devpdtb, devparses)
    clf = NonExplicitSenseClassifier()
    clf.prod_rules = all_productions
    clf.dep_rules = all_dep_rules
    clf.fit_on_features(train_data)
    clf.model.show_most_informative_features()
    clf.save('/tmp')
    print('....................................................................ON TRAINING DATA..................')
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, train_data)))

    print('....................................................................ON DEVELOPMENT DATA..................')
    _, _, val_data = generate_pdtb_features(devpdtb, devparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, val_data)))

    print('....................................................................ON TEST DATA..................')
    _, _, test_data = generate_pdtb_features(testpdtb, testparses)
    print('ACCURACY {}'.format(nltk.classify.accuracy(clf.model, test_data)))
