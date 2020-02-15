import os
import os
import ujson as json

import discopy.evaluate.exact
from discopy.data.conll16 import get_conll_dataset_extended
from discopy.data.crawled import load_corpus
from discopy.parsers.bilstm_args import NeuralExplicitArgumentExtractor, NeuralConnectiveArgumentExtractor, \
    NeuralTriConnectiveArgumentExtractor
from discopy.parsers.gosh import GoshParser
from discopy.parsers.lin import LinParser, LinArgumentParser
from discopy.semi_utils import get_arguments
from discopy.utils import init_logger

args = get_arguments()

os.makedirs(args.dir, exist_ok=True)

logger = init_logger()


# def get_additional_data(parser, additional_parses, parses_train, pdtb_train, confidence_threshold=0.70, n_take=10):
#     preds = extract_discourse_relations(parser, additional_parses)
#     confident_documents = [doc_id for doc_id, doc in preds.items()
#                            if doc['Confidence'] > confidence_threshold]
#     logger.info("Average Confidence: {}".format(np.mean([doc['Confidence'] for doc in preds.values()])))
#     logger.info("Maximum Confidence: {}".format(np.max([doc['Confidence'] for doc in preds.values()])))
#     logger.info("Found confident documents: {}".format(len(confident_documents)))
#
#     if len(confident_documents) == 0:
#         logger.info("No confident documents: take {} best".format(n_take))
#         confident_documents = [doc_id for doc_id, _ in
#                                sorted(preds.items(), key=(lambda d: d[1]['Confidence']), reverse=True)][:n_take]
#
#     parses_semi_train = copy.copy(parses_train)
#     pdtb_semi_train = copy.copy(pdtb_train)
#     for doc_id in confident_documents:
#         parses_semi_train[doc_id] = additional_parses[doc_id]
#         pdtb_semi_train.extend(preds[doc_id]['Relations'])
#
#     return parses_semi_train, pdtb_semi_train


# def bootstrap_parses(parses, n_straps=3, ratio=0.7, n_samples=None, replace=True):
#     n_samples = n_samples or int(len(parses) * ratio)
#     doc_ids = list(parses.keys())
#     straps = []
#     for i in range(n_straps):
#         strap_doc_ids = set(np.random.choice(doc_ids, size=n_samples, replace=replace))
#         strap_parses = {doc_id: doc for doc_id, doc in parses.items() if doc_id in strap_doc_ids}
#         straps.append(strap_parses)
#     return straps


# if __name__ == '__main__':
#     bbc_corpus = load_corpus(args.conll, 'bbc_corpus_featured')
#     gv_corpus = load_corpus(args.conll, 'gv_corpus_featured')
#     ted_corpus = load_corpus(args.conll, 'ted_corpus_featured')
#
#     parses_val, pdtb_val = get_conll_dataset_extended(args.conll, 'en.dev', connective_mapping=True)
#     parses_test, pdtb_test = get_conll_dataset_extended(args.conll, 'en.test', connective_mapping=True)
#     parses_train, pdtb_train = get_conll_dataset_extended(args.conll, 'en.train', connective_mapping=True)
#
#     logger.info('init parser...')
#     parser = discopy.parsers.lin.LinParser()
#     parser_path = args.dir
#
#     if args.base_dir and os.path.exists(os.path.join(args.base_dir, 'parser.joblib')):
#         logger.info('load base model from ' + args.base_dir)
#         parser = discopy.parsers.lin.LinParser.from_path(args.base_dir)
#         parser_path = args.base_dir
#     elif os.path.exists(args.dir) and os.path.exists(os.path.join(args.dir, 'parser.joblib')):
#         logger.info('load pre-trained parser...')
#         parser = discopy.parsers.lin.LinParser.from_path(args.dir)
#         parser_path = args.dir
#     else:
#         logger.info('train parser...')
#         parser.train(pdtb_train, parses_train)
#         parser.save(os.path.join(args.dir))
#
#     if not args.skip_eval:
#         logger.info('component evaluation (test)')
#         parser.score(pdtb_test, parses_test)
#
#         logger.info('extract discourse relations from test data')
#         pdtb_pred = extract_discourse_relations(parser_path, parses_test)
#         evaluate_parser(pdtb_test, pdtb_pred, threshold=0.8)
#
#     logger.info('load additional data...')
#     parses_semi = load_corpus(args.corpus)
#     logger.info("loaded documents: {}".format(len(parses_semi)))
#
#     for iteration in range(args.iters):
#         logger.info("current Iteration: {}".format(iteration))
#         parses_bootstrap = bootstrap_parses(parses_semi, n_straps=1, n_samples=args.samples, replace=False)[0]
#         logger.info("sampled documents: {}".format(len(parses_bootstrap)))
#         logger.info('get additional data for samples')
#         parses_semi_train, pdtb_semi_train = get_additional_data(parser_path, parses_bootstrap, parses_train,
#                                                                  pdtb_train,
#                                                                  confidence_threshold=args.threshold)
#         logger.info('re-train model')
#         parser.train(pdtb_semi_train, parses_semi_train)
#         parser_path = os.path.join(args.dir, str(iteration + 1))
#         parser.save(parser_path)
#         pdtb_pred = extract_discourse_relations(parser_path, parses_test)
#         evaluate_parser(pdtb_test, pdtb_pred)


def evaluate_parser(pdtb_gold, pdtb_pred, threshold=0.7):
    gold_relations = discopy.utils.load_relations(pdtb_gold)
    pred_relations = discopy.utils.load_relations([r for doc in pdtb_pred.values() for r in doc['Relations']])
    return discopy.evaluate.exact.evaluate_explicit_arguments(gold_relations, pred_relations, threshold=threshold)


parsers = {
    'lin': LinParser(),
    'lin-arg': LinArgumentParser(),
    'gosh': GoshParser(),
    'nca': NeuralConnectiveArgumentExtractor(),
    'nea': NeuralExplicitArgumentExtractor(),
    'ntca': NeuralTriConnectiveArgumentExtractor(),
    'elmo-nca': NeuralConnectiveArgumentExtractor(elmo=True),
    'elmo-nea': NeuralExplicitArgumentExtractor(elmo=True),
    # 'bilstm1': BiLSTMDiscourseParser1(),
    # 'bilstm2': BiLSTMDiscourseParser2(),
    # 'bilstm3': BiLSTMDiscourseParser3(),
}

if __name__ == '__main__':
    logger.info('Init Parser...')
    parser = parsers.get(args.parser, LinParser)
    parser_path = args.dir

    if args.train:
        parses_train, pdtb_train = get_conll_dataset_extended(args.conll, 'en.train', connective_mapping=True)
        parses_val, pdtb_val = get_conll_dataset_extended(args.conll, 'en.dev', connective_mapping=True)
        parses_test, pdtb_test = get_conll_dataset_extended(args.conll, 'en.test', connective_mapping=True)
        # parses_blind, pdtb_blind = get_conll_dataset_extended(args.conll, 'en.blind-test', connective_mapping=True)

        logger.info('Train end-to-end Parser...')
        parser.fit(pdtb_train, parses_train, pdtb_val, parses_val)
        parser.save(os.path.join(args.dir))
        logger.info('component evaluation (test)')
        parser.score(pdtb_test, parses_test)
        logger.info('extract discourse relations from test data')
        pdtb_pred = parser.parse_documents(parses_test)
        evaluate_parser(pdtb_test, pdtb_pred, threshold=args.threshold)
    elif os.path.exists(args.dir):
        logger.info('Load pre-trained Parser...')
        parser.load(args.dir)
    else:
        raise ValueError('Training and Loading not clear')

    bbc_corpus = load_corpus(args.conll + '/..', 'bbc_corpus_featured')
    # gv_corpus = load_corpus(args.conll+'/..', 'gv_corpus_featured')
    # ted_corpus = load_corpus(args.conll+'/..', 'ted_corpus_featured')

    logger.info('extract discourse relations from additional data')
    bbc_pred = parser.parse_documents(bbc_corpus)
    with open(args.out, 'w') as fh:
        fh.writelines('\n'.join(
            ['{}'.format(json.dumps(relation)) for doc_id, relations in bbc_pred.items() for relation in
             relations['Relations']]))

    # logger.info('extract discourse relations from BLIND data')
    # pdtb_pred = extract_discourse_relations(parser, parses_blind)
    # evaluate_parser(pdtb_blind, pdtb_pred, threshold=args.threshold)
