import copy
import os
import ujson as json

import discopy.evaluate.exact
import numpy as np
from discopy.parser import DiscourseParser
from discopy.semi_utils import args
from discopy.semi_utils import extract_discourse_relations, evaluate_parser, load_corpus
from discopy.utils import init_logger

os.makedirs(args.dir, exist_ok=True)

logger = init_logger(os.path.join(args.dir, 'self.log'))


def get_additional_data(parser, additional_parses, parses_train, pdtb_train, confidence_threshold=0.70):
    preds = extract_discourse_relations(parser, additional_parses)
    confident_documents = [doc_id for doc_id, doc in preds.items()
                           if doc['Confidence'] > confidence_threshold]
    logger.info("Average Confidence: {}".format(np.mean([doc['Confidence'] for doc in preds.values()])))
    logger.info("Maximum Confidence: {}".format(np.max([doc['Confidence'] for doc in preds.values()])))
    logger.info("Found confident documents: {}".format(len(confident_documents)))

    if len(confident_documents) == 0:
        logger.info("No confident documents: take five best")
        confident_documents = [doc_id for doc_id, _ in
                               sorted(preds.items(), key=(lambda d: d[1]['Confidence']), reverse=True)][:5]

    parses_semi_train = copy.copy(parses_train)
    pdtb_semi_train = copy.copy(pdtb_train)
    for doc_id in confident_documents:
        parses_semi_train[doc_id] = additional_parses[doc_id]
        pdtb_semi_train.extend(preds[doc_id]['Relations'])

    return parses_semi_train, pdtb_semi_train


def bootstrap_parses(parses, n_straps=3, ratio=0.7, n_samples=None, replace=True):
    n_samples = n_samples or int(len(parses) * ratio)
    doc_ids = list(parses.keys())
    straps = []
    for i in range(n_straps):
        strap_doc_ids = set(np.random.choice(doc_ids, size=n_samples, replace=replace))
        strap_parses = {doc_id: doc for doc_id, doc in parses.items() if doc_id in strap_doc_ids}
        straps.append(strap_parses)
    return straps


if __name__ == '__main__':
    pdtb_train = [json.loads(s) for s in open(os.path.join(args.conll, 'en.train/relations.json'), 'r')]
    parses_train = json.loads(open(os.path.join(args.conll, 'en.train/parses.json'), 'r').read())

    pdtb_test = [json.loads(s) for s in open(os.path.join(args.conll, 'en.test/relations.json'), 'r')]
    parses_test = json.loads(open(os.path.join(args.conll, 'en.test/parses.json'), 'r').read())

    logger.info('init parser...')
    parser = discopy.parser.DiscourseParser(n_estimators=args.estimators)
    parser_path = args.dir

    if args.base_dir and os.path.exists(os.path.join(args.base_dir, 'parser.joblib')):
        logger.info('load base model from ' + args.base_dir)
        parser = discopy.parser.DiscourseParser.from_path(args.base_dir)
        parser_path = args.base_dir
    elif os.path.exists(args.dir) and os.path.exists(os.path.join(args.dir, 'parser.joblib')):
        logger.info('load pre-trained parser...')
        parser = discopy.parser.DiscourseParser.from_path(args.dir)
        parser_path = args.dir
    else:
        logger.info('train parser...')
        parser.train(pdtb_train, parses_train)
        parser.save(os.path.join(args.dir))

    if not args.skip_eval:
        logger.info('component evaluation (test)')
        parser.score(pdtb_test, parses_test)

        logger.info('extract discourse relations from test data')
        pdtb_pred = extract_discourse_relations(parser_path, parses_test)
        evaluate_parser(pdtb_test, pdtb_pred, threshold=0.8)

    logger.info('load additional data...')
    parses_semi = load_corpus(args.corpus)
    logger.info("loaded documents: {}".format(len(parses_semi)))

    for iteration in range(args.iters):
        logger.info("current Iteration: {}".format(iteration))
        parses_bootstrap = bootstrap_parses(parses_semi, n_straps=1, n_samples=args.samples, replace=False)[0]
        logger.info("sampled documents: {}".format(len(parses_bootstrap)))
        logger.info('get additional data for samples')
        parses_semi_train, pdtb_semi_train = get_additional_data(parser_path, parses_bootstrap, parses_train,
                                                                 pdtb_train,
                                                                 confidence_threshold=args.threshold)
        logger.info('re-train model')
        parser.train(pdtb_semi_train, parses_semi_train)
        parser_path = os.path.join(args.dir, str(iteration + 1))
        parser.save(parser_path)
        pdtb_pred = extract_discourse_relations(parser_path, parses_test)
        evaluate_parser(pdtb_test, pdtb_pred)
