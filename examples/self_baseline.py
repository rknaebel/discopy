import os
import ujson as json

from discopy.data.conll16 import get_conll_dataset_extended
from discopy.data.crawled import load_corpus
from discopy.parsers.tri_parser import NeuralTriConnectiveArgumentExtractor
from discopy.semi_utils import get_arguments, get_relation_distances, eval_parser, combine_data
from discopy.utils import init_logger

if __name__ == '__main__':
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    MODE = 'majority'

    os.makedirs(args.dir, exist_ok=True)
    logger = init_logger(path=os.path.join(args.dir, 'baseline.log'))

    logger.info('Init Parser...')
    tri_parser = NeuralTriConnectiveArgumentExtractor(hidden_sizes=(256, 512, 512),
                                                      rnn_sizes=(512, 256, 512),
                                                      window_length=args.window_size)
    parser_path = args.dir

    parses_train, pdtb_train = get_conll_dataset_extended(args.conll, 'en.train', connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset_extended(args.conll, 'en.dev', connective_mapping=True)
    parses_test, pdtb_test = get_conll_dataset_extended(args.conll, 'en.test', connective_mapping=True)
    parses_blind, pdtb_blind = get_conll_dataset_extended(args.conll, 'en.blind-test', connective_mapping=True)

    parses_all, pdtb_all = parses_train, pdtb_train
    if args.train:
        logger.info('Train end-to-end Parser...')
        tri_parser.fit(pdtb_all, parses_all, pdtb_val, parses_val, epochs=10, bootstrap=True)
        try:
            tri_parser.save(os.path.join(args.dir, 'tri-0'))
        except Exception:
            print('Error during saving tri parser.')

    else:
        tri_parser.load(os.path.join(args.dir, 'tri-0'))

    logger.info('DISTRIBUTION info for pdtb train')
    logger.info(str(get_relation_distances(pdtb_train)))
    eval_parser('test', tri_parser, parses_test, pdtb_test)

    bbc_corpus = load_corpus(args.conll + '/..', 'bbc_corpus_featured')
    bbc_sport_corpus = load_corpus(args.conll + '/..', 'bbcsport_corpus_featured')
    # gv_corpus = load_corpus(args.conll+'/..', 'gv_corpus_featured')
    # ted_corpus = load_corpus(args.conll+'/..', 'ted_corpus_featured')

    logger.info('EXTRACT discourse relations from additional data')
    bbc_relations = tri_parser.extract_document_relations(bbc_corpus, mode=MODE)
    bbc_sport_relations = tri_parser.extract_document_relations(bbc_sport_corpus, mode=MODE)

    logger.info('DISTRIBUTION info for bbc')
    logger.info(str(get_relation_distances(bbc_relations)))
    logger.info('DISTRIBUTION info for bbc-sports')
    logger.info(str(get_relation_distances(bbc_sport_relations)))

    with open(os.path.join(args.dir, 'bbc.0.rel.json'), 'w') as fh:
        for relation in bbc_relations:
            fh.write('{}\n'.format(json.dumps(relation)))
    with open(os.path.join(args.dir, 'bbc-sport.0.rel.json'), 'w') as fh:
        for relation in bbc_sport_relations:
            fh.write('{}\n'.format(json.dumps(relation)))

    for it in range(10):
        parses_all, pdtb_all = combine_data(
            [parses_train, parses_blind, bbc_corpus, bbc_sport_corpus],
            [pdtb_train, pdtb_train, pdtb_blind, pdtb_blind, bbc_relations, bbc_sport_relations]
        )
        if args.train:
            tri_parser.fit(pdtb_all, parses_all, pdtb_val, parses_val, epochs=10, bootstrap=True, init_model=False)
            try:
                tri_parser.save(os.path.join(args.dir, 'tri-{}'.format(it + 1)))
            except Exception:
                print('Error during saving tri parser.')
        else:
            tri_parser.load(os.path.join(args.dir, 'tri-{}'.format(it + 1)))

        eval_parser('test', tri_parser, parses_test, pdtb_test)

        logger.info('EXTRACT discourse relations from additional data')
        bbc_relations = tri_parser.extract_document_relations(bbc_corpus, mode=MODE)
        bbc_sport_relations = tri_parser.extract_document_relations(bbc_sport_corpus, mode=MODE)
        logger.info(str(get_relation_distances(bbc_relations)))
        logger.info(str(get_relation_distances(bbc_sport_relations)))

        with open(os.path.join(args.dir, 'bbc.{}.rel.json'.format(it + 1)), 'w') as fh:
            for relation in bbc_relations:
                fh.write('{}\n'.format(json.dumps(relation)))
        with open(os.path.join(args.dir, 'bbc-sport.{}.rel.json'.format(it + 1)), 'w') as fh:
            for relation in bbc_sport_relations:
                fh.write('{}\n'.format(json.dumps(relation)))
