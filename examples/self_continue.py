import os
import ujson as json
from glob import glob

from discopy.data.conll16 import get_conll_dataset_extended
from discopy.data.crawled import load_corpus
from discopy.parsers.bilstm_args import NeuralExplicitArgumentExtractor
from discopy.semi_utils import get_arguments, get_relation_distances, eval_parser, combine_data
from discopy.utils import init_logger

if __name__ == '__main__':
    args = get_arguments()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logger = init_logger(path=os.path.join(args.dir, 'continue.log'))

    logger.info('Init Parser...')
    nea_parser = NeuralExplicitArgumentExtractor(hidden_size=256, rnn_size=512, window_length=args.window_size)
    parser_path = args.dir

    parses_train, pdtb_train = get_conll_dataset_extended(args.conll, 'en.train', connective_mapping=True)
    parses_val, pdtb_val = get_conll_dataset_extended(args.conll, 'en.dev', connective_mapping=True)
    parses_test, pdtb_test = get_conll_dataset_extended(args.conll, 'en.test', connective_mapping=True)
    parses_blind, pdtb_blind = get_conll_dataset_extended(args.conll, 'en.blind-test', connective_mapping=True)

    if args.train:
        logger.info('Train end-to-end Parser...')
        model_path = os.path.join(args.dir, 'nea-baseline')
        nea_parser.fit(pdtb_train, parses_train, pdtb_val, parses_val, path=model_path, epochs=25)
        # try:
        #     nea_parser.save(os.path.join(args.dir, 'nea-baseline'))
        # except Exception:
        #     print('Error during saving new parser.')
    else:
        nea_parser.load(os.path.join(args.dir, 'nea-baseline'))

    logger.info(str(get_relation_distances(pdtb_train)))
    eval_parser('test', nea_parser, parses_test, pdtb_test)
    eval_parser('blind', nea_parser, parses_blind, pdtb_blind)

    bbc_corpus = load_corpus(args.conll + '/..', 'bbc_corpus_featured')
    bbc_sport_corpus = load_corpus(args.conll + '/..', 'bbcsport_corpus_featured')
    # gv_corpus = load_corpus(args.conll+'/..', 'gv_corpus_featured')
    # ted_corpus = load_corpus(args.conll+'/..', 'ted_corpus_featured')

    logger.info('load discourse relations from additional data')
    bbc_path = sorted(glob(os.path.join(args.dir, "bbc.*")))[-1]
    logger.info('LOAD relations from {}'.format(bbc_path))
    with open(bbc_path, 'r') as fh:
        bbc_relations = ([json.loads(line) for line in fh])
    bbc_sport_path = sorted(glob(os.path.join(args.dir, "bbc-sport.*")))[-1]
    logger.info('LOAD relations from {}'.format(bbc_sport_path))
    with open(bbc_sport_path, 'r') as fh:
        bbc_sport_relations = ([json.loads(line) for line in fh])

    logger.info(str(get_relation_distances(bbc_relations)))
    logger.info(str(get_relation_distances(bbc_sport_relations)))

    parses_all, pdtb_all = combine_data(
        [parses_train],
        [pdtb_train, pdtb_train]
    )
    parses_noisy, pdtb_noisy = combine_data(
        [bbc_corpus, bbc_sport_corpus],
        [bbc_relations, bbc_sport_relations]
    )
    model_path = os.path.join(args.dir, 'nea-more')
    nea_parser.fit_noisy(pdtb_all, parses_all, pdtb_val, parses_val, pdtb_noisy, parses_noisy, path=model_path,
                         epochs=25)
    # try:
    #     nea_parser.save(os.path.join(args.dir, 'nea-more'))
    # except Exception:
    #     print('Error during saving tri parser.')

    eval_parser('test', nea_parser, parses_test, pdtb_test)
    eval_parser('blind', nea_parser, parses_blind, pdtb_blind)
