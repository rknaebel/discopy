import argparse
import os

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--dir", help="",
                             default='tmp')
argument_parser.add_argument("--conll", help="",
                             default='')
argument_parser.add_argument("--parser", help="", choices=['nca', 'nea'])
argument_parser.add_argument("--train", action='store_true', help="")
argument_parser.add_argument("--bert", action='store_true', help="")
argument_parser.add_argument("--mt", action='store_true', help="")
argument_parser.add_argument("--threshold", help="",
                             default=0.9, type=float)
argument_parser.add_argument("--gpu", help="",
                             default='0')
args = argument_parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from discopy.data.conll16 import load_conll_dataset
import discopy.parsers.bilstm_args
import discopy.parsers.bilstm_mt
from discopy.semi_utils import eval_parser

import discopy.evaluate.exact
from discopy.utils import init_logger

os.makedirs(args.dir, exist_ok=True)

logger = init_logger(os.path.join(args.dir, 'model.log'))

if __name__ == '__main__':
    logger.info('Init Parser...')
    if args.mt:
        if args.parser == 'nca':
            parser = discopy.parsers.bilstm_mt.NeuralConnectiveArgumentExtractor(hidden_size=256, rnn_size=256,
                                                                                 use_bert=args.bert)
            load_trees = True
        elif args.parser == 'nea':
            parser = discopy.parsers.bilstm_mt.NeuralExplicitArgumentExtractor(hidden_size=256, rnn_size=256,
                                                                               use_bert=args.bert)
            load_trees = False
    else:
        if args.parser == 'nca':
            parser = discopy.parsers.bilstm_args.NeuralConnectiveArgumentExtractor(hidden_size=256, rnn_size=256,
                                                                                   use_bert=args.bert)
            load_trees = True
        elif args.parser == 'nea':
            parser = discopy.parsers.bilstm_args.NeuralExplicitArgumentExtractor(hidden_size=256, rnn_size=256,
                                                                                 use_bert=args.bert)
            load_trees = False

    if args.train:
        logger.info('Load Data')
        parses_train, pdtb_train = load_conll_dataset(args.conll, 'en.train', load_trees=load_trees,
                                                      connective_mapping=True, use_bert=args.bert)
        parses_val, pdtb_val = load_conll_dataset(args.conll, 'en.dev', load_trees=load_trees,
                                                  connective_mapping=True, use_bert=args.bert)
        logger.info('Train end-to-end Parser...')
        parser.fit(pdtb_train, parses_train, pdtb_val, parses_val, epochs=50)
        if args.dir:
            parser.save(os.path.join(args.dir))
    else:
        parser.load(os.path.join(args.dir))
    parses_test, pdtb_test = load_conll_dataset(args.conll, 'en.test', load_trees=load_trees,
                                                connective_mapping=True, use_bert=args.bert)
    # parses_blind, pdtb_blind = load_conll_dataset(args.conll, 'en.blind-test', load_trees=load_trees,
    #                                                   connective_mapping=True, use_bert=args.bert)

    logger.info('extract discourse relations from test data')
    all_results = eval_parser('test', parser, parses_test, pdtb_test)
