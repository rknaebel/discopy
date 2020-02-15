import sys
import ujson as json
from glob import glob


# from discopy.semi_utils import get_arguments


def print_rel(r):
    print('|= {} relation ({})\n  Arg1: {}\n        {}\n  Arg2: {}\n        {}\n  Conn: {}\n        {}\n'.format(
        r['Type'], r['DocID'],
        r['Arg1']['RawText'], [i[2] for i in r['Arg1']['TokenList']],
        r['Arg2']['RawText'], [i[2] for i in r['Arg2']['TokenList']],
        r['Connective']['RawText'], [i[2] for i in r['Connective']['TokenList']]))


if __name__ == '__main__':
    # args = get_arguments()
    bbc_path = sorted(glob(sys.argv[1]))[-1]
    with open(bbc_path, 'r') as fh:
        bbc_relations = ([json.loads(line) for line in fh])

    for r in bbc_relations:
        # token_idxs = [i[2] for p in [r['Arg1'], r['Arg2'], r['Connective']] for i in p['TokenList']]
        # token_span = 1 + max(token_idxs) - min(token_idxs)
        # span_min = min(token_idxs)
        # table_items = [
        #     [span_min + i for i in range(token_span)],
        #     ['' for i in range(token_span)],
        #     ['None' for i in range(token_span)],
        # ]
        # print(tabulate(table_items, headers='first_row'))
        print_rel(r)
