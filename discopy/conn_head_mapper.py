class ConnHeadMapper(object):

    def __init__(self):
        self.mapping = ConnHeadMapper.DEFAULT_MAPPING

    def map_raw_connective(self, raw_connective):
        raw_connective = raw_connective.strip().lower()
        head_connective = self.mapping.get(raw_connective, raw_connective)
        # find the index of the head connectives
        raw_connective_token_list = raw_connective.split(' ')
        head_connective_token_list = head_connective.lower().split(' ')
        start_point = 0
        indices = []
        for head_connective_token in head_connective_token_list:
            for i in range(start_point, len(raw_connective_token_list)):
                token = raw_connective_token_list[i]
                if head_connective_token == token or \
                        head_connective_token == self.mapping.get(token, token):
                    indices.append(i)
                    start_point = i + 1
                    break
        assert (len(head_connective_token_list) == len(indices)), (
            raw_connective_token_list, head_connective_token_list, indices)
        return head_connective, indices

    DEFAULT_MAPPING = {
        '18 months after': 'after',
        '25 years after': 'after',
        '29 years and 11 months to the day after': 'after',
        'a few hours after': 'after',
        'about six months before': 'before',
        'almost simultaneously': 'simultaneously',
        'as much as': 'much as',
        'at least not when': 'when',
        'at least when': 'when',
        'back when': 'when',
        'eight months after': 'after',
        'even after': 'after',
        'even as': 'as',
        'even before': 'before',
        'even if': 'if',
        'even still': 'still',
        'even then': 'then',
        'even though': 'though',
        'even when': 'when',
        'even while': 'while',
        'ever since': 'since',
        'five minutes before': 'before',
        'if only': 'if',
        'in the first 25 minutes after': 'after',
        'in the meantime': 'meantime',
        'in the meanwhile': 'meanwhile',
        'just after': 'after',
        'just because': 'because',
        'just five months after': 'after',
        'just when': 'when',
        'largely as a result': 'as a result',
        'later on': 'later',
        'months after': 'after',
        'nearly two months after': 'after',
        'one day after': 'after',
        'only when': 'when',
        'partly because': 'because',
        'perhaps because': 'because',
        'primarily because': 'because',
        'shortly after': 'after',
        'shortly afterwards': 'afterwards',
        'shortly before': 'before',
        'shortly thereafter': 'thereafter',
        'soon after': 'after',
        'three months after': 'after',
        'two weeks after': 'after',
        'within minutes after': 'after',
        'a day after': 'after',
        'a day or two before': 'before',
        'a decade before': 'before',
        'a few months after': 'after',
        'a few weeks after': 'after',
        'a full five minutes before': 'before',
        'a month after': 'after',
        'a week after': 'after',
        'a week before': 'before',
        'a year after': 'after',
        'about a week after': 'after',
        'about three weeks after': 'after',
        'almost before': 'before',
        'almost immediately after': 'after',
        'an average of six months before': 'before',
        'apparently because': 'because',
        'at least partly because': 'because',
        'at least until': 'until',
        'especially after': 'after',
        'especially as': 'as',
        'especially because': 'because',
        'especially if': 'if',
        'especially since': 'since',
        'especially when': 'when',
        'except when': 'when',
        'five years after': 'after',
        'four days after': 'after',
        'fully eight months before': 'before',
        'immediately after': 'after',
        'in large part because': 'because',
        'in part because': 'because',
        'in the 3 1/2 years before': 'before',
        'just 15 days after': 'after',
        'just a day after': 'after',
        'just a month after': 'after',
        'just as': 'as',
        'just as soon as': 'as soon as',
        'just before': 'before',
        'just days before': 'before',
        'just eight days before': 'before',
        'just minutes after': 'after',
        'just until': 'until',
        'largely because': 'because',
        'less than a month after': 'after',
        'long after': 'after',
        'long before': 'before',
        'mainly because': 'because',
        'merely because': 'because',
        'minutes after': 'after',
        'more than a year after': 'after',
        'nearly a year and a half after': 'after',
        'not because': 'because',
        'not only because': 'because',
        'only after': 'after',
        'only as long as': 'as long as',
        'only because': 'because',
        'only if': 'if',
        'only three years after': 'after',
        'only two weeks after': 'after',
        'only until': 'until',
        'particularly after': 'after',
        'particularly as': 'as',
        'particularly because': 'because',
        'particularly if': 'if',
        'particularly since': 'since',
        'particularly when': 'when',
        'presumably because': 'because',
        'reportedly after': 'after',
        'right after': 'after',
        'seven years after': 'after',
        'several months before': 'before',
        'shortly afterward': 'afterward',
        'simply because': 'because',
        'since before': 'before',
        'six years after': 'after',
        'so much as': 'much as',
        'some time after': 'after',
        'sometimes after': 'after',
        'two days after': 'after',
        'two days before': 'before',
        'two months before': 'before',
        'two years before': 'before',
        'typically, if': 'if',
        'usually when': 'when',
        'within a year after': 'after',
        'years after': 'after',
        'years before': 'before',
        'days after': 'after',
        'in start contrast': 'in contrast',
        'hours before': 'before',
        'three years later': 'later'
    }


if __name__ == '__main__':
    chm = ConnHeadMapper()

    raw_connective = "29 years and 11 months to the day after"
    head_connective, indices = chm.map_raw_connective(raw_connective)
    assert (head_connective == "after")
    assert (indices == [8])

    raw_connective = "Largely as a result"
    head_connective, indices = chm.map_raw_connective(raw_connective)
    assert (head_connective == "as a result")
    assert (indices == [1, 2, 3])
