from discopy.confusion_matrix import ConfusionMatrix

#
# argument extractor
#
discourse_adverbial = {'accordingly', 'additionally', 'afterwards', 'also', 'alternatively', 'as a result',
                       'as an alternative', 'as well', 'besides', 'by comparison', 'by contrast',
                       'by then', 'consequently', 'conversely', 'earlier', 'either..or', 'except', 'finally',
                       'for example', 'for instance', 'further', 'furthermore', 'hence', 'in addition',
                       'in contrast', 'in fact', 'in other words', 'in particular', 'in short', 'in sum',
                       'in the end', 'in turn', 'indeed', 'instead', 'later', 'likewise', 'meantime',
                       'meanwhile', 'moreover', 'nevertheless', 'next', 'nonetheless',
                       'on the contrary', 'on the other hand', 'otherwise', 'overall', 'previously',
                       'rather', 'regardless', 'separately', 'similarly', 'simultaneously', 'specifically',
                       'still', 'thereafter', 'thereby', 'therefore', 'thus', 'ultimately', 'whereas'
                       }

coordinating_connective = {'and', 'but', 'else', 'if then', 'neither nor', 'nor',
                           'on the one hand on the other hand', 'or', 'plus', 'then', 'yet'}

subordinating_connective = {'after', 'although', 'as', 'as if', 'as long as', 'as soon as', 'as though', 'because',
                            'before', 'before and after', 'for', 'however', 'if', 'if and when', 'insofar as',
                            'lest', 'much as', 'now that', 'once', 'since', 'so', 'so that', 'though', 'till', 'unless',
                            'until', 'when', 'when and if', 'while'}
#
# connective
#
single_connectives = {'accordingly', 'additionally', 'after', 'afterward', 'also', 'alternatively',
                      'although', 'and', 'because', 'besides', 'but', 'consequently', 'conversely', 'earlier',
                      'else', 'except', 'finally', 'further', 'furthermore', 'hence', 'however', 'indeed',
                      'instead', 'later', 'lest', 'likewise', 'meantime', 'meanwhile', 'moreover',
                      'nevertheless', 'next', 'nonetheless', 'nor', 'once', 'or', 'otherwise', 'overall',
                      'plus', 'previously', 'rather', 'regardless', 'separately', 'similarly',
                      'simultaneously', 'since', 'specifically', 'still', 'then', 'thereafter', 'thereby',
                      'therefore', 'though', 'thus', 'till', 'ultimately', 'unless', 'until', 'whereas',
                      'while', 'yet'}

multi_connectives = list(map(lambda s: s.split(' '), [
    'as a result',
    'as a result',
    'as an alternative',
    'as if',
    'as long as',
    'as soon as',
    'as though',
    'as well',
    'as',
    'before and after',
    'before',
    'by comparison',
    'by contrast',
    # 'by',
    'for example',
    'for instance',
    'for',
    'if then',
    'if and when',
    'if',
    'in addition',
    'in contrast',
    'in fact',
    'in other words',
    'in particular',
    'in short',
    'in sum',
    'in the end',
    'in turn',
    'insofar as',
    'much as',
    'now that',
    'on the contrary',
    'on the other hand',
    'so that',
    'so',
    'when and if',
    'when',
    'neither nor',
    'either or',
]))

distant_connectives = list(map(lambda s: s.split(' '), [
    'if then',
    'neither nor',
    'either or',
]))

multi_connectives_first = {'as', 'before', 'by', 'for', 'either', 'if', 'in', 'insofar', 'much', 'neither',
                           'now', 'on', 'so', 'when'}


def fscore_conf(classifier, testSet):
    conf_obj = ConfusionMatrix()
    true_label = []
    predicted_label = []
    for i in testSet:
        true_label.append(i[1])
        predicted_label.append(classifier.classify(i[0]))
    conf_obj.add_list(predicted_label, true_label)
    conf_obj.print_matrix()
    f1_score = conf_obj.compute_average_f1()
    print('F1 score = ', f1_score)
