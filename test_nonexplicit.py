import nltk
import unittest
import discopy.nonexplicit

class TestNonexplicit(unittest.TestCase):

    def test_mutual_information(self):
        """
            Tests the correct calculation of the MutualInformation score.
            Example taken from: https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html
        """
        clf = discopy.nonexplicit.NonExplicitSenseClassifier()

        terms = {'export', 'apple'}
        classes = {'poultry', 'pig'}

        n_11 = [('export', 'poultry')] * 49
        n_01 = [('apple', 'poultry')] * 141
        n_10 = [('export', 'pig')] * 27652
        n_00 = [('apple', 'pig')] * 774106
        n = n_11 + n_01 + n_10 + n_00
        rels = [[[a]] for (a, b) in n]
        senses = [b for (a, b) in n]

        _, mis = clf.select_by_mutual_information(rels, senses, terms, 2)
        self.assertEqual(mis['export'], 0.00011053558610110263)

    def test_production_rules(self):
        """
            Tests the correct extraction of production rules from a tree.
              - Rules should contain the node label followed by an arrow 
                and then a list of all first children.
        """
        tstring = """
                (ROOT
                  (S
                    (S
                      (NP
                        (NP (DT The) (JJS strongest) (NN rain))
                        (VP
                          (ADVP (RB ever))
                          (VBN recorded)
                          (PP (IN in)
                            (NP (NNP India)))))
                      (VP
                        (VP (VBD shut)
                          (PRT (RP down))
                          (NP
                            (NP (DT the) (JJ financial) (NN hub))
                            (PP (IN of)
                              (NP (NNP Mumbai)))))
                        (, ,)
                        (VP (VBD snapped)
                          (NP (NN communication) (NNS lines)))
                        (, ,)
                        (VP (VBD closed)
                          (NP (NNS airports)))
                        (CC and)
                        (VP (VBD forced)
                          (NP
                            (NP (NNS thousands))
                            (PP (IN of)
                              (NP (NNS people))))
                          (S
                            (VP (TO to)
                              (VP
                                (VP (VB sleep)
                                  (PP (IN in)
                                    (NP (PRP$ their) (NNS offices))))
                                (CC or)
                                (VP (VB walk)
                                  (NP (NN home))
                                  (PP (IN during)
                                    (NP (DT the) (NN night))))))))))
                    (, ,)
                    (NP (NNS officials))
                    (VP (VBD said)
                      (NP-TMP (NN today)))
                    (. .)))"""
        
        tree = nltk.ParentedTree.fromstring(tstring)
        feat = discopy.nonexplicit.ProductionRuleFeature([], [])

        tokens = ['The', 'strongest', 'rain', 'ever', 'recorded', 'in', 'India', 'shut', 'down', 'the', 'financial', 'hub', 'of', 'Mumbai', ',', 'snapped', 'communication', 'lines', ',', 'closed', 'airports', 'and', 'forced', 'thousands', 'of', 'people', 'to', 'sleep', 'in', 'their', 'offices', 'or', 'walk', 'home', 'during', 'the', 'night', ',', 'officials', 'said', 'today', '.'] 

        tokens_arg1 = tokens[:7] # The strongest rain ever recorded in India
        tokens_arg2 = tokens[23:37] # forced thousands of people to sleep in their offices or walk home during the night
        
        correct_arg1 = {'RB -> ever', 'VBN -> recorded', 'PP -> IN NP', 'NN -> rain', 'NP -> NNP', 'NP -> DT JJS NN', 'IN -> in', 'VP -> ADVP VBN PP', 'DT -> The', 'ADVP -> RB', 'NNP -> India', 'JJS -> strongest', 'NP -> NP VP'}

        correct_arg2 = {'NP -> NP PP', 'NN -> home', 'NP -> NNS', 'VP -> VP CC VP', 'CC -> or', 'NNS -> thousands', 'DT -> the', 'VB -> sleep', 'VB -> walk', 'NP -> NN', 'NP -> PRP$ NNS', 'NNS -> people', 'PP -> IN NP', 'VP -> VB PP', 'VP -> TO VP', 'IN -> in', 'NN -> night', 'PRP$ -> their', 'IN -> during', 'IN -> of', 'TO -> to', 'VP -> VB NP PP', 'S -> VP', 'NNS -> offices', 'NP -> DT NN'}
        correct_both = {'VP -> VB PP', 'NN -> home', 'PRT -> RP', 'NN -> today', 'NP -> NP VP', 'CC -> and', 'NNS -> thousands', 'VBD -> snapped', 'JJ -> financial', 'NNS -> people', 'PRP$ -> their', 'VP -> VBD NP S', 'NP -> PRP$ NNS', 'NNP -> India', 'IN -> during', 'S -> S , NP VP .', 'VBN -> recorded', 'VP -> VB NP PP', 'NNP -> Mumbai', 'NNS -> offices', 'NP -> NP PP', 'NP-TMP -> NN', 'VP -> VBD PRT NP', 'NNS -> airports', 'NNS -> officials', 'VBD -> shut', 'S -> VP', 'JJS -> strongest', 'NP -> NN NNS', 'NP -> NNS', 'NN -> rain', 'VP -> VBD NP', ', -> ,', 'VB -> sleep', 'VBD -> said', 'DT -> The', 'NN -> communication', 'VP -> VP CC VP', 'S -> NP VP', 'PP -> IN NP', 'ROOT -> S', 'NP -> DT JJ NN', 'VBD -> forced', 'TO -> to', 'VP -> VBD NP-TMP', 'NP -> NNP', 'RP -> down', 'RB -> ever', 'VBD -> closed', 'NP -> NN', 'VB -> walk', 'NN -> night', 'NNS -> lines', 'IN -> of', '. -> .', 'NP -> DT NN', 'NN -> hub', 'ADVP -> RB', 'VP -> ADVP VBN PP', 'VP -> VP , VP , VP CC VP', 'VP -> TO VP', 'NP -> DT JJS NN', 'CC -> or', 'IN -> in', 'DT -> the'}

        self.assertEqual(feat.extract_productions([tree], tokens_arg1), correct_arg1)
        self.assertEqual(feat.extract_productions([tree], tokens_arg2), correct_arg2)
        self.assertEqual(feat.extract_productions([tree], tokens),      correct_both)

    """
        Tests the correct extraction of dependency rules from a CoNLL-style dependency parse.
          - Rules start with the dependent followed by an '<-' arrow and then a list of
            all dependency labels that point towards the dependent enclosed in brackets.
    """
    def test_dependency_rules(self):
        dependencies = [ ['nsubj',  'had-0',      'We-1'],
                         ['dobj',   'had-0',      'problems-3'],
                         ['det',    'problems-3', 'no-2'],
                         ['nn',     'problems-3', 'operating-4'],
                         ['advmod', 'problems-3', 'at-5'],
                         ['dep',    'at-5',       'all-6'] ]

        tokens_arg1 = ['We', 'had', 'no', 'operating', 'problems']
        tokens_arg2 = ['at', 'all']

        feat = discopy.nonexplicit.DependencyRuleFeature([], [])
        rules_arg1 = feat.extract_dependencies([dependencies], tokens_arg1)
        rules_arg2 = feat.extract_dependencies([dependencies], tokens_arg2)
        rules_both = feat.extract_dependencies([dependencies], tokens_arg1 + tokens_arg2)

        correct_arg1 = { 'had <- <nsubj> <dobj>',
                         'problems <- <det> <nn>' }
        correct_arg2 = { 'at <- <dep>' }
        correct_both = { 'had <- <nsubj> <dobj>',
                         'problems <- <det> <nn> <advmod>',
                         'at <- <dep>' }

        self.assertEqual(rules_both, correct_both)
        self.assertEqual(rules_arg1, correct_arg1) 
        self.assertEqual(rules_arg2, correct_arg2) 

if __name__ == "__main__":
    unittest.main()
