from nltk import ParentedTree

from discopy.utils import discourse_adverbial, coordinating_connective, subordinating_connective


def get_root_path(clause) -> str:
    path = []
    while clause.parent():
        clause = clause.parent()
        path.append(clause.label())
    return "-".join(path)


def get_clause_context(clause) -> str:
    clause_pos = clause.label()
    clause_parent_pos = clause.parent().label() if clause.parent() else 'NULL'
    if clause.left_sibling():
        clause_ls_pos = clause.left_sibling().label()
    else:
        clause_ls_pos = 'NULL'
    if clause.right_sibling():
        clause_rs_pos = clause.right_sibling().label()
    else:
        clause_rs_pos = 'NULL'
    clause_context = clause_pos + '-' + clause_parent_pos + '-' + clause_ls_pos + '-' + clause_rs_pos
    return clause_context


def get_connective_category(conn_head) -> str:
    if conn_head in discourse_adverbial:
        conn_cat = 'Adverbial'
    elif conn_head in coordinating_connective:
        conn_cat = 'Coordinating'
    elif conn_head in subordinating_connective:
        conn_cat = 'Subordinating'
    else:
        conn_cat = 'NULL'
    return conn_cat


def get_relative_position(pos1, pos2):
    for i in range(min(len(pos1), len(pos2))):
        if pos1[i] < pos2[i]:
            return 'left'
        if pos1[i] > pos2[i]:
            return 'right'
    # if pos2 is contained by pos1
    return 'contains'


def height(clause) -> int:
    i = 0
    while clause.parent() and (clause.parent().label() != ''):
        clause = clause.parent()
        i += 1
    return i


def get_clause_direction_path(conn, clause) -> str:
    if height(conn) == height(conn.root()):
        return str(conn.label())
    if height(conn) == height(clause):
        return str(conn.label() + 'U' + conn.parent().label() + 'D' + clause.label())
    elif height(conn) > height(clause):
        distance = height(conn) - height(clause) + 1
        p = conn.label()
        parent = conn
        while distance != 0 and parent.parent():
            parent = parent.parent()
            p += 'U' + parent.label()
            distance -= 1
        distance = height(clause) - height(parent)
        parent = clause
        down = []
        while distance != 0 and parent.parent():
            parent = parent.parent()
            down.append(parent.label())
            distance -= 1
        d = 'D' + clause.label()
        p += d
        return str(p)


def get_sibling_counts(ptree: ParentedTree) -> (int, int):
    if not ptree.parent():
        return 0, 0
    left_siblings = ptree.parent_index()
    right_siblings = len(ptree.parent()) - left_siblings - 1
    return left_siblings, right_siblings


def get_clauses(ptree):
    clauses = ((ptree[pos], pos) for pos in ptree.treepositions() if type(ptree[pos]) != str)
    clauses = [(subtree, pos) for subtree, pos in clauses if
               subtree.height() > 1 and subtree.label() in ['VP', 'S', 'SBAR']]
    return clauses


def get_connective_sentence_position(indices, ptree):
    length = len(ptree.leaves())
    m1 = length * (1 / 3)
    m2 = length * (2 / 3)
    if indices[len(indices) // 2] < m1:
        pos = 'START'
    elif m1 <= indices[len(indices) // 2] < m2:
        pos = 'MIDDLE'
    else:
        pos = 'END'
    return pos


def lca(ptree, leaf_index):
    lca_loc = ptree.treeposition_spanning_leaves(leaf_index[0], leaf_index[-1] + 1)
    if type(ptree[lca_loc]) == str:
        lca_loc = lca_loc[:-1]
    return lca_loc


def get_pos_features(ptree, leaf_index, head, position):
    pl = ptree.pos()
    other_position = leaf_index[0] + position
    lca_loc = lca(ptree, leaf_index)
    conn_tag = ptree[lca_loc].label()

    if other_position >= 0:
        word, pos = pl[other_position]
    else:
        word = pos = 'NONE'
    word_head = "{},{}".format(word, head)
    pos_conn_tag = "{},{}".format(pos, conn_tag)

    return word, word_head, pos, pos_conn_tag


def get_index_tree(ptree):
    tree = ptree.copy(deep=True)
    for idx, _ in enumerate(tree.leaves()):
        tree_location = tree.leaf_treeposition(idx)
        tree[tree_location[:-1]][0] = idx
    return tree
