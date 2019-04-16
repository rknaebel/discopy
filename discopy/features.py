from nltk import ParentedTree

from discopy.utils import discourse_adverbial, coordinating_connective, subordinating_connective


def get_root_path(clause) -> str:
    path = []
    while clause.parent():
        clause = clause.parent()
        path.append(clause.label())
    return "-".join(path)


def get_clause_context(clause):
    clause_pos = clause.label()
    clause_parent_pos = clause.parent().label()
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


def get_connective_category(conn_head):
    if conn_head in discourse_adverbial:
        conn_cat = 'Adverbial'
    elif conn_head in coordinating_connective:
        conn_cat = 'Coordinating'
    elif conn_head in subordinating_connective:
        conn_cat = 'Subordinating'
    else:
        conn_cat = None
    return conn_cat


def get_relative_position(pos1, pos2):
    for i in range(min(len(pos1), len(pos2))):
        if pos1[i] < pos2[i]:
            return 'left'
        if pos1[i] > pos2[i]:
            return 'right'
    # if pos2 is contained by pos1
    return 'contains'


def height(clause):
    i = 0
    while clause.parent() and (clause.parent().label() != ''):
        clause = clause.parent()
        i += 1
    return i


def get_clause_direction_path(conn, clause):
    if height(conn) == height(clause):
        return conn.label() + 'U' + conn.parent().label() + 'D' + clause.label()
    elif height(conn) > height(clause):
        distance = height(conn) - height(clause) + 1
        p = conn.label()
        parent = conn
        while distance != 0:
            parent = parent.parent()
            p += 'U' + parent.label()
            distance -= 1
        distance = height(clause) - height(parent)
        parent = clause
        down = []
        while distance != 0:
            parent = parent.parent()
            down.append(parent.label())
            distance -= 1
        d = 'D' + clause.label()
        p += d
        return p


def get_sibling_counts(ptree: ParentedTree):
    if not ptree.parent():
        return 0, 0
    left_siblings = ptree.parent_index()
    right_siblings = len(ptree.parent()) - left_siblings - 1
    return left_siblings, right_siblings


def get_clauses(ptree):
    clauses = ((ptree[pos], pos) for pos in ptree.treepositions() if type(ptree[pos]) != str and len(pos) > 0)
    clauses = [(subtree, pos) for subtree, pos in clauses if
               subtree.height() > 2 and subtree.label() in ['VP', 'S', 'SBAR']]
    return clauses
