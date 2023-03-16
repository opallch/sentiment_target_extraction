# -*- coding: utf-8 -*-
"""
Some helpful classes, functions and lists for feature engineering.
"""
import spacy
import pandas as pd
from spacy import displacy
from nltk.tree import ParentedTree, Tree
from nltk.tokenize import word_tokenize

#### Exceptions ####
class NotATargetRelationError(Exception):
    pass


class SpansError(Exception):
    pass


class TargetConstituencySubtreeNotFoundError(Exception):
    pass


#### for dependency features ####

DEP_TAGS = [
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "agent",
    "amod",
    "appos",
    "attr",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "clf",
    "complm",
    "compound",
    "conj",
    "cop",
    "csubj",
    "csubjpass",
    "dative",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "dobj",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "hmod",
    "hyph",
    "infmod",
    "intj",
    "iobj",
    "list",
    "mark",
    "meta",
    "neg",
    "nmod",
    "nn",
    "npadvmod",
    "nsubj",
    "nsubjpass",
    "nounmod",
    "npmod",
    "num",
    "number",
    "nummod",
    "oprd",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "partmod",
    "pcomp",
    "pobj",
    "poss",
    "possessive",
    "preconj",
    "prep",
    "prt",
    "punct",
    "quantmod",
    "rcmod",
    "relcl",
    "reparandum",
    "root",
    "ROOT",
    "vocative",
    "xcomp"
    ]
FINE_GRAINED_POS_TAGS = [
    ".",
    ",",
    "-LRB-",
    "-RRB-",
    "``",
    '""',
    "''",
    ":",
    "$",
    "#",
    "AFX",
    "CC",
    "CD",
    "DT",
    "EX",
    "FW",
    "HYPH",
    "IN",
    "JJ",
    "JJR",
    "JJS",
    "LS",
    "MD",
    "NIL",
    "NN",
    "NNP",
    "NNPS",
    "NNS",
    "PDT",
    "POS",
    "PRP",
    "PRP$",
    "RB",
    "RBR",
    "RBS",
    "RP",
    "TO",
    "UH",
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",
    "WDT",
    "WP",
    "WP$",
    "WRB",
    "SP",
    "ADD",
    "NFP",
    "GW",
    "XX",
    "BES",
    "HVS",
    "_SP",
    ]

def lowest_common_ancestor(node1, node2, root):
    """returns the lowest ancestor between 2 nodes."""
    if root is None:
        return None
    
    if node1 == root or node2 == root:
        return root

    if list(root.children) is None or len(list(root.children)) == 0:
        return None

    else:
        result = []
        for child in list(root.children):
            result.append(
                lowest_common_ancestor(node1, node2, child)
                )
        
        result = [elem for elem in result if elem is not None]

        if len(result) == 2:
            return root 
        
        elif len(result) == 1:
            return result[0]
        
        else:
            return None

def distance_btw_3_pts(node1, node2, ancestor):
    """
    It returns the distance between node 1 and node 2 via their common ancestor.
    If ancestor is None, it indicates that node1 and node2 are in two different sentences
    which, we assume, are connected together with a parent node.
    For calculating the distance in this case, please refer to `_distance_btw_child_ancestor()` 
    and the `README` in `/features/`.
    """
    return  _distance_btw_child_ancestor(node1, ancestor) + \
            _distance_btw_child_ancestor(node2, ancestor)
            

def _distance_btw_child_ancestor(child, ancestor):
    """
    A help function of `distance_btw_3_pts()` which returns the distance between a child and its ancestor.
    If ancestor is None, the distance is the distance between the child and its root +1.
    The reason for adding one is that we assume the corresponding sentence is connected to an external node
    which binds another sentence.
    For more information, please refer to `_distance_btw_child_ancestor()` 
    and the `README` in `/features/`.
    """
    try:
        if child == ancestor:
            return 0
        
        else:
            current_child = child
            current_parent = current_child.head
            steps = 1

            if ancestor is None: # go along way to the root + 1
                while current_child.dep_ != 'ROOT':
                    steps += 1
                    current_child = current_parent
                    current_parent = current_parent.head
                steps += 1
                    
            else:
                while current_parent != ancestor:
                    steps += 1
                    current_child = current_parent
                    current_parent = current_parent.head
                
            return steps
    
    except AttributeError:
        return -1

#### for constituency features ####

POS_TAGS = [
    "NP", 
    "VP", 
    "PP", 
    "S", 
    "TOP", 
    "NML", 
    "ADJP", 
    "SBAR", 
    "WHNP", 
    "PRN",
    "PRT", 
    "NNP", 
    "NN", 
    "DT", 
    "VBN", 
    "NNS", 
    "PRP", 
    "VBD", 
    "VBG", 
    "VB",
    "FW", 
    "VBP", 
    "SINV", 
    "SQ", 
    "PRP$", 
    "NNPS", 
    "JJ", 
    "VBZ", 
    "RB", 
    "VBZ"
    ]

class InvalidFilenameError(Exception):
    pass


class IndexableSpannotatableParentedTree(ParentedTree):

    def __init__(self, node, children=None):
        self._id = 0
        self._span_start = None
        self._span_end = None
        super().__init__(node, children)

    def add_spans(self):
        self._span_start = 0
        x = 0
        for subtree in self.subtrees():
            subtree._span_start = x
            subtree._span_end = x + len(subtree.leaves())
            if subtree.height() == 2:
                x += 1

    def add_indices(self):
        i = 1
        for child in self.subtrees():
            if isinstance(child, Tree):
                child._id = i
                i += 1

    def node_id(self):
        return self._id

    def span_start(self):
        return self._span_start

    def span_end(self):
        return self._span_end


def get_subtree_by_span(tree, span_start, span_end):
    subtrees = []

    for tree in tree.subtrees():
        if tree.span_start() == span_start and tree.span_end() == span_end:
            subtrees.append(tree)

    if subtrees:
        return subtrees[-1]
    
    return None


def char_span_to_token_span(df_row, tokenize_func=word_tokenize):
    return (
        len(tokenize_func(df_row["sentence"][:df_row["sentexprStart"]])),
        len(tokenize_func(df_row["sentence"][:df_row["sentexprEnd"]])),
        len(tokenize_func(df_row["sentence"][:df_row["targetStart"]])),
        len(tokenize_func(df_row["sentence"][:df_row["targetEnd"]]))
    )


def token_span_to_char_span(df_row, token_span_start, token_span_end, tokenize_func=word_tokenize):
    tokens = tokenize_func(df_row["sentence"])
    tmp_str = ''
    for i in range(0, token_span_start):
        tmp_str +=  tokens[i]
        tmp_str += ' '
    char_span_start = len(tmp_str)

    for i in range(token_span_start, token_span_end):
        tmp_str += tokens[i]
        tmp_str += ' '
    char_span_end = len(tmp_str) - 1 # to exclude the last space

    return char_span_start, char_span_end


def parse_sent(sent, parser):
    tree = IndexableSpannotatableParentedTree.convert(parser.parse(sent))
    tree.add_indices()
    tree.add_spans()
    return tree


def get_candidates(tree):
    return [
        subtree for subtree in list(tree.subtrees())[1:]
        if subtree.label() in {"NP", "S"}
    ]