# -*- coding: utf-8 -*-
"""
Some helpful classes, functions and lists.
"""

from nltk.tree import ParentedTree, Tree
from nltk.tokenize import word_tokenize


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
 
POS_TAGS = ["NP", "VP", "PP", "S", "TOP", "NML", "ADJP", "SBAR", "WHNP", "PRN",
            "PRT", "NNP", "NN", "DT", "VBN", "NNS", "PRP", "VBD", "VBG", "VB",
            "FW", "VBP", "SINV", "SQ", "PRP$", "NNPS", "JJ", "VBZ", "RB", "VBZ"]


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
            subtree._span_end = x + len(subtree.leaves()) - 1
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


def create_training_data(df, parser):
    cpf = ConstituencyParseFeatures()
    dpf = DependencyParseFeatures()
    X, y = [], []
    for row in df.iterrows():
        row = row[1]
        for k, v in transform_spans(row).items():
            row[k] = v
        tree = self.parse_sent(row["sentence"])
        tree.add_spans()
        candidates = get_candidates(row["sentence"], parser, tree)
        for c in candidates:
            instance = row.copy()
            if (c.span_start() == row["targetStart"]) and (c.span_end() == row["targetEnd"]):
                y.append(1)
            else:
                y.append(0)
            instance["targetStart"] = c.span_start()
            instance["targetEnd"] = c.span_end()
            c_f = cpf.get_features(instance)
            d_f = dpf.get_features(row)
            X.append(c_f + d_f)
    return X, y


def get_subtree_by_span(tree, span_start, span_end):
    trees = list(
        tree.subtrees(
            filter=lambda x: (x.span_start() == span_start) and \
                             (x.span_end() == span_end)
        )
    )
    if trees:
        return trees[-1]
    else:
        return None


def transform_spans(df_row):
    return {
        "sentexprStart": len(word_tokenize(df_row["sentence"][:df_row["sentexprStart"]])),
        "sentexprEnd": len(word_tokenize(df_row["sentence"][:df_row["sentexprEnd"]])) - 1,
        "targetStart": len(word_tokenize(df_row["sentence"][:df_row["targetStart"]])),
        "targetEnd": len(word_tokenize(df_row["sentence"][:df_row["targetEnd"]])) - 1
    }


def parse_sent(sent, parser):
    tree = IndexableSpannotatableParentedTree.convert(parser.parse(sent))
    tree.add_indices()
    tree.add_spans()
    return tree


def get_candidates(sent, tree):
    return [
        subtree for subtree in list(tree.subtrees())[1:]
        if subtree.label() in {"NP", "S"}
    ]
