# -*- coding: utf-8 -*-
"""
Class for generating features related to constituency parsing
"""

from sklearn.preprocessing import OneHotEncoder

from .feature_utils import get_subtree_by_span, POS_TAGS
from .abstract_features import AbstractFeatures


class ConstituencyParseFeatures(AbstractFeatures):

    def get_features(self, df_row, tree):
        """Return vector of feature values.

        Features:
            - target tree label
            - label of lowest common ancestor of target and senti expression

        Args:
            df_row (pd.Series): instance as pd.Series containing at least the
                word level spans of the target (candidate) and senti expression
            tree (IndexableSpannotatableParentedTree): could probably be easier
        """
        target_tree = get_subtree_by_span(tree,
                                          df_row["targetStart"],
                                          df_row["targetEnd"])
        if target_tree is not None:
            # one hot encode tree labels
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit([[i] for i in POS_TAGS])
            oh_tlabel = list(enc.transform([[target_tree.label()]]).toarray()[0])
            lca = self._get_lowest_common_ancestor(tree, df_row)
            if lca is not None:
                oh_lcalabel = list(enc.transform([[lca.label()]]).toarray()[0])
                return oh_tlabel + oh_lcalabel
        return None

    @staticmethod
    def _get_lowest_common_ancestor(tree, row):
        """Find phrase that connects target to sentiment expression."""
        trees = tree.subtrees(
            filter=lambda x: all([
                row["targetStart"] in range(x.span_start(), x.span_end() + 1),
                row["targetEnd"] in range(x.span_start(), x.span_end() + 1),
                (row["sentexprStart"] in range(x.span_start(), x.span_end() + 1) or \
                 row["sentexprEnd"] in range(x.span_start(), x.span_end() + 1))
            ])
        )
        trees = list(trees)
        if trees:
            return trees[-1]  # last tree in list is lowest common node
        return None
