# -*- coding: utf-8 -*-
"""
Class for generating features related to constituency parsing.
Features implemented:
1. Tree label of the target
2. Label of the lowest common ancestor of the target and the sentiment expression
"""
from sklearn.preprocessing import OneHotEncoder

from feature_utils import get_subtree_by_span, transform_spans, POS_TAGS
from abstract_features import AbstractFeatures


class ConstituencyParesFeatures(AbstractFeatures):

    def __init__(self, trees):
        self._trees = trees 

    def get_features(self, df_row):
        """Return vector of feature values.

        Args:
            df_row (pd.Series): instance as pd.Series containing at least the
                word level spans of the target (candidate) and senti expression
        """
        tree = self._trees[df_row["sentence"]]

        # transforms character spans to token spans  
        _, _, target_start_token_span, target_end_token_span =  \
            transform_spans(df_row)

        target_tree = get_subtree_by_span(tree,
                                          target_start_token_span,
                                          target_end_token_span)

        if target_tree is not None:
            # one hot encode tree labels
            enc = OneHotEncoder(handle_unknown="ignore")
            enc.fit([[i] for i in POS_TAGS])
            oh_tlabel = list(enc.transform([[target_tree.label()]]).toarray()[0])
            # lca = self._get_lowest_common_ancestor(tree, df_row) 
            # if lca is not None:
            #     oh_lcalabel = list(enc.transform([[lca.label()]]).toarray()[0]) 
            #     return oh_tlabel + oh_lcalabel
            return oh_tlabel
        
        return None
    
    # TODO: something went wrong with this!
    @staticmethod
    def _get_lowest_common_ancestor(tree, row):
        """Find phrase that connects target to sentiment expression."""
        trees = tree.subtrees(
            filter=lambda x: all([
                row["targetStart"] in range(x.span_start(), x.span_end()),
                row["targetEnd"] in range(x.span_start(), x.span_end()),

                (row["sentexprStart"] in range(x.span_start(), x.span_end()) or \
                 row["sentexprEnd"] in range(x.span_start(), x.span_end()))
            ])
        )
        trees = list(trees)
        if trees:
            return trees[-1]  # last tree in list is the lowest common node
        return None


########## For Trial ##########

def test_write_all_instances_to_file(items_df_path):
    from benepar import Parser
    import pandas as pd
    from tqdm import tqdm
    from feature_utils import parse_sent
    
    items_df = pd.read_csv(items_df_path)
    parser = Parser("benepar_en3")
    print("Parsing sentences...")
    trees = {
        sent: parse_sent(sent, parser)
        for sent in tqdm(items_df["sentence"].unique())
    }

    constituency_feature = ConstituencyParesFeatures(trees)
    with open("../output/instances/test_constituency.csv", "w") as f_out:
            for idx in range(0,len(items_df)):
                item = items_df.iloc[idx]
                print(constituency_feature.get_features(item), file=f_out)
                break

if __name__ == "__main__":
    test_write_all_instances_to_file('../output/UNSC_2014_SPV.7154_sentsplit.csv')






