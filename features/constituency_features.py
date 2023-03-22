# -*- coding: utf-8 -*-
"""
Class for generating features related to constituency parsing.
Features implemented:
1. Tree label of the target
2. Label of the lowest common ancestor of the target and the sentiment expression
"""
import os

from sklearn.preprocessing import OneHotEncoder

from features.feature_utils import get_subtree_by_span, char_span_to_token_span, POS_TAGS, NotATargetRelationError
from features.abstract_features import AbstractFeatures


class ConstituencyParseFeatures(AbstractFeatures):

    def __init__(self, trees):
        self._trees = trees 

    def get_features(self, df_row):
        """Return vector of feature values.

        Args:
            df_row (pd.Series): instance as pd.Series containing at least the
                word level spans of the target (candidate) and senti expression
        """
        if df_row.targetStart == -1 or df_row.targetEnd == -1:
            raise NotATargetRelationError

        tree = self._trees[df_row["sentence"]]

        if df_row.sentence == df_row.sentence[df_row.targetStart:df_row.targetEnd]: # if target is the whole sentence
            target_tree = tree 

        else:
            # transform character spans to token spans  
            _, _, target_start_token_span, target_end_token_span =  \
                char_span_to_token_span(df_row)

            target_tree = get_subtree_by_span(tree,
                                            target_start_token_span,
                                            target_end_token_span)

        # one hot encode tree labels
        enc = OneHotEncoder(handle_unknown="ignore")
        enc.fit([[i] for i in POS_TAGS])
        oh_tlabel = list(enc.transform([[target_tree.label()]]).toarray()[0])
        lca = self._get_lowest_common_ancestor(tree, df_row) 
        oh_lcalabel = list(enc.transform([[lca.label()]]).toarray()[0]) 
        return oh_tlabel + oh_lcalabel
           
           
    @staticmethod
    def _get_lowest_common_ancestor(tree, df_row):
        """Find phrase that connects target to sentiment expression."""
        # transform character spans to token spans  
        sentexpr_start_token_span, sentexpr_end_token_span, target_start_token_span, target_end_token_span =  \
            char_span_to_token_span(df_row)

        trees = tree.subtrees(
            filter=lambda x: all([
                target_start_token_span in range(x.span_start(), x.span_end()),
                target_end_token_span in range(x.span_start(), x.span_end()),

                (sentexpr_start_token_span in range(x.span_start(), x.span_end()) or \
                 sentexpr_end_token_span in range(x.span_start(), x.span_end()))
            ])
        )
        trees = list(trees)
        if trees:
            return trees[-1]  # last tree in list is the lowest common node
        
        return tree # if target is the whole sentence or the lca is already the root of the target 


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

    constituency_feature = ConstituencyParseFeatures(trees)
    with open("../output/instances/test_constituency.csv", "w") as f_out:
            for idx in range(0,len(items_df)):
                try:
                    item = items_df.iloc[idx]
                    print(constituency_feature.get_features(item), file=f_out)
                except NotATargetRelationError:
                    print(f'Row {idx} in {os.path.split(items_df_path)[1]} is skipped, since it is not a sentiment-expression-to-target (probably a sentiment-expression-to-source relation instead).')
                    continue

if __name__ == "__main__":
    test_write_all_instances_to_file('../output/UNSC_2014_SPV.7154_sentsplit.csv')






