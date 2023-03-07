# -*- coding: utf-8 -*-
"""
Class for generating features related to dependency parsing by using get_features(),
which needs the information from a row (pandas.Series) from the corpus reader DataFrame.
Features implemented:
1. Relation between the two heads
2. POS-tag of the sentiment head
3. POS-tag of the target head
4. Distance between the two heads
"""
import os

import pandas as pd
import spacy
from sklearn.preprocessing import OneHotEncoder

from features.abstract_features import AbstractFeatures
from features.feature_utils import DEP_TAGS, FINE_GRAINED_POS_TAGS, lowest_common_ancestor, distance_btw_3_pts, NotATargetRelationError, SpansError, char_span_to_token_span


class DependencyParseFeatures(AbstractFeatures):

    def __init__(self):
        """Constructor of the DependencyParseFeatures class.
        
        Attributes:
            self.nlp(spacy.lang.en.English)
            self.features(list): list of features
            self.target_head(spacy.Token)
            self.senti_head(spacy.Token)
            self.lowest_ancestor_of_heads(spacy.Token): lowest common ancestor
                of target head and sentiment expression head
            self.sent_doc(spacy.Doc)
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.features = []

        self.current_sentence_id = -1
        self.target_head = None
        self.senti_head = None
        self.lowest_ancestor_of_heads = None
        self.sent_doc = None

        self.ohe_dep = OneHotEncoder(handle_unknown='ignore')
        self.ohe_dep.fit([[tag] for tag in DEP_TAGS])

        self.ohe_pos = OneHotEncoder(handle_unknown='ignore')
        self.ohe_pos.fit([[tag] for tag in FINE_GRAINED_POS_TAGS])

    def get_features(self, df_row):
        """Create and return the feature vector for the incoming instance.

        Args:
            df_row(pd.Series): a row from the corpus reader dataframe.

        Returns:
            list: list of features for a row from the corpus reader dataframe.

        Raises:
            SpansError: If target head and sentiment expression head cannot be
                found in the sentence.
        """
        # preparatory calculations for creating the features
        self._preparation_for_features(df_row)
        
        # create and save features
        if self.senti_head is None or self.target_head is None:
            raise SpansError
        else:
            self._rel_btw_heads()
            self._pos_senti_head()
            self._pos_target_head()
            self._distance_btw_heads()

        return self.features
    
    def _preparation_for_features(self, df_row): 
        """Complete preparatory work for feature engineering.
        Most importantly, calculate the heads of the target/sentiment
        expression and the lowest common ancestor of these heads in the
        dependency parse.

        raises:
            NotATargetRelationError: when the start/end span of the target of
                row is -1. This indicates that the row corresponds to a source relation
                instead of a target relation.
        """
        if df_row.targetStart == -1 or df_row.targetEnd == -1:
            raise NotATargetRelationError

        self._reset_attributes()
        
        # generates dependency parse if it is an unparsed sentence to avoid duplicate parsing
        if df_row.sentenceID != self.current_sentence_id:
            self.current_sentence_id = df_row.sentenceID
            self.sent_doc = self.nlp(df_row.sentence)

        # convert character span from annotation to token span 
        senti_start_token_i, senti_end_token_i, target_start_token_i, target_end_token_i = \
            self._find_token_idx(df_row)
        
        # find heads and their lowest ancestor
        self.senti_head = self._find_head(self.sent_doc[senti_start_token_i:senti_end_token_i])
        self.target_head = self._find_head(self.sent_doc[target_start_token_i:target_end_token_i])
        self.lowest_ancestor_of_heads = self._lowest_ancestor_of_heads()

    def _reset_attributes(self):
        """Reset some of the attriutes for working with another sentence."""
        self.features = []
        self.target_head = None
        self.senti_head = None
        self.lowest_ancestor_of_heads = None

    def _find_token_idx(self, df_row):
        """
        It returns the start and end indices of the given phrase in a sentence.
        Args:
            phrase(str): i.e. sentiment expression/target

        Returns:
            start_idx: starting token index of the phrase in the sentence
            end_idx: ending token index of the phrase in the sentence

        Raises:
            IndexError: If the given phrase cannot be found in the sent_doc.
            -1, -1 will be returned as the indices in this case.
            This can be traced back to the mistakes in the spans
            of sentiment expression and/or target from the annotation. 
        """
        return char_span_to_token_span(df_row, tokenize_func=self.nlp)
    
    def _find_head(self, tokens_in_phrase):
        """It returns the head of the a phrase.
        Args:
             tokens_in_phrase (list of spacy.Token): list of tokens of a given phrase i.e. sentiment expression/target
         Returns:
             head(spacy.Token): head of the phrase
    """
        head = None
        # for one-token-phrase
        if len(tokens_in_phrase) == 1:
            head = tokens_in_phrase[0]
        # for phrase of multiple tokens
        else:
            for token in tokens_in_phrase:
                if token.head not in tokens_in_phrase or token.dep_ == 'ROOT':
                    head = token
                    break
        return head

    def _lowest_ancestor_of_heads(self):
        """Find lowest common ancestor of target and sentiment expression head."""
        return lowest_common_ancestor(self.senti_head, self.target_head, self._find_root())

    def _find_root(self):
        """Find the sentence root.

        Returns:
            Root of sentence.
            None if senti head and target head are in different sentences.
        """
        for sent in self.sent_doc.sents:
            if self.senti_head in sent and self.target_head in sent:
                return sent.root
        return None

    def _distance_btw_heads(self):
        """Find the distance between target and sentiment expression head."""
        self.features.append(
            distance_btw_3_pts(self.senti_head,
                               self.target_head,
                               self.lowest_ancestor_of_heads)
        )

    def _pos_senti_head(self):
        self.features.extend(
            list(self.ohe_pos.transform([[self.senti_head.tag_]]).toarray()[0])
        )

    def _pos_target_head(self):
        self.features.extend(
            list(self.ohe_pos.transform([[self.target_head.tag_]]).toarray()[0])
        )

    def _rel_btw_heads(self):
        """Find the relation between target and sentiment expression head."""
        # current_child is self.target_head
        self.features.extend(
            list(self.ohe_dep.transform([[self.target_head.dep_]]).toarray()[0])
        )

########## For Trial ##########

def test_single_instance(n, items_df_path, dep_feature):
    items_df = pd.read_csv(items_df_path)
    item = items_df.iloc[n]
    print(len(dep_feature.get_features(item)))
    print("target head:", dep_feature.target_head)
    print("senti head:", dep_feature.senti_head)


def test_write_all_instances_to_file(items_df_path, dep_feature):
    items_df = pd.read_csv(items_df_path)
    with open("../output/instances/test_dependency.csv", "w") as f_out:
            for idx in range(0,len(items_df)):
                try:
                    item = items_df.iloc[idx]
                    print(dep_feature.get_features(item), file=f_out)
                except NotATargetRelationError:
                    print(f'Row {idx} in {os.path.split(items_df_path)[1]} is skipped, since it is not a sentiment-expression-to-target (probably a sentiment-expression-to-source relation instead).')
                    continue


if __name__ == "__main__":
    items_df_path = "../output/UNSC_2014_SPV.7154_sentsplit.csv"
    dep_feature = DependencyParseFeatures()
    # test_single_instance(3, items_df_path, dep_feature)
    test_write_all_instances_to_file(items_df_path, dep_feature)