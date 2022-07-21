# -*- coding: utf-8 -*-
"""
Class for generating features related to dependency parsing by using get_features(),
which needs the information from a row (pandas.Series) from the corpus reader data frame.
"""

import pandas as pd
import spacy
from spacy import displacy
from sklearn.preprocessing import OneHotEncoder
from tree_utils.tree_op import lowest_common_ancestor, distance_btw_3_pts

from .abstract_features import AbstractFeatures
from .feature_utils import DEP_TAGS, FINE_GRAINED_POS_TAGS


class SpansError(Exception):
    pass


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
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.features = []

        self.target_head = None
        self.senti_head = None
        self.lowest_ancestor_of_heads = None

        self.ohe_dep = OneHotEncoder(handle_unknown='ignore')
        self.ohe_dep.fit([[tag] for tag in DEP_TAGS])

        self.ohe_pos = OneHotEncoder(handle_unknown='ignore')
        self.ohe_pos.fit([[tag] for tag in FINE_GRAINED_POS_TAGS])

    def get_features(self, df_row):
        """Create and return the feature vector for the incoming instance.

        Args:
            df_row(pd.Series): a row from the corpus reader dataframe.

        Returns:
            list: list of features

        Raises:
            SpansError: If target head and sentiment expression head canno be
                found in the sentence. A list of -1.0 will be returned as the
                features vector in this case. This can be traced back to the
                mistakes in the spans of sentiment expression and/or target
                from the annotation.
        """
        sentence = df_row.sentence
        sent_span_start = df_row.sentexprStart
        sent_span_end = df_row.sentexprEnd
        target_span_start = df_row.targetStart
        target_span_end = df_row.targetEnd
        # preparatory calculations for creating the features
        self._preparation_for_features(sentence,
                                       sent_span_start,
                                       sent_span_end,
                                       target_span_start,
                                       target_span_end)
        # create and save features
        try:
            if self.senti_head is None or self.target_head is None:
                raise SpansError
            else:
                self._rel_btw_heads()
                self._pos_senti_head()
                self._pos_target_head()
                self._distance_btw_heads()
        except SpansError:
            # WARNING: hardcode the length of vector
            self.features.extend([-1.0] * 184)
        return self.features
    
    def _preparation_for_features(self, sentence, sent_span_start, sent_span_end, target_span_start, target_span_end):
        """Complete preparatory work for feature engineering, 

        Most importantly, calculate the heads of the target/sentiment
        expression and the lowest common ancestor of these heads in the
        dependency parse.

        Args:
            sentence(str): sentence which includes the sentiment expression and
                target
            sent_span_start(int): span start of sentiment expression
            sent_span_end(int): span end of sentiment expression
            target_span_start(int): span start of target
            target_span_end(int): span end of target
        """
        self._reset_attributes()
        self.sent_doc = self.nlp(sentence)
        self._heads_of_senti_target(sentence,
                                    sent_span_start,
                                    sent_span_end,
                                    target_span_start,
                                    target_span_end)
        self._lowest_ancestor_of_heads()

    def _reset_attributes(self):
        """Reset some of the attriutes for working with another sentence."""
        self.features = []
        self.target_head = None
        self.senti_head = None
        self.lowest_ancestor_of_heads = None
    
    def _heads_of_senti_target(self, sentence, senti_span_start, senti_span_end, target_span_start, target_span_end):
        """Find and save the heads of the sentiment expression and target.

        Args:
            sentence(str): sentence which includes the sentiment expression
                and target
            sent_span_start(int): span start of sentiment expression
            sent_span_end(int): span end of sentiment expression
            target_span_start(int): span start of target
            target_span_end(int): span end of target
        """
        # senti_start_i, senti_end_i, target_start_i, target_end_i = \
            # self._span2i(sentence, senti_span_start, senti_span_end, target_span_start, target_span_end)

        self.senti_head = self._find_head(senti_span_start, senti_span_end, self.sent_doc)
        self.target_head = self._find_head(target_span_start, target_span_end, self.sent_doc)

    # def _span2i(self, sentence, senti_span_start, senti_span_end, target_span_start, target_span_end):
        # """Convert the character level spans to token level indices.

        # Args:
            # sentence(str): sentence which includes the sentiment expression and
                # target
            # sent_span_start(int): span start of sentiment expression
            # sent_span_end(int): span end of sentiment expression
            # target_span_start(int): span start of target
            # target_span_end(int): span end of target

        # Returns:
            # senti_start_i(int): starting token index of the sentiment
                # expression in the sentence
            # senti_end_i(int): ending token index of the sentiment expression in
                # the sentence
            # target_start_i(int): starting token index of the target in the
                # sentence
            # target_end_i(int): ending token index of the target in the sentence
        # """
        # sent_ex = [token.text for token in self.nlp(sentence[senti_span_start:senti_span_end])]
        # target = [token.text for token in self.nlp(sentence[target_span_start:target_span_end])]

        # # find the start and end indices of sentiment expr. and target respectively
        # senti_start_i, senti_end_i = self._find_token_i(sent_ex, self.sent_doc)
        # target_start_i, target_end_i = self._find_token_i(target, self.sent_doc)

        # return senti_start_i, senti_end_i, target_start_i, target_end_i

    def _find_token_i(self, phrase, sent_doc):
        """
        It returns the start and end indices of the given phrase in a sentence.
        Args:
            phrase(list of str): list of tokens in the phrase
            sent_doc(spacy.Doc): Doc object of the corresponding sentence

        Returns:
            start_i: starting token index of the phrase in the sentence
            end_i: ending token index of the phrase in the sentence

        Raises:
            IndexError: If the given phrase cannot be found in the sent_doc.
            -1, -1 will be returned as the indices in this case.
            This can be traced back to the mistakes in the spans
            of sentiment expression and/or target from the annotation. 
        """
        try:
            start_i, end_i = -1, -1
            for token in sent_doc:
                if token.text == phrase[0]:
                    phrase_found = True
                    # check if all the tokens in ex match
                    for j in range(1, len(phrase)):
                        if sent_doc[token.i + j].text != phrase[j]:
                            phrase_found = False
                            break
                    if phrase_found:
                        start_i = token.i
                        end_i = start_i + len(phrase) 

            if start_i == -1 and end_i == -1:
                raise IndexError

            return start_i, end_i # end_i is used for slicing, so exclusive
        
        except IndexError:
            return -1, -1

    def _find_head(self, phrase_start_i, phrase_end_i, sent_doc):
        """
        It returns the head of the a phrase.
        Args:
            phrase_start_i: starting token index of the phrase in the sentence
            phrase_end_i: ending token index of the phrase in the sentence
            sent_doc: Doc object of the corresponding sentence
        
        Returns:
            head(spacy.Token): head of the phrase
        """
        head = None
        # for one-token-phrase
        if phrase_end_i - phrase_start_i == 1:
            head = self.sent_doc[phrase_start_i]
        
        # for phrase of multiple tokens
        else:
            tokens_in_phrase = sent_doc[phrase_start_i:phrase_end_i]
            for token in tokens_in_phrase:
                if token.head not in tokens_in_phrase or token.dep_ == 'ROOT':
                    head = token
        return head

    def _lowest_ancestor_of_heads(self):
        """Find lowest common ancestor of target and sentiment expression head."""
        self.lowest_ancestor_of_heads = lowest_common_ancestor(self.senti_head, 
                                                                self.target_head,
                                                                self._find_root())

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


def test_single_instance(n, items_df, dep_feature):
    item = items_df.iloc[n]

    print(len(dep_feature.get_features(item)))
    print("target head:", dep_feature.target_head)
    print("senti head:", dep_feature.senti_head)
    # show parse info in text
    # for token in dep_feature.sent_doc:
    #     print(f"{token}\t\t{token.dep_}\t\t{token.head.text}\t\t") 
    #visualize dependency parse
    #displacy.serve(dep_feature.sent_doc, style='dep')


def test_write_all_instances_to_file(items_df, dep_feature):
    with open("../test_files/test_dependency.csv", "w") as f_out:
        for idx in range(0,len(items_df)):
            item = items_df.iloc[idx]
            print(dep_feature.get_features(item), file=f_out)


if __name__ == "__main__":
    items_df = pd.read_pickle("../test_files/items.pkl")
    dep_feature = DependencyParseFeatures()
    test_single_instance(441, items_df, dep_feature)
    #test_write_all_instances_to_file(items_df, dep_feature)