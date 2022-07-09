# in: Serie (a row from the copus reader df)
# out: feautres vector

# GOAL: 
# first step: create positive instances
# second step: negativ instances, we need info from constituency parsing! (e.g. NP as candidate)

# TMR:
# traverse the dependency parse
# fix Index Error issue lol
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
import spacy
from spacy import displacy
from abstract_features import AbstractFeatures
from tree_utils.tree_op import lowest_common_ancestor, distance_btw_3_pts

class DependencyParseFeatures(AbstractFeatures):
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.features = []

        self.target_head = None
        self.senti_head = None
        self.lowest_ancestor_of_heads = None

        self.index_err_count = 0
        self.rec_err_count = 0

    def _reset_attributes(self):
        self.features = []
        self.target_head = None
        self.senti_head = None
        self.lowest_ancestor_of_heads = None

    def get_features(self, sentence, sent_span_start, sent_span_end, target_span_start, target_span_end):
        # preparations
        self._reset_attributes()
        self.sent_doc = self.nlp(sentence)
        self._heads_of_senti_target(sentence, sent_span_start, sent_span_end, target_span_start, target_span_end)
        self._lowest_ancestor_of_heads()
        # self.target_parent_i = self.sent_doc[self.target_head_i].i
   
        # FOR TEST
        #show dependency parse
        #displacy.serve(self.sent_doc, style='dep')
        # for token in self.sent_doc:
        #      print(f"{token}\t\t{token.dep_}\t\t{token.head.text}\t\t") 

        # features
        if self.senti_head and self.target_head:
            self._rel_btw_heads()
            self._distance_btw_heads()
        else:
            self.features.append("")
            self.features.append(-1)

        return self.features
    
    def _heads_of_senti_target(self, sentence, senti_span_start, senti_span_end, target_span_start, target_span_end):
        """finds the heads of sentiment expr and target."""
        senti_start_i, senti_end_i, target_start_i, target_end_i = \
            self._span2i(sentence, senti_span_start, senti_span_end, target_span_start, target_span_end)

        # find the indices of the target head and its parent
        self.senti_head = self._find_head(senti_start_i, senti_end_i, self.sent_doc)
        self.target_head = self._find_head(target_start_i, target_end_i, self.sent_doc)

        print("target head:", self.target_head)
        print("senti head:", self.senti_head)
    
    def _span2i(self,sentence, senti_span_start, senti_span_end, target_span_start, target_span_end):
        """coverts the spans of sentiment expr and target to token indices in the sentence."""
        sent_ex = [token.text for token in self.nlp(sentence[senti_span_start:senti_span_end])]
        target = [token.text for token in self.nlp(sentence[target_span_start:target_span_end])]
        
        #print("sent ex from span: ", sent_ex)
        #print("target from span: ", target)

        # find the start and end indices of sentiment expr. and target respectively
        senti_start_i, senti_end_i = self._find_token_i(sent_ex, self.sent_doc)
        target_start_i, target_end_i = self._find_token_i(target, self.sent_doc)

        return senti_start_i, senti_end_i, target_start_i, target_end_i

    def _find_token_i(self, phrase, sent_doc):
        """returns the start and end indices of the given phrase in a sentence."""
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
            # DUE TO INCORRECT SPAN FROM GATE
            print("Incorrect Spans")
            self.index_err_count += 1
            return -1, -1

    def _find_head(self, phrase_start_i, phrase_end_i, sent_doc):
        """returns the head of the a phrase."""
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

        self.lowest_ancestor_of_heads = lowest_common_ancestor(self.senti_head, 
                                                                self.target_head,
                                                                self._find_root())

    def _find_root(self):
        root = None
        for sent in self.sent_doc.sents:
            print("root: ", sent.root)
            if self.senti_head in sent and self.target_head in sent:
                root = sent.root

        print(root)
        return root

    def _distance_btw_heads(self):
        self.features.append(
            distance_btw_3_pts(self.senti_head, self.target_head, self.lowest_ancestor_of_heads)
        )

    def _rel_btw_heads(self):
        try:
            current_child = self.target_head
            # current_parent = current_child.head
            
            # while current_parent != self.lowest_ancestor_of_heads:
            #     current_child = current_parent
            #     current_parent = current_parent.head
            
            self.features.append(current_child.dep_)
        
        except AttributeError:
            self.features.append("")
        
def single_instance(n):
    items_df = pd.read_pickle("../test_files/items.pkl")

    item = items_df.iloc[n]
    sent = item.sentence
    print(sent)
    #print(f"sentExpr: {sent[item.sentexprStart:item.sentexprEnd]}\nTarget: {sent[item.targetStart:item.targetEnd]}")
    
    print()
    dep_feature = DependencyParseFeatures()
    print(
        dep_feature.get_features(sent, item.sentexprStart, item.sentexprEnd, item.targetStart, item.targetEnd,)
    )
    print("target head:", dep_feature.target_head)
    print("senti head:", dep_feature.senti_head)
    
    

def write_all_instances():
    try:
        skipped = [441, 465, 611, 703, 905]
        items_df = pd.read_pickle("../test_files/items.pkl")
        dep_feature = DependencyParseFeatures()
        with open("../test_files/test_dependency3", "w") as f_out:
            for idx in range(0,len(items_df)):
                if idx in skipped:
                    continue
                
                print(idx)
                item = items_df.iloc[idx]
                sent = item.sentence
                print(
                    dep_feature.get_features(sent, item.sentexprStart, item.sentexprEnd, item.targetStart, item.targetEnd,)
                    , file=f_out
                )
        
        print(f"SpansError:{dep_feature.index_err_count}\nRecursionError: {dep_feature.rec_err_count}")
    
    except RecursionError:
        print("RecursionError")
        dep_feature.rec_err_count += 1
    

if __name__ == "__main__":
    single_instance(905)
    #write_all_instances()
    
"""The United States has been preparing annual reports on human rights in 190 countries for 25 years while ignoring the real situation at home.
104 139 0 17
SENTI: ignoring the real situation at home
TARGET: The United States

pretty print dependency tree: https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
"""