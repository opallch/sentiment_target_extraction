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

        self.target_head_i = -1
        self.senti_head_i = -1
        self.lowest_ancestor_of_heads_i = -1

        self.index_err_count = 0
        self.rec_err_count = 0

    def _reset_attributes(self):
        self.features = []
        self.target_head_i = -1
        self.senti_head_i = -1

    def get_features(self, sentence, sent_expr_start, sent_expr_end, target_start, target_end):
        
        self._reset_attributes()
        self._preparation(sentence, sent_expr_start, sent_expr_end, target_start, target_end)
        self._distance_btw_heads()
        #self._rel_btw_heads(self.target_parent_i)
   
        # FOR TEST
        # show dependency parse
        # displacy.serve(self.sent_doc, style='dep')
        # for token in self.sent_doc:
        #      print(f"{token}\t\t{token.dep_}\t\t{token.head.text}\t\t") 

        return self.features  
    
    def _preparation(self, sentence, senti_span_start, senti_span_end, target_span_start, target_span_end):
        # check the indices of target and senti expr
        self.sent_doc = self.nlp(sentence)
        sent_ex = [token.text for token in self.nlp(sentence[senti_span_start:senti_span_end])]
        target = [token.text for token in self.nlp(sentence[target_span_start:target_span_end])]
        
        # find the start and end indices of sentiment expr. and target respectively
        senti_start_i, senti_end_i = self._find_token_i(sent_ex, self.sent_doc)
        target_start_i, target_end_i = self._find_token_i(target, self.sent_doc)

        # find the indices of the target head and its parent
        self.senti_head_i = self._find_head_i(senti_start_i, senti_end_i, self.sent_doc)
        self.target_head_i = self._find_head_i(target_start_i, target_end_i, self.sent_doc)
        self.lowest_ancestor_of_heads_i = lowest_common_ancestor(self.senti_head_i, self.target_head_i)
        self.target_parent_i = self.sent_doc[self.target_head_i].i
    
    def _find_token_i(self, phrase, sent_doc):
        """returns the start and end indices of the given phrase in a sentence."""
        try:
            start_i, end_i = -1, -1
            for token in sent_doc:
                if token.text == phrase[0]:
                    phrase_found = True
                    # check if all the tokens in ex match
                    for j in range(1, len(phrase)):
                        #print(ex[j])
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
            #print("Index Error")
            #print(token.text)
            self.index_err_count += 1
            return -1, -1

    def _find_head_i(self, phrase_start_i, phrase_end_i, sent_doc):
        """returns the index of the head of the a phrase."""
        head_i = 0
        # for one-token-phrase
        if phrase_end_i - phrase_start_i == 1:
            head_i = phrase_start_i
        
        # for phrase of multiple tokens
        else:
            tokens_in_phrase = sent_doc[phrase_start_i:phrase_end_i]
            for token in tokens_in_phrase:
                if token.head not in tokens_in_phrase: 
                    head_i = token.i
        return head_i

    def _distance_btw_heads(self):
        # LATER:
        # self.features.append(
        #     distance_btw_3_pts(self.senti_head_i, self.target_head_i, self.lowest_ancestor_of_heads_i, self.doc)
        # )
        
        self.features.append(
            np.abs(self.senti_head_i - self.target_head_i)
        )

    def _rel_btw_heads(self):
        current_child = self.sent_doc[self.target_head_i]
        current_parent = current_child.head
        
        while current_parent.i != self.lowest_ancestor_of_heads_i:
            current_child = current_parent
            current_parent = current_parent.head
        
        self.features.append(current_child.dep_)

    # def _rel_btw_heads(self, ancestor_i):
    #     rel = self._find_rel_btw_heads(ancestor_i)
    #     self.features.append(rel)
    
    # def _find_rel_btw_heads(self, ancestor_i):

    #     # rel as or the hash (token.dep instead of token.dep_)
    #     target_ancestor = self.sent_doc[ancestor_i]
        
    #     try:
    #         # base case 1
    #         if self.senti_head_i == ancestor_i: 
    #             return self.sent_doc[ancestor_i].dep_
            
    #         # base case 2
    #         elif self.sent_doc[self.senti_head_i] in list(self.sent_doc[ancestor_i].children):
    #             for child in self.sent_doc[ancestor_i].children:
    #                 if child.i == self.senti_head_i:
    #                     return child.dep_           
            
    #         # recursive case, need to be fixed!
    #         else: 
    #             return self._find_rel_btw_heads(target_ancestor.head.i) # target grandparent, or "ancestor"
        
    #     except RecursionError:
    #         #print(sent_doc[senti_head_i])
    #         #print(list(sent_doc[target_parent_i].children))
    #         self.rec_err_count += 1
    #         return ""
        
def single_instance(n):
    items_df = pd.read_pickle("../test_files/items.pkl")

    item = items_df.iloc[n]
    sent = item.sentence
    print(sent)
    print(f"sentExpr: {sent[item.sentexprStart:item.sentexprEnd]}\nTarget: {sent[item.targetStart:item.targetEnd]}")
    dep_feature = DependencyParseFeatures()
    print(
        dep_feature.get_features(sent, item.sentexprStart, item.sentexprEnd, item.targetStart, item.targetEnd,)
    )
    print("target head:", dep_feature.target_head_i)
    print("senti head:", dep_feature.senti_head_i)

def write_all_instances():
    items_df = pd.read_pickle("../test_files/items.pkl")
    dep_feature = DependencyParseFeatures()
    with open("../test_files/test_dependency2", "w") as f_out:
        for idx in range(0,len(items_df)):
            item = items_df.iloc[idx]
            sent = item.sentence
            print(
                dep_feature.get_features(sent, item.sentexprStart, item.sentexprEnd, item.targetStart, item.targetEnd,)
                , file=f_out
            )
    
    print(f"SpansError:{dep_feature.index_err_count}\nRecursionError: {dep_feature.rec_err_count}")

if __name__ == "__main__":
    single_instance(1100)
    #write_all_instances()
    
"""The United States has been preparing annual reports on human rights in 190 countries for 25 years while ignoring the real situation at home.
104 139 0 17
SENTI: ignoring the real situation at home
TARGET: The United States

pretty print dependency tree: https://stackoverflow.com/questions/36610179/how-to-get-the-dependency-tree-with-spacy
"""