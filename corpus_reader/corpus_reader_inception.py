# -*- coding: utf-8 -*-
"""
Reader class for the UNSC sub-corpus annotated with INCEpTION and in json format 
extracting sentiment expressions as well as 
their corresponding targets and the sentences they are in.
"""
import argparse
import os
import json

import pandas as pd


class RelationBeyondSentenceError(Exception):
    pass

class InceptionCorpusReader:

    def __init__(self, inception_anns_root, raw_text_root):
        """Constructor of the InceptionCorpusReader class.

        Args:
            inception_anns_root (str): path of the directory containing the json
                annotation files created with Prodigy 

        Attributes:
            self.anns_files: list of all annotation files in directory
            self.items: pd.DataFrame with sentiment items
        """
        self.raw_text_root = raw_text_root
        
        self.anns_files = self._find_all_anns_files(inception_anns_root)
        self.items_df = self. _all_jsons_to_df()

    def _find_all_anns_files(self, prodigy_anns_root):
        return [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(prodigy_anns_root)
            for filename in filenames if filename.endswith(".json")
        ]

    def items_df_to_csv(self, f_out:str):
        """Writes the annotation items dataframe into a csv file."""
        if f_out.endswith('.csv'):
            self.items_df.to_csv(f_out)
        else:
            raise Exception('please pass a csv filename!')
   
    def _all_jsons_to_df(self):
        """Returns an annotation dataframe for all json files in the root directory."""
        items_df = pd.DataFrame(columns=[
                                    "rawTextFilename",
                                    "sentenceID",
                                    "sentence",
                                    "sentexprStart",
                                    "sentexprEnd",
                                    "targetStart",
                                    "targetEnd",
                                    "sourceStart",
                                    "sourceEnd",
                                    ])

        for json_path in self.anns_files:
            items_df = pd.concat([items_df, self._json_to_tmp_df(json_path)],
                                 ignore_index=True)
        return items_df

    def _json_to_tmp_df(self, json_path) -> pd.DataFrame:
        '''Returns a dataframe from a single json file, which contains the following columns:
        "rawTextFilename", "sentenceID", "sentence", "sentexprStart", "sentexprEnd", 
        "targetStart", "targetEnd", "sourceStart" and "sourceEnd".
        Each row of the dataframe corresponds to one relation.'''
        raw_text_path = os.path.join(
            self.raw_text_root,
            os.path.split(json_path)[1].replace('.json', '.txt')
        )
        
        with open(json_path, 'r') as json_file, open(raw_text_path, 'r') as raw_text_file:
            anns_dict = json.load(json_file)
            raw_text = raw_text_file.read()
        
        sentence_spans_dict , senti_expr_items, target_items, source_items, rel_items = self._group_anns_items(anns_dict)
        tmp_rows = [] # for dataframe

        for item in rel_items: # iterate over all relations
            current_sentence_id, current_sentence_span = self._find_current_sentence(item, sentence_spans_dict)
            sentexprStart, sentexprEnd, targetStart, targetEnd, sourceStart, sourceEnd = -1, -1, -1, -1, -1, -1

            try:
                # find governor and dependent
                governor_id = item['@Governor'] # must be sentexpr
                dependent_id = item['@Dependent'] # either target or source

                # all the spans are based on the whole document and thus need to be subtracted by the sentence span begin,
                # since only the sentence instead of the whole document i.e. speech will be store in the dataframe.
                sentexprStart = senti_expr_items[governor_id]['begin'] - current_sentence_span[0]
                sentexprEnd = senti_expr_items[governor_id]['end'] - current_sentence_span[0]

                if target_items.get(dependent_id) is None:
                    sourceStart = source_items[dependent_id]['begin'] - current_sentence_span[0]
                    sourceEnd = source_items[dependent_id]['end'] - current_sentence_span[0]
                else:
                    targetStart = target_items[dependent_id]['begin'] - current_sentence_span[0]
                    targetEnd = target_items[dependent_id]['end'] - current_sentence_span[0]

                tmp_rows.append([
                    os.path.split(json_path)[1].replace('.json', '.txt'),
                    current_sentence_id,
                    raw_text[current_sentence_span[0]:current_sentence_span[1]], 
                    sentexprStart,
                    sentexprEnd,
                    targetStart,
                    targetEnd,
                    sourceStart,
                    sourceEnd
                ])

            except KeyError: 
                print(f"Relation (id: {item['%ID']}) in {os.path.split(json_path)[1]} skipped due to annotation mistakes, e.g. the direction of the relation was annotated reversely.")
                continue

            except RelationBeyondSentenceError:
                print(f"Relation (id: {item['%ID']}) in {os.path.split(json_path)[1]} skipped since it goes across sentences.")
                continue

        return pd.DataFrame(tmp_rows,
                            columns=[
                                    "rawTextFilename",
                                    "sentenceID",
                                    "sentence",
                                    "sentexprStart",
                                    "sentexprEnd",
                                    "targetStart",
                                    "targetEnd",
                                    "sourceStart",
                                    "sourceEnd",
                                    ])

    def _group_anns_items(self, anns_dict):
        """Groups annotation items according to the nature (ORLSpan/ ORLRelation/Sentence) or 
        label of a ORLSpan(SubExp, Source, Target) and returns them."""
        sentence_spans = dict()
        senti_expr_items = dict()
        target_items = dict()
        source_items = dict()
        rel_items = []

        for item in anns_dict['%FEATURE_STRUCTURES']:
            item_id = int(item['%ID'])

            if item['%TYPE'] == 'webanno.custom.ORLRelation':
                rel_items.append(item) 
            
            elif item['%TYPE'] == 'webanno.custom.ORLSpan':
                if item['Roles2'] == 'SubExp':
                    senti_expr_items[item_id] = item
                elif item['Roles2'] == 'Target':
                    target_items[item_id] = item
                elif item['Roles2'] == 'Source':
                    source_items[item_id] = item
            
            elif item['%TYPE'] == 'de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence':
                sentence_spans[item_id] = (item['begin'], item['end'])
        
        return sentence_spans, senti_expr_items, target_items, source_items, rel_items

    def _find_current_sentence(self, relation_item, sentence_spans_dict):
        """Retrieves the id and span of the current sentence by comparing
        the relation item span to the span of each sentence. 

        raises:
            RelationBeyondSentenceError: raised if the relation goes across sentence,
                since the annotation must be sentence-based (for parsing later).
        """
        for sentence_id, sentence_span in sentence_spans_dict.items():
            # check if the item locates within the sentence
            if relation_item['begin'] >= sentence_span[0] and relation_item['end'] <= sentence_span[1]:
                return sentence_id, sentence_span

        raise RelationBeyondSentenceError()

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--anno_root", help="root directory of the annotation json files") # ../data/inception/
    arg_parser.add_argument("-r", "--raw_text_root", help="root directory of the raw texts") # ../data/inception/raw_text/
    arg_parser.add_argument("-o", "--csv_path", help="path of csv file to which the annotation dataframe should be written") # ../output/UNSC_2014_SPV.7154_sentsplit.csv
    args = arg_parser.parse_args()
    
    corpus_reader = InceptionCorpusReader(args.anno_root, args.raw_text_root)
    corpus_reader.items_df_to_csv(args.csv_path)