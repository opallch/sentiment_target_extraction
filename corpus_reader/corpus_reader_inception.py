# -*- coding: utf-8 -*-
"""
Reader class for the UNSC sub-corpus annotated with INCEpTION and in json format 
extracting sentiment expressions as well as 
their corresponding targets and the sentences they are in.
"""
import argparse
import os
import json

import numpy as np
import pandas as pd


class InceptionCorpusReader:

    def __init__(self, inception_anns_root):
        """Constructor of the InceptionCorpusReader class.

        Args:
            inception_anns_root (str): path of the directory containing the json
                annotation files created with Prodigy 

        Attributes:
            self.anns_files: list of all annotation files in directory
            self.items: pd.DataFrame with sentiment items
        """
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
        items_df = pd.DataFrame(columns=["rawTextFilename",
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
        """Returns a dataframe containing spans of annotation items from a single json file."""
        with open(json_path, 'r') as json_in:
            anns_dict = json.load(json_in)
        
        senti_expr_items, target_items, source_items, rel_items = self._group_anns_items(anns_dict)
        tmp_rows = [] # for dataframe

        for item in rel_items:
            sentexprStart, sentexprEnd, targetStart, targetEnd, sourceStart, sourceEnd = \
                -1, -1, -1, -1, -1, -1

            try:
                # find governor and dependent
                governor_id = item['@Governor'] # must be sentexpr
                dependent_id = item['@Dependent'] # either target or source

                sentexprStart = senti_expr_items[governor_id]['begin']
                sentexprEnd = senti_expr_items[governor_id]['end']

                if target_items.get(dependent_id) is None:
                    sourceStart = source_items[dependent_id]['begin']
                    sourceEnd = source_items[dependent_id]['end']
                else:
                    targetStart = target_items[dependent_id]['begin']
                    targetEnd = target_items[dependent_id]['end']

                tmp_rows.append([
                    os.path.split(json_path)[1].replace('.json', '.txt'), # raw text filename
                    sentexprStart,
                    sentexprEnd,
                    targetStart,
                    targetEnd,
                    sourceStart,
                    sourceEnd
                ])

            except KeyError: # catch annotation mistakes
                continue    

        return pd.DataFrame(tmp_rows,
                            columns=["rawTextFilename",
                                    "sentexprStart",
                                    "sentexprEnd",
                                    "targetStart",
                                    "targetEnd",
                                    "sourceStart",
                                    "sourceEnd",
                                    ])

    def _group_anns_items(self, anns_dict):
        """Groups annotation items according to the nature (ORLSpan/ ORLRelation) or 
        label(SubExp, Source, Target) and returns them."""
        senti_expr_items = dict()
        target_items = dict()
        source_items = dict()
        rel_items = []

        for item in anns_dict['%FEATURE_STRUCTURES']:
            if item['%TYPE'] == 'webanno.custom.ORLRelation':
                rel_items.append(item) 
            elif item['%TYPE'] == 'webanno.custom.ORLSpan':
                item_id = int(item['%ID'])
                
                if item['Roles2'] == 'SubExp':
                    senti_expr_items[item_id] = item
                elif item['Roles2'] == 'Target':
                    target_items[item_id] = item
                elif item['Roles2'] == 'Source':
                    source_items[item_id] = item
        
        return senti_expr_items, target_items, source_items, rel_items

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--anno_root", help="root directory of the annotation json files") # ../data/inception/
    arg_parser.add_argument("-o", "--csv_path", help="path of csv file to which the annotation dataframe should be written") # ../output/UNSC_2014_SPV.7154_sentsplit.csv
    args = arg_parser.parse_args()
    
    corpus_reader = InceptionCorpusReader(args.anno_root)
    corpus_reader.items_df_to_csv(args.csv_path)