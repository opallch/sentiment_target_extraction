# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd


class ProdigyCorpusReader:
    
    def __init__(self, anno_dir_path):
        """Constructor of the ProdigyCorpusReader class.

        Args:
            anno_dir_path (str): path of the directory containing the xml
                annotation files created with GATE

        Attributes:
            self.items: pd.DataFrame with sentiment items
        """
        self.items_df = self. _all_jsonls_to_df(anno_dir_path)
    
    def items_df_to_csv(self, f_out:str):
        """writes the annotation items dataframe into a csv file."""
        if f_out.endswith('.csv'):
            self.items_df.to_csv(f_out)
        else:
            raise Exception('please pass a csv filename!')
    
    def _all_jsonls_to_df(self, anno_dir_path):
        """returns a annotation items dataframe for all jsonl files in the given directory."""
        items_df = pd.DataFrame(columns=["sentence",
                                      "sentexprStart",
                                      "sentexprEnd",
                                      "targetStart",
                                      "targetEnd"])
        
    def _jsonl_to_tmp_df(self, jsonl) -> pd.DataFrame:
        """returns a temporary annotation items dataframe for one single jsonl file."""
        tmp_rows = []

        with open(jsonl, 'r') as jsonl_in:
            for line in jsonl_in:
                tmp_rows.extend(
                    [row for row in self._jsonl_str_to_rows(line)]
                    )
        
        return pd.DataFrame(tmp_rows)

    def _jsonl_str_to_rows(self, line) -> list:
        """converts a line in jsonl file to a list of annotation items."""
        tmp_rows = []
        json_obj = json.loads(line)
        sentence = json_obj['text']

        for rel in json_obj['relations']:
            tmp_rows.append([sentence,
                            rel['child_span']['start'],
                            rel['child_span']['end'],
                            rel['head_span']['start'],
                            rel['head_span']['end']
                            ])
        
        return tmp_rows

if __name__ == "__main__":
    reader = ProdigyCorpusReader('anno_files/')
    test_jsonl = '../test_files/ner_rels.jsonl'
    df = reader._jsonl_to_tmp_df(test_jsonl)
    print(df.head(10))