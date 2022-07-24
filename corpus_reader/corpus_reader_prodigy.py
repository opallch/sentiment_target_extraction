# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import pandas as pd


class ProdigyCorpusReader:
    
    def __init__(self, prodigy_anns_root):
        """Constructor of the ProdigyCorpusReader class.

        Args:
            prodigy_anns_root (str): path of the directory containing the jsonl
                annotation files created with Prodigy

        Attributes:
            self.items: pd.DataFrame with sentiment items
        """
        self.anns_files = self._find_all_anns_files(prodigy_anns_root)
        self.items_df = self. _all_jsonls_to_df()
    
    def _find_all_anns_files(self, prodigy_anns_root):
        return [os.path.join(root, filename)
                for root, _, filenames in os.walk(prodigy_anns_root) 
                for filename in filenames if filename.endswith(".jsonl") 
                ]
    
    def items_df_to_csv(self, f_out:str):
        """writes the annotation items dataframe into a csv file."""
        if f_out.endswith('.csv'):
            self.items_df.to_csv(f_out)
        else:
            raise Exception('please pass a csv filename!')
    
    def _all_jsonls_to_df(self):
        """returns a annotation items dataframe for all jsonl files in the given directory."""
        items_df = pd.DataFrame(columns=["sentence",
                                      "sentexprStart",
                                      "sentexprEnd",
                                      "targetStart",
                                      "targetEnd"])
        
        for jsonl_file in self.anns_files:
            items_df = pd.concat([items_df, 
                                self._jsonl_to_tmp_df(jsonl_file)],
                                ignore_index=True
                                )
        
        return items_df
        
    def _jsonl_to_tmp_df(self, jsonl) -> pd.DataFrame:
        """returns a temporary annotation items dataframe for one single jsonl file."""
        tmp_rows = []

        with open(jsonl, 'r') as jsonl_in:
            for line in jsonl_in:
                tmp_rows.extend(
                    [row for row in self._jsonl_str_to_rows(line)]
                    )
        
        return pd.DataFrame(tmp_rows,
                            columns=["sentence",
                                      "sentexprStart",
                                      "sentexprEnd",
                                      "targetStart",
                                      "targetEnd"]
                            )

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
    reader = ProdigyCorpusReader('../test_files/')
    df = reader._all_jsonls_to_df()
    print(df.head(10))