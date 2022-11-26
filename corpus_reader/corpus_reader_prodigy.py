# -*- coding: utf-8 -*-
"""
Reader for prodigy annotated files.
"""
import argparse
import os
import json

import pandas as pd


class ProdigyCorpusReader:

    def __init__(self, prodigy_anns_root):
        """Constructor of the ProdigyCorpusReader class.

        Args:
            prodigy_anns_root (str): path of the directory containing the jsonl
                annotation files created with Prodigy

        Attributes:
            self.anns_files: list of all annotation files in directory
            self.items: pd.DataFrame with sentiment items
        """
        self.anns_files = self._find_all_anns_files(prodigy_anns_root)
        self.items_df = self. _all_jsonls_to_df()

    def _find_all_anns_files(self, prodigy_anns_root):
        return [
            os.path.join(root, filename)
            for root, _, filenames in os.walk(prodigy_anns_root)
            for filename in filenames if filename.endswith(".jsonl")
        ]

    def items_df_to_csv(self, f_out:str):
        """Write the annotation items dataframe into a csv file."""
        if f_out.endswith('.csv'):
            self.items_df.to_csv(f_out)
        else:
            raise Exception('please pass a csv filename!')

    def _all_jsonls_to_df(self):
        """Return annotation dataframe for all jsonl files of directory."""
        items_df = pd.DataFrame(columns=["sentence",
                                      "sentexprStart",
                                      "sentexprEnd",
                                      "targetStart",
                                      "targetEnd"])

        for jsonl_file in self.anns_files:
            items_df = pd.concat([items_df, self._jsonl_to_tmp_df(jsonl_file)],
                                 ignore_index=True)
        return items_df
        
    def _jsonl_to_tmp_df(self, jsonl) -> pd.DataFrame:
        """Return a temporary annotation dataframe for a single jsonl file."""
        tmp_rows = []
        # read in jsonl file linewise
        with open(jsonl, 'r') as jsonl_in:
            for line in jsonl_in:
                tmp_rows.extend([row for row in self._jsonl_str_to_rows(line)])
        return pd.DataFrame(tmp_rows,
                            columns=["sentence",
                                     "sentexprStart",
                                     "sentexprEnd",
                                     "targetStart",
                                     "targetEnd"])

    def _jsonl_str_to_rows(self, line) -> list:
        """Convert a line in jsonl file to a list of annotation items."""
        tmp_rows = []
        json_obj = json.loads(line)
        sentence = json_obj['text']

        for rel in json_obj['relations']:
            tmp_rows.append([sentence,
                             rel['child_span']['start'],
                             rel['child_span']['end'],
                             rel['head_span']['start'],
                             rel['head_span']['end']])
        return tmp_rows

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-a", "--anno_root", help="root directory of the annotation jsonl files") # ../data/unsc_prodigy/
    arg_parser.add_argument("-o", "--csv_path", help="path of csv file to which the annotation dataframe should be written") # ../output/unsc.csv
    args = arg_parser.parse_args()
    
    reader = ProdigyCorpusReader(args.anno_root)
    reader.items_df_to_csv(args.csv_path)