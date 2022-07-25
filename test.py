# -*- coding: utf-8 -*-
import os

#import pandas as pd

#from classifier.features.create_feature_vectors import FeatureVectorCreator
from corpus_reader.corpus_reader_GATE import GATECorpusReader
from corpus_reader.corpus_reader_prodigy import ProdigyCorpusReader


# ROOT = os.path.dirname(os.path.abspath(__file__))
# ITEMS = os.path.join(ROOT, "test_files/items.pkl")
# TEST_ALL_FEATURES = os.path.join(ROOT, "test_files/test_all_features.pkl")

# fvc = FeatureVectorCreator(ITEMS, TEST_ALL_FEATURES)
# fvc.get_vectors()

# # to check if the dataframe looks good
# df = pd.read_pickle(TEST_ALL_FEATURES)
# print(df[:10])
# print(df.columns)
# print(df[240].value_counts())

# Test for Gate Reader

gate_anno_dir_path = "mpqa_corpus/gate_anns"
gate_text_dir_path = "mpqa_corpus/docs"
gate_corpus_reader = GATECorpusReader(gate_anno_dir_path, gate_text_dir_path)
gate_corpus_reader.items.to_pickle("./test_files/items_neu.pkl")
print(gate_corpus_reader.items)

# Test for Prodigy Reader
prodigy_anno_dir_path = "./unsc_corpus" 
prodigy_corpus_reader = ProdigyCorpusReader(prodigy_anno_dir_path)
prodigy_corpus_reader.items_df_to_csv("./test_files/test_prodigy.csv")


