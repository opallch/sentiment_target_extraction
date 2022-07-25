# -*- coding: utf-8 -*-
import os

#import pandas as pd

#from classifier.features.create_feature_vectors import FeatureVectorCreator
from corpus_reader.corpus_reader_GATE import GATECorpusReader


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

print("hi")

anno_dir_path = "mpqa_corpus/gate_anns"
text_dir_path = "mpqa_corpus/docs"
corpus_reader = GATECorpusReader(anno_dir_path, text_dir_path)
corpus_reader.items.to_pickle("./test_files/items_neu.pkl")

print(corpus_reader.items)

