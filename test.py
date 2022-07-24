# -*- coding: utf-8 -*-
import os

import pandas as pd

from classifier.features.create_feature_vectors import FeatureVectorCreator


ROOT = os.path.dirname(os.path.abspath(__file__))
ITEMS = os.path.join(ROOT, "test_files/items.pkl")
TEST_ALL_FEATURES = os.path.join(ROOT, "test_files/test_all_features.pkl")

fvc = FeatureVectorCreator(ITEMS, TEST_ALL_FEATURES)
fvc.get_vectors()

# to check if the dataframe looks good
df = pd.read_pickle(TEST_ALL_FEATURES)
print(df[:10])
print(df.columns)
print(df[240].value_counts())
