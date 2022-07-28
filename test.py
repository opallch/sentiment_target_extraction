# -*- coding: utf-8 -*-
import os

import pandas as pd

from classifier.features.create_feature_vectors import FeatureVectorCreator
#from corpus_reader.corpus_reader_prodigy import ProdigyCorpusReader

from model import train_classifier, classify, evaluate


TRAINING_DATA = "test_files/items_neu.pkl"
TRAINING_LABELS = "test_files/neu_labels.pkl"
ITEMS = "test_files/unsc_items.csv"
INSTANCE_FILE = "test_files/unsc_instances.pkl"
LABELS = "test_files/unsc_labels.pkl"

MODEL = "test_files/mpqa_model.pkl"
TEST_MPQA_INSTANCES = "test_files/instances_mpqa_test.pkl"
TEST_MPQA_LABELS = "test_files/label_mpqa_test.pkl"

fvc = FeatureVectorCreator(ITEMS, INSTANCE_FILE, LABELS)
fvc.get_vectors()

X_train = pd.read_pickle(TRAINING_DATA)
y_train = pd.read_pickle(TRAINING_LABELS)

clf = train_classifier(X_train, y_train, filename=MODEL, test_filename="test_files/mpqa_test.pkl")

mpqa_gold = pd.read_pickle(TEST_MPQA_LABELS)
unsc_gold = pd.read_pickle(LABELS)

mpqa_pred = classify(TEST_MPQA_INSTANCES, clf)
unsc_pred = classify(INSTANCE_FILE, clf)

print("------------------------------------------------------------------")
print("Testing on MPQA test data:")
evaluate(mpqa_gold, mpqa_pred)
print("Testing on UNSC test data:")
evaluate(unsc_gold, unsc_pred)
