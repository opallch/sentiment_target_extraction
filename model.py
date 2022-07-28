# -*- coding: utf-8 -*-
"""
Functions for training of classifiers and classification data.
"""

import os
import pickle

import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from classifier.features.feature_utils import InvalidFilenameError


def train_classifier(X, y, filename="", test_filename="", sampling="rus"):
    # apply sampling
    if sampling == "rus":
        rus = RandomUnderSampler(random_state=42)
        X, y = rus.fit_resample(X, y)
    elif sampling == "ros":
        ros = RandomOverSampler(random_state=42)
        X, y = rus.fit_resample(X, y)
    elif sampling != "all":
        print("Unrecognized sampling method. Choose between 'rus' for "
              "random undersampling, 'ros' for random oversampling or all to "
              "turn off sampling entirely")
    # split data if test_filename has been provided
    if test_filename:
        X, X_test, y, y_test = train_test_split(
            X, y, test_size=0.25
        )
        head, tail = os.path.split(test_filename)
        if tail.endswith("pkl"):
            pd.to_pickle(X_test, os.path.join(head, "instances_" + tail))
            pd.to_pickle(y_test, os.path.join(head, "label_" + tail))
        elif tail.endswith(csv):
            X_test.to_csv(os.path.join(head, "instances_" + tail),
                          encoding="utf-8")
            y_test.to_csv(os.path.join(head, "label_" + tail),
                          encoding="utf-8")
        else:
            print("File name for the test files must end with .pkl or .csv")
    # train classifier
    clf = svm.SVC()
    clf.fit(X, y)
    # save model
    if filename:
        assert filename.endswith("pkl"), "Filename must end with .pkl"
        pickle.dump(clf, open(filename, 'wb'))
    return clf


def load_classifier(model_file):
    """Load model from pkl-file."""
    assert filename.endswith("pkl"), "Filename must end with .pkl"
    clf = pickle.load(open(model_file, 'rb'))
    return clf


def classify(instance_file, clf, out_file=""):
    """Classify instances with given classifier."""
    # read instances from file
    if instance_file.endswith("pkl"):
        X_test = pd.read_pickle(instance_file)
    elif instance_file.endswith("csv"):
        X_test = pd.read_csv(instance_file, encoding="utf-8")
    else:
        raise InvalidFilenameError("Instance file must .csv or .pkl")
    # classify
    y_pred = clf.predict(X_test)
    # save it out_file has been provided
    if out_file:
        if out_file.endswith("pkl"):
            pd.to_pickle(
                y_pred,
                out_file
            )
        elif out_file.endswith("csv"):
            y_pred.to_csv(out_file, encoding="utf-8")
        else:
            raise InvalidFilenameError("Label file must be .csv or .pkl")
    return y_pred


def evaluate(y_true, y_pred):
    """Calculate precision, recall, f1 score and accuracy."""
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print("precision:", precision)
    print("recall:", recall)
    print("f1 score:", f1)
    print("accuracy:", accuracy)


if __name__ == "__main__":
    tr_data = pd.read_csv("test_files/tr_data_cd.csv")
    print(tr_data)
    gold = tr_data.pop("gold")
    train_classifier(tr_data, gold)
