# -*- coding: utf-8 -*-
"""
Functions for training of classifiers and classification data.
"""
import pandas as pd
from sklearn import svm
from sklearn.model_selection import cross_validate


if __name__ == "__main__":
    df = pd.read_csv("output/instances/test_all_features.csv")

    ## (1) Prepare feature vectors
    # Get rid of columns we don't need for classification
    labels = df.label.values
    df_selected = df.drop(["Unnamed: 0", "label"],
                          axis=1)
    feature_vecs = df_selected.to_numpy()  # or df_selected.values

    model = svm.SVC()
    cv_results = cross_validate(
        model,
        feature_vecs,
        labels,
        cv=10,
        scoring=['f1_weighted', 'f1_micro', 'f1_macro']
    )

    for key, value in cv_results.items():
        print(value, '\t', key)