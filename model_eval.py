# -*- coding: utf-8 -*-
"""
Functions for training of classifiers and classification data.
"""
import pandas as pd

from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

from create_feature_vectors import FeatureVectorCreator


if __name__ == "__main__":
    df = pd.read_csv("output/instances/UNSC_2014_SPV.7154_sentsplit_instances.csv")

    ## (1) Prepare feature vectors
    # Get rid of columns we don't need for classification
    labels = df.label.values
    df_selected = df.drop(["Unnamed: 0", "label"],
                          axis=1)
    feature_vecs = df_selected.to_numpy()  # or df_selected.values

    dummy_clf = DummyClassifier(strategy='most_frequent') # alternative: 'prior', 'stratified', 'uniform'
    model = svm.SVC()
    
    cv_results_model = cross_validate(
        model,
        feature_vecs,
        labels,
        cv=10,
        scoring=['f1_weighted', 'f1_micro', 'f1_macro']
    )

    cv_results_dummy = cross_validate(
        dummy_clf,
        feature_vecs,
        labels,
        cv=10,
        scoring=['f1_weighted', 'f1_micro', 'f1_macro']
    )

    with open('./result.txt', 'w') as f_out:
        print('svm:', file=f_out)
        for key, value in cv_results_model.items():
            print(value, '\t', key, file=f_out)
        
        print('majority class:', file=f_out)
        for key, value in cv_results_dummy.items():
            print(value, '\t', key, file=f_out)