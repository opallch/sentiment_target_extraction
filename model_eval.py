# -*- coding: utf-8 -*-
"""
Functions for training of classifiers and classification data.
"""
import os
import pandas as pd

from sklearn import svm
from sklearn.model_selection import cross_validate
from sklearn.dummy import DummyClassifier

from create_feature_vectors import FeatureVectorCreator


if __name__ == "__main__":
    # create feature vectors
    ITEMS_DF_PATH = "./output/UNSC_2014_SPV.7154_sentsplit.csv"
    INSTANCES_DF_PATH = "./output/instances/UNSC_2014_SPV.7154_sentsplit_instances.csv"
    FEATURE_CLASSES = ['constituency', 'dependency', 'word2vec']
    SCORING=['f1_weighted', 'f1_micro', 'f1_macro']
    K = 10
    RESULT_ROOT = './results/'
    RESULT_FILENAME = f'{K}_fold_result_{"_".join(FEATURE_CLASSES)}.txt'

    if not os.path.exists(RESULT_ROOT): os.mkdir(RESULT_ROOT)

    features_creator = FeatureVectorCreator(items_df_path=ITEMS_DF_PATH, 
                                            filepath_out=INSTANCES_DF_PATH, 
                                            undersample=True,
                                            feature_classes=FEATURE_CLASSES
                        )
    features_creator.get_vectors()

    df = pd.read_csv(INSTANCES_DF_PATH)

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
        cv=K,
        scoring=SCORING
    )

    cv_results_dummy = cross_validate(
        dummy_clf,
        feature_vecs,
        labels,
        cv=K,
        scoring=SCORING
    )

    with open(os.path.join(RESULT_ROOT, RESULT_FILENAME), 'w') as f_out:
        print('svm:', file=f_out)
        for key, value in cv_results_model.items():
            print(value, '\t', key, file=f_out)
        
        print('majority class:', file=f_out)
        for key, value in cv_results_dummy.items():
            print(value, '\t', key, file=f_out)