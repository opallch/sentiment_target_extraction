# -*- coding: utf-8 -*-
"""
Class creating features vectors for all instances from the corpus, which has been preprocessed by `corpus_reader.py`.
All features vectors will be stored as a DataFrame, which will be written into a file. 
"""

import numpy as np
import pandas as pd
from benepar import Parser

from .constituency_features import ConstituencyParseFeatures
from .dependency_features import DependencyParseFeatures
from .feature_utils import parse_sent


class FeatureVectorCreator:

    def __init__(self, in_file_pkl:str, out_file_pkl:str):
        """Constructor of FeatureVectorCreator.

        Args:
            in_file_pkl(str): filename of the `.pkl` file storing the DataFrame from Corpus Reader
            out_file_pkl(str): filename of the `.pkl` file in which the features vectors will be stored

        Attributes:
            self.features_classes(list): list of objects which inherit from features.AbstractFeatures
            self.df_corpus_reader(pd.DataFrame)
            self.out_file_pkl(str): 
            self.list_of_vecs(list): list of feature vectors 
        """
        self._df_corpus_reader = pd.read_pickle(in_file_pkl)
        self._features_classes = [ConstituencyParseFeatures(self._trees()),
                                  DependencyParseFeatures()]
        self._out_file_pkl = out_file_pkl
        self._list_of_vecs = []

    def get_vectors(self) -> None:
        """Wrapper function for handling the feature vectors.

        Functionalities:
        - create features vectors
        - save them in a dataframe
        - write them into a `.pkl` file
        """
        self._all_features_for_all_instances()
        self._write_vectors_to_file()

    def _all_features_for_all_instances(self) -> None:
        """Append the instance features lists to self._list_of_vecs."""
        # for i in range (0, len(self._df_corpus_reader)):
            # self._append_vector_to_list(
                # self._all_features_for_each_instance(
                    # self._df_corpus_reader.iloc[i]
                # )
            # )
        self._df_corpus_reader.apply(
            lambda x: self._append_vector_to_list(
                self._all_features_for_each_instance(x),
            ),
            axis=1
        )

    def _append_vector_to_list(self, features_vec:list) -> list:
        """Append the features lists for an instance to `self._list_of_vecs`"""
        if features_vec is not None:
            self._list_of_vecs.append(features_vec)

    def _all_features_for_each_instance(self, df_row:pd.Series) -> list:
        """Combine all features vectors from different feature classes."""
        all_vectors = []
        # collect vectors
        for features_class in self._features_classes:
            features = features_class.get_features(df_row)
            if features is not None:
                all_vectors.append(features)
            else:
                return None
        return [f for vec in all_vectors for f in vec]

    def _write_vectors_to_file(self) -> None:
        """Write self.df_vectors to the ouput `.pkl` file."""
        pd.to_pickle(
            self._list_of_vecs2df(),
            self._out_file_pkl
        )

    def _list_of_vecs2df(self) -> pd.DataFrame:
        """Turn `self._list_of_vecs` into a dataframe."""
        return pd.DataFrame(np.array(self._list_of_vecs))

    def _trees(self):
        """Parse all sentences once to avoid double parsing."""
        parser = Parser("benepar_en3")
        trees = {}
        for sent in self._df_corpus_reader["sentence"].unique():
            if sent not in trees:
                trees[sent] = parse_sent(sent, parser)
        return trees


if __name__ == "__main__":
    fvc = FeatureVectorCreator("../test_files/items.pkl", "../test_files/test_all_features.pkl")
    fvc.get_vectors()

    # to check if the dataframe looks good
    print(pd.read_pickle("../test_files/test_all_features.pkl")[:10])