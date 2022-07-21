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
from .feature_utils import parse_sent, InvalidFilenameError, transform_spans


class FeatureVectorCreator:

    def __init__(self, filename_in:str, filename_out:str):
        """Constructor of FeatureVectorCreator.

        Args:
            file_in(str): filename of the `.pkl` or `.csv` file storing the DataFrame from Corpus Reader
            out_file_pkl(str): filename of the `.pkl` file in which the features vectors will be stored

        Attributes:
            self.features_classes(list): list of objects which inherit from features.AbstractFeatures
            self.df_corpus_reader(pd.DataFrame)
            self.out_file_pkl(str): 
            self.list_of_vecs(list): list of feature vectors 
        """
        self._df_corpus_reader = self._load_df(filename_in)
        self._features_classes = [ConstituencyParseFeatures(self._trees()),
                                  DependencyParseFeatures()]
        self._filename_out = filename_out
        self._list_of_vecs = []

    def _load_df(self, filename:str):
        if filename.endswith("pkl"):
            df = pd.read_pickle(filename)
        elif filename.endswith("csv"):
            df =  pd.read_csv(filename)
        else:
            raise InvalidFilenameError("Input file must be .csv or .pkl")
        df[["sentexprStart", "sentexprEnd", "targetStart", "targetEnd"]] = df.apply(
            lambda x: transform_spans(x), axis=1, result_type="expand"
        )
        return df

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
                all_vectors += features
            else:
                return None
        return all_vectors

    def _write_vectors_to_file(self) -> None:
        """Write self.df_vectors to the ouput `.pkl` file."""
        if self._filename_out.endswith("pkl"):
            pd.to_pickle(
            self._list_of_vecs2df(),
            self._filename_out
        )
        elif self._filename_out.endswith("csv"):
            self._list_of_vecs2df().to_csv(self._filename_out)
        else:
            raise InvalidFilenameError("Output file must be .csv or .pkl")

    def _list_of_vecs2df(self) -> pd.DataFrame:
        """Turn `self._list_of_vecs` into a dataframe."""
        return pd.DataFrame(np.array(self._list_of_vecs))

    def _trees(self):
        """Parse all sentences once to avoid double parsing."""
        parser = Parser("benepar_en3")
        return {
            sent: parse_sent(sent, parser)
            for sent in self._df_corpus_reader["sentence"].unique()
        }
