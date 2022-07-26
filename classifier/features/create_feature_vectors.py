# -*- coding: utf-8 -*-
"""
Class creating features vectors for all instances from the corpus, which has been preprocessed by `corpus_reader.py`.
All features vectors will be stored as a DataFrame, which will be written into a file. 
"""

import random
import warnings

import numpy as np
import pandas as pd
from benepar import Parser
from tqdm import tqdm

from .constituency_features import ConstituencyParseFeatures
from .dependency_features import DependencyParseFeatures
from .feature_utils import parse_sent, InvalidFilenameError, transform_spans, get_candidates

tqdm.pandas()
warnings.filterwarnings("ignore")


class FeatureVectorCreator:

    PARSER = Parser("benepar_en3")

    def __init__(self, filename_in:str, filename_out:str, label_filename="", undersample=False):
        """Constructor of FeatureVectorCreator.

        Args:
            file_in(str): filename of the `.pkl` or `.csv` file storing the
                DataFrame from Corpus Reader
            out_file_pkl(str): filename of the `.pkl` file in which the
                features vectors will be stored
            label_filename (str): filename of gold instance label file, if one
                is to be created

        Attributes:
            self.features_classes(list): list of objects which inherit from
                features.AbstractFeatures
            self.df_corpus_reader(pd.DataFrame)
            self.out_file_pkl(str): 
            self.list_of_vecs(list): list of feature vectors 
        """
        self._undersample = undersample
        self._label_filename = label_filename
        self._df_corpus_reader = self._load_df(filename_in)
        self._trees = self._get_trees()
        if label_filename:
            self._df_corpus_reader = self._add_negative_instances()
        self._features_classes = [ConstituencyParseFeatures(self._trees),
                                  DependencyParseFeatures()]
        self._filename_out = filename_out
        self._list_of_vecs = []

    def _load_df(self, filename:str):
        if filename.endswith("pkl"):
            df = pd.read_pickle(filename)
        elif filename.endswith("csv"):
            df =  pd.read_csv(filename, encoding="utf-8")
        else:
            raise InvalidFilenameError("Input file must be .csv or .pkl")
        # transform character spans to word level spans
        df[["sentexprStart", "sentexprEnd", "targetStart", "targetEnd"]] = df.apply(
            lambda x: transform_spans(x), axis=1, result_type="expand"
        )
        # return dataframe with just the columns that are needed here
        return df[["sentence", "sentexprStart", "sentexprEnd", "targetStart", "targetEnd"]]

    def get_vectors(self) -> None:
        """Wrapper function for handling the feature vectors.

        Functionalities:
        - create features vectors
        - save them in a dataframe
        - write them into a `.pkl` file
        """
        self._all_features_for_all_instances()
        self._write_vectors_to_file()

    def _add_negative_instances(self):
        """Add negative instances for classifier training to dataframe."""
        # assign positive label to gold instances so far
        self._df_corpus_reader["label"] = 1
        # collect negative instances
        negative_instances = self._df_corpus_reader.apply(
            lambda x: self._add_negative_instances_to_row(x),
            axis=1
        )
        # return dataframe with negative instances added
        return pd.concat([self._df_corpus_reader] + list(negative_instances),
                         axis=0,
                         ignore_index=True)

    def _add_negative_instances_to_row(self, df_row):
        """Create negative instances for classifier training."""
        tree = self._trees[df_row["sentence"]]
        # collect candidates
        candidates = get_candidates(df_row["sentence"], tree)
        candidates = [
            self._create_candidate_row(df_row, c) for c in candidates
            if not ((c.span_start() == df_row["targetStart"] - 1) and (c.span_end() == df_row["targetEnd"] - 1))
        ]
        if self._undersample:
            candidates = random.sample(candidates, len(self._df_corpus_reader))
        # return as dataframe
        df = pd.DataFrame(candidates, columns=df_row.index)
        return df

    @staticmethod
    def _create_candidate_row(df_row, candidate):
        """Return candidate as item."""
        return (
            df_row["sentence"],  # sentence
            df_row["sentexprStart"],  # senti expr start
            df_row["sentexprEnd"],  # senti expr end
            candidate.span_start(),  # target cadidate start
            candidate.span_end() + 1,  # target candidate end
            0  # label
        )

    def _all_features_for_all_instances(self) -> None:
        """Append the instance features lists to self._list_of_vecs."""
        print("Creating vectors...")
        self._df_corpus_reader.progress_apply(
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

    def _write_labels_to_file(self):
        """Save labels in .pkl or .csv file."""
        if self._label_filename.endswith("pkl"):
            pd.to_pickle(
                self._df_corpus_reader["label"],
                self._label_filename
            )
        elif self._label_filename.endswith("csv"):
            self._df_corpus_reader["label"].to_csv(self._filename_out, encoding="utf-8")
        else:
            raise InvalidFilenameError("Label file must be .csv or .pkl")

    def _write_vectors_to_file(self) -> None:
        """Write self.df_vectors to the ouput `.pkl` file."""
        # if labels are to be saved:
        if self._label_filename:
            self._write_labels_to_file()
        # save feature vectors
        if self._filename_out.endswith("pkl"):
            pd.to_pickle(
            self._list_of_vecs2df(),
            self._filename_out
        )
        elif self._filename_out.endswith("csv"):
            self._list_of_vecs2df().to_csv(self._filename_out, encoding="utf-8")
        else:
            raise InvalidFilenameError("Output file must be .csv or .pkl")

    def _list_of_vecs2df(self) -> pd.DataFrame:
        """Turn `self._list_of_vecs` into a dataframe."""
        return pd.DataFrame(np.array(self._list_of_vecs))

    def _get_trees(self):
        """Parse all sentences once to avoid double parsing."""
        parser = self.PARSER
        print("Parsing sentences...")
        return {
            sent: parse_sent(sent, parser)
            for sent in tqdm(self._df_corpus_reader["sentence"].unique())
        }
