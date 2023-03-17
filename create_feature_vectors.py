# -*- coding: utf-8 -*-
"""
Class creating features vectors for all instances from the corpus, which has been preprocessed by `corpus_reader.py`.
All features vectors will be stored as a DataFrame, which will be written into a file. 
"""
import warnings

import numpy as np
import pandas as pd
from benepar import Parser
from tqdm import tqdm

from features.constituency_features import ConstituencyParseFeatures
from features.dependency_features import DependencyParseFeatures
from features.word_embedding import WordEmbeddingFeatures
from features.feature_utils import *
tqdm.pandas()
warnings.filterwarnings("ignore")


class FeatureVectorCreator:

    PARSER = Parser("benepar_en3")

    def __init__(self, items_df_path:str, filepath_out:str, feature_classes, undersample=True):
        """Constructor of FeatureVectorCreator.

        Args:
            items_df_path(str): filename of the `.pkl` or `.csv` file storing the
                DataFrame from Corpus Reader
            filepath_out(str): filename of the `.pkl` file in which the
                features vectors will be stored

        Attributes:
            self.features_classes(list): list of objects which inherit from
                features.AbstractFeatures
            self.df_corpus_reader(pd.DataFrame)
            self.out_file_pkl(str): 
            self.list_of_vecs(list): list of feature vectors 
        """
        self._filepath_out = filepath_out
        self._undersample = undersample
        self.items_df = self._load_df(items_df_path)
        
        self._trees = self._get_trees()
        
        # for ablation
        self._features_classes = []
        if 'constituency' in feature_classes:
            self._features_classes.append(ConstituencyParseFeatures(self._trees))
        if 'dependency' in feature_classes:
            self._features_classes.append(DependencyParseFeatures())
        if 'word2vec' in feature_classes:
            self._features_classes.append(WordEmbeddingFeatures('./output/word2vec.model', './data/inception/raw_text/'))

        self.items_df = self._add_negative_instances()
        # temporary         
        self._list_of_vecs = []
        self._labels = []
        
    def _load_df(self, filename:str):
        if filename.endswith(".pkl"):
            df = pd.read_pickle(filename)
        elif filename.endswith(".csv"):
            df =  pd.read_csv(filename, encoding="utf-8")
        else:
            raise InvalidFilenameError("Input file must be .csv or .pkl")
        df = df.drop(columns=['sourceStart', 'sourceEnd'])

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

    def _add_negative_instances(self, n="all"):
        """Add negative instances for classifier training to dataframe."""
        # assign positive label to gold instances so far
        self.items_df["label"] = 1
        # collect negative instances
        negative_instances = pd.concat(
            list(
                self.items_df.apply(
                    lambda x: self._add_negative_instances_to_row(x),
                    axis=1
                )
            ),
            axis=0,
            ignore_index=True
        )
        # if self._undersample is True, reduce number of negative instances
        if self._undersample and (len(self.items_df) < len(negative_instances)):
            negative_instances = negative_instances.sample(len(self.items_df))
        # return dataframe with negative instances added
        return pd.concat([self.items_df, negative_instances],
                         axis=0,
                         ignore_index=True)

    def _add_negative_instances_to_row(self, df_row):
        """Create negative instances for classifier training."""
        tree = self._trees[df_row["sentence"]]
        candidate_items = []
        for c in get_candidates(tree):
            candidate_items.append(
                self._create_candidate_row(df_row, c)
            )
        df = pd.DataFrame(candidate_items, columns=df_row.index)
        return df

    @staticmethod
    def _create_candidate_row(df_row, candidate):
        """Return candidate as item."""
        target_start_char_span, target_end_char_span =  \
                token_span_to_char_span(df_row, candidate.span_start(), candidate.span_end())
        
        return (
            df_row['Unnamed: 0'],
            df_row['rawTextFilename'],
            df_row['sentenceID'],
            df_row["sentence"],  # sentence
            df_row["sentexprStart"],  # senti expr start
            df_row["sentexprEnd"],  # senti expr end
            target_start_char_span,  # target cadidate start
            target_end_char_span,  # target candidate end 
            0  # label
        )

    def _all_features_for_all_instances(self) -> None:
        """Append the instance features lists to self._list_of_vecs."""
        print("Creating vectors...")
        for idx in range(0,len(self.items_df)):
            try:
                item = self.items_df.iloc[idx]
                self._list_of_vecs.append(self._all_features_for_each_instance(item))
                self._labels.append(item['label'])
            except NotATargetRelationError:
                pass
                # print(f'Row {idx} is skipped, since it is not a sentiment-expression-to-target (probably a sentiment-expression-to-source relation instead).')
                print(e)
            except KeyError as e:
                print(e)
            except TargetHeadNotFoundError:
                print('TargetHeadNotFoundError')
                print(item.sentence)
                print(item.sentence[item.targetStart:item.targetEnd])
            except SentiExprHeadNotFoundError:
                print('SentiExprHeadNotFoundError')
                print(item.sentence)
                print(item.sentence[item.sentexprStart:item.sentexprEnd])
            finally:
                continue

    def _append_vector_to_list(self, features_vec:list, label) -> list:
        """Append the features lists for an instance to `self._list_of_vecs`"""
        if features_vec is not None:
            self._list_of_vecs.append(features_vec)
            self._labels.append(label)

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

    def _list_of_vecs2df(self) -> pd.DataFrame:
        """Turn `self._list_of_vecs` into a dataframe."""
        vectors_df = pd.DataFrame(np.array(self._list_of_vecs))
        vectors_df['label'] = np.array(self._labels, dtype=np.int64)
        return vectors_df

    def _get_trees(self):
        """Parse all sentences once to avoid double parsing."""
        parser = self.PARSER
        print("Parsing sentences...")
        parses_dict = {
            sent: parse_sent(sent, parser)
            for sent in tqdm(self.items_df["sentence"].unique())
        }
        #print(parses_dict["Prime Minister Yatsenyuk went to Donetsk; although he did not meet with the protesters, he did say the right things about dialogue, decentralization and his desire to find a way out of the problem."])
        return parses_dict

    def _write_vectors_to_file(self) -> None:
        """Write self.df_vectors to the ouput `.pkl` file."""
        df = self._list_of_vecs2df()

        if self._filepath_out.endswith(".pkl"):
            pd.to_pickle(
                df,
                self._filepath_out
            )
        elif self._filepath_out.endswith(".csv"):
            df.to_csv(self._filepath_out, encoding="utf-8")
        else:
            raise InvalidFilenameError("Output file must be .csv or .pkl")

if __name__ == "__main__":
    features_creator = FeatureVectorCreator(items_df_path="./output/UNSC_2014_SPV.7154_sentsplit.csv", 
                                            filepath_out="./output/instances/UNSC_2014_SPV.7154_sentsplit_instances.csv", 
                                            undersample=True)
    features_creator.get_vectors()