import os

import pandas as pd
from gensim.models import Word2Vec
import spacy

from features.abstract_features import AbstractFeatures
from features.feature_utils import *
from nltk.tokenize import word_tokenize


class WordEmbeddingFeatures(AbstractFeatures):

    def __init__(self, word2vec_model_path, raw_text_files_root):
        
        self.word2vec_model = Word2Vec.load(word2vec_model_path) if os.path.exists(word2vec_model_path) \
            else self._train_model(word2vec_model_path, raw_text_files_root)

        self.nlp = spacy.load('en_core_web_sm')


    def get_features(self, df_row):
        # features to be implemented: sentiment target head word embedding
        # find the head token of target expression using _find_head()
        if df_row.targetStart == -1 or df_row.targetEnd == -1:
            raise NotATargetRelationError
        
        tokens = self.nlp(df_row.sentence)
        _, _, target_token_span_start, target_token_span_end = char_span_to_token_span(df_row, tokenize_func=self.nlp)
        
        head = self._find_head(tokens[target_token_span_start:target_token_span_end])
        
        vector = self.word2vec_model.wv[head.text]
        return list(vector)


    def _train_model(self, word2vec_model_output_path, raw_text_files_root):
        tokenized_sentences = self._get_tokenized_sentences(raw_text_files_root)
        model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
        model.save(word2vec_model_output_path) 

        return model
    

    def _get_tokenized_sentences(self, raw_text_files_root):
        tokenized_sentences = []

        for root, _, filenames in os.walk(raw_text_files_root):
            for filename in filenames:
                if filename.endswith(".txt"):
                    with open(os.path.join(root, filename)) as f_in:
                        for line in f_in:
                              tokens = word_tokenize(line)
                              if len(tokens) > 0:
                                tokenized_sentences.append(tokens)
        return tokenized_sentences

    def _find_head(self, tokens_in_phrase):
        """It returns the head of the a phrase.
        Args:
             tokens_in_phrase (list of spacy.Token): list of tokens of a given phrase i.e. sentiment expression/target
         Returns:
             head(spacy.Token): head of the phrase
        """
        # for one-token-phrase
        if len(tokens_in_phrase) == 1:
            return tokens_in_phrase[0]
        # for phrase of multiple tokens
        else:
            for token in tokens_in_phrase:
                if token.head not in tokens_in_phrase or token.dep_ == 'ROOT':
                    return token
        
        raise SpansError