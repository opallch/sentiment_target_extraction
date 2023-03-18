import os
import pickle
import pandas as pd 
import numpy as np

from create_feature_vectors import *
from features.constituency_features import *
from features.dependency_features import *
from features.word_embedding import *
from features.feature_utils import *

FEATURE_CLASSES =  ['constituency', 'dependency']

MODEL_DIR = './output/'
MODEL_FILENAME = f'{"_".join(FEATURE_CLASSES)}_svm.pkl'
DF_PATH = 'output/UNSC_2014_SPV.7154_sentsplit.csv'

# load model
with open(os.path.join(MODEL_DIR, MODEL_FILENAME), 'rb') as f_in:
        svm_model = pickle.load(f_in)

# load DataFrame
df = pd.read_csv(DF_PATH)

vecs_creator = FeatureVectorCreator(FEATURE_CLASSES)

for idx, row in df.iterrows():
        if row.targetStart == -1 or row.targetEnd == -1:
                continue

        sentence = row.sentence
        # create candidates, S/NP
        candidates = get_candidates(vecs_creator.trees[sentence])
        # TODO choose the most probable candidate
        arg_max = None
        max_prob = np.NINF
        for c in candidates:
                candidate_row = pd.DataFrame([
                        vecs_creator.create_candidate_row(row, c)[:-1] + (-1, -1)# create row without label, but with sourceStart and sourceEnd
                        ], columns=row.index) 
                vec = [vecs_creator.all_features_for_each_instance(candidate_row.iloc[0])]
                prob = svm_model.predict_log_proba(vec)
                
                if prob[0][0] > max_prob:
                        max_prob = prob[0][0]
                        arg_max = c
        
        ## Display
        print(f'Sentence: {sentence}')
        # display sentiment expr
        print(f'Sentiment Expression: {sentence[row.sentexprStart:row.sentexprEnd]}')
        # TODO display most probable target (predicted)
        print(f'Predicted Target: {arg_max}')
        # display gold
        print(f'Gold Target: {sentence[row.targetStart:row.targetEnd]}')