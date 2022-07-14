"""
A classifier class should train the classifier with the data from the feature vectors dataframe,
and write the prediction of test data (ndarray) into a `.pkl` file.
"""
from abc import ABC, abstractclassmethod
from random import shuffle
from numpy import ndarray
import pandas as pd
from sklearn.model_selection import train_test_split

class AbstractClassifier(ABC):
    
    def __init__(self, pkl_file, test_percentage, random_state):
        self.vectors_df = self._load_df(pkl_file)
        self.X, self.y = self._df2arrays()
        
        # dont know if the split make sense if you do e.g. cross-validation
        self.X_train, self.y_train, self.X_test, self.y_test = \
            self.train_test_split(test_percentage, random_state)

    def _load_df(self, pkl_file):
        """loads dataframe from a .pkl file."""
        return pd.read_pickle(pkl_file)
    
    def _df2arrays(self):
        # X, y 
        return self.vectors_df.iloc[:,:-1].values.flatten(), \
                self.vectors_df.iloc[:,-1:].values.flatten()
    
    def train_test_split(self, test_percentage, random_state=7):
        return train_test_split(
                self.X, 
                self.y, 
                test_size=test_percentage,
                random_state=random_state,
                shuffle=True)
    
    @abstractclassmethod
    def train(self):
        pass
    
    @abstractclassmethod
    def predict(self):
        pass
    
    def save_pred(self, y_pred: ndarray, pkl_file:str):
        with open(pkl_file, 'w') as pkl_file:
            pd.to_pickle(y_pred, pkl_file)
    
    def save_gold(self, pkl_file):
        with open(pkl_file, 'w') as pkl_file:
            pd.to_pickle(self.y_test, pkl_file)