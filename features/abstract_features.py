from abc import ABC, abstractclassmethod
# in: Dataframe
# out: feature vectors in DataFrame .pkl

"""
Features:
1) Relational 
2) Word + Syntactic
3) Dependency and Constituency Parse Information

challenge:
1) how can we parse -> create features
"""

class AbstractFeatures(ABC):
    
    @abstractclassmethod
    def get_features(self):
        pass

