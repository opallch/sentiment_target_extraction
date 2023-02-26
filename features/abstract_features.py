from abc import ABC, abstractclassmethod


class AbstractFeatures(ABC):
    
    @abstractclassmethod
    def get_features(self):
        pass

