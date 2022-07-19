"""Evaluation class which prints different evaluation results (at the moment only) on console."""
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score


class Evaluation:
    
    def __init__(self, y_gold_pkl, y_pred_pkl):
        self.y_gold = pd.read_pickle(y_gold_pkl)
        self.y_pred = pd.read_pickle(y_pred_pkl)
    
    def f1_score(self):
        return f1_score(self.y_gold, self.y_pred)


if __name__ == "__main__":
    eval = Evaluation("../test_files/y_gold_svm.pkl", "../test_files/y_pred_svm.pkl")
    print(eval.f1_score())