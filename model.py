# decision tree
# svm
import pandas as pd
import numpy as np

from benepar import Parser
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

from features.feature_utils import *
from features.constituency_features import ConstituencyParseFeatures
from features.dependency_features import DependencyParseFeatures


def train_classifier(x, y):
    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_resample(x, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25
    )
    print("Länge x train:", len(X_train))
    clf = svm.SVC()
    #clf = DecisionTreeClassifier(max_depth=5, random_state = 42)
    clf.fit(X_train, y_train)
    classify(X_test, y_test, clf)


def classify(X_test, y_test, clf):
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("precision:", precision)
    print("recall:", recall)
    print("accuracy:", accuracy)
    lol = np.random.choice([0, 1], size=(len(y_test),))
    pr = precision_score(y_test, lol)
    r = recall_score(y_test, lol)
    a = accuracy_score(y_test, lol)
    print("precision:", pr)
    print("recall:", r)
    print("accuracy:", a)


if __name__ == "__main__":
    tr_data = pd.read_csv("test_files/tr_data_cd.csv")
    print(tr_data)
    gold = tr_data.pop("gold")
    train_classifier(tr_data, gold)
