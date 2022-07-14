from sklearn.svm import SVC
from abstract_classifier import AbstractClassifier

class SVClassifier(AbstractClassifier):

    def __init__(self, pkl_file, test_percentage, random_state):
        super().__init__(pkl_file, test_percentage, random_state)
        self.clf = SVC() # need to specify other params

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
    
    def predict(self):
        return self.clf.predict(self.X_test)

if __name__ == "__main__":
    svm = SVClassifier("../test_files/vec.pkl")
    svm.train()
    svm.save_pred(
        svm.predict,
        "../test_files/y_pred_svm.pkl"
    )
    svm.save_gold("../test_files/y_gold_svm.pkl")