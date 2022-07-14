from sklearn.tree import DecisionTreeClassifier
from abstract_classifier import AbstractClassifier

class DTreeClassifier(AbstractClassifier):
    
    def __init__(self, pkl_file, test_percentage, random_state):
        super().__init__(pkl_file, test_percentage, random_state)
        self.clf = DecisionTreeClassifier() # need to specify other params

    def train(self):
        self.clf.fit(self.X_train, self.y_train)
    
    def predict(self):
        return self.clf.predict(self.X_test)

if __name__ == "__main__":
    dtree = DTreeClassifier("../test_files/vec.pkl")
    dtree.train()
    dtree.save_pred(
        dtree.predict,
        "../test_files/y_pred_dtree.pkl"
    )
    dtree.save_gold("../test_files/y_gold_dtree.pkl")