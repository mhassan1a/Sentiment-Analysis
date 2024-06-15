from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, 
                 min_samples_split=2, min_samples_leaf=1, n_jobs=-1):
        self.model = make_pipeline(StandardScaler(),
                                   RF(n_estimators=n_estimators, 
                                      max_depth=max_depth, 
                                      min_samples_split=min_samples_split, 
                                      min_samples_leaf=min_samples_leaf,
                                      verbose=1,
                                      n_jobs=n_jobs))
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

if __name__ == "__main__":
    # Sample training data
    x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_train = [0, 1, 0, 1]

    x_test = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_test = [0, 1, 0, 1]
    
    rf_model = RandomForestClassifier()
    rf_model.fit(x_train, y_train)
    print("Random Forest Classifier Accuracy:", rf_model.score(x_test, y_test))
