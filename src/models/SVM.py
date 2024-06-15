from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class SVMClassifier:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='scale', coef0=0.0, verbose=True):
        self.model = make_pipeline(StandardScaler(),
                                   SVC(C=C, kernel=kernel,
                                       degree=degree, gamma=gamma, 
                                       coef0=coef0, verbose=verbose,
                                       ))
    
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
    
    # SVM Classifier
    svm_model = SVMClassifier()
    svm_model.fit(x_train, y_train)
    x_test = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_test = [0, 1, 0, 1]
    print("SVM Classifier Accuracy:", svm_model.score(x_test, y_test))
    
