from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0):
        self.model = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, coef0=coef0))
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
if __name__ == "__main__":
    model = SVM()
    x_train = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_train = [0, 1, 0, 1]
    model.fit(x_train, y_train)
    x_test = [[1, 2], [2, 3], [3, 4], [4, 5]]
    y_test = [0, 1, 0, 1]
    print(model.score(x_test, y_test))