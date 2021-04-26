import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=1, epochs=100):
        self.W = np.zeros(input_size+1)
        # add one for bias
        self.epochs = epochs
        self.lr = lr
    
    def activation_fn(self, x):
        #return (x >= 0).astype(np.float32)
        return 1 if x >= 0 else 0
 
    def predict(self, x):
        z = self.W.T.dot(x)
        a = self.activation_fn(z)
        return a
 
    def fit(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.W = self.W + self.lr * e * x
    
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if(target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_



# X = np.array([
#         [0, 0],
#         [0, 1],
#         [1, 0],
#         [1, 1]
#     ])
# d = np.array([0, 0, 0, 1])
 
# perceptron = Perceptron(input_size=2)
# perceptron.fit(X, d)
# print(perceptron.W)




# Load the data set
bc = datasets.load_breast_cancer()
X = bc.data
y = bc.target
# Create training and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
# Instantiate CustomPerceptron
prcptrn = Perceptron(input_size=2)
# Fit the model
prcptrn.fit(X_train, y_train)
# Score the model
print(prcptrn.score(X_test, y_test), prcptrn.score(X_train, y_train))



model = LinearRegression()
model.fit(X_train, y_train)
print(score(X_test, y_test), score(X_train, y_train))