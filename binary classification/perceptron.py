# https://analyticsindiamag.com/perceptron-is-the-only-neural-network-without-any-hidden-layer/

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



class Perceptron(object):
    def __init__(self, learning_rate=0.01, n_iter=100, random_state=1):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rand = np.random.RandomState(self.random_state)
        self.weights = rand.normal(loc=0.0, scale=0.01, size=1 +  X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for x, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(x))
                self.weights[1:] += update * x
                self.weights[0] += update
                # take a look again, not multiplications, line 26 & line 27
                errors += int(update != 0.0)
                self.errors_.append(errors)
            return self
    
    def net_input(self, X):
        z = np.dot(X, self.weights[1:]) + self.weights[0]
        return z
    
    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, -1)
    
    def score(self, X, y):
        misclassified_data_count = 0
        for xi, target in zip(X, y):
            output = self.predict(xi)
            if(target != output):
                misclassified_data_count += 1
        total_data_count = len(X)
        self.score_ = (total_data_count - misclassified_data_count)/total_data_count
        return self.score_



X,y = load_iris(return_X_y=False)
# plt.scatter(X[:50, 0], X[:50, 1],
#             color='green', marker='x', label='setosa')
# plt.scatter(X[50:100, 0], X[50:100, 1],
#             color='red', marker='o', label='versicolor')
# plt.xlabel('sepal length')
# plt.ylabel('petal length')
# plt.legend(loc='upper right')
# plt.show()

per = Perceptron(learning_rate=0.1, n_iter=100, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # when not continuous features

per.fit(X_train, y_train)
print(per.score(X_test, y_test), per.score(X_train, y_train))

# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Perceptron.html



model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test), model.score(X_train, y_train))


# plt.plot(range(1, len(per.errors_) + 1), per.errors_, marker='o')
# plt.xlabel('Epochs')
# plt.ylabel('Number of updates')
# plt.show()

