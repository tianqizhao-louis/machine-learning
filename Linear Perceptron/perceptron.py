import random
import numpy as np


class My_Perceptron(object):
    def __init__(self):
        self.weight_vector = []

    def function(self, features):
        '''Activation function
        
        z is the input vector Z
        weight_vector is the vector of weights
        '''
        total_sum = self.weight_vector[0]
        for i in range(1, len(features)):
            # f(x) = wx + b
            total_sum += features[i] * self.weight_vector[i + 1]

        return 1.0 if total_sum >= 0.0 else 0.0

    def train(self, features, labels, r=0.01, max_iter=5):
        '''Train the weights

        dataset is the input dataset
        '''
        # initialize all weights to 0
        self.weight_vector = [0 for i in range(len(features[0]) + 1)]

        total_sum_dataset = len(features)

        for each_iter in range(max_iter):
            error_count = 0
            for instance_features, label in zip(features, labels):
                # Calculate the actual output
                actual_output = self.function(instance_features)
                if actual_output != label:
                    # wrong output
                    error_count += 1
                    # update the weights
                    diff = r * (label - actual_output)

                    for w in range(1, len(self.weight_vector)):
                        self.weight_vector[w] += diff * instance_features[w - 1]

                    # update the bias
                    self.weight_vector[0] += diff

            print('Number of Errors: ' + str(error_count))
            print('Score: ' + str(self.score(error_count, total_sum_dataset)))
            # random.shuffle(dataset)
        return

    def score(self, errors, total_number):
        return (total_number - errors) / total_number


my_perceptron = My_Perceptron()
list_of_lists = []
with open('bankdata.txt', 'r') as txt_file:
    list_of_lists = [[float(item) for item in line.strip().split(',')] for line in txt_file]

all_features = [item[:-1] for item in list_of_lists]
all_labels = [item[-1] for item in list_of_lists]

my_perceptron.train(all_features, all_labels, r=0.001, max_iter=20)

# from sklearn.datasets import load_digits
# from sklearn.linear_model import Perceptron

# X_array = []
# for item in float_list:
#     X_array.append(item[:len(item) - 2])

# y_array = []
# for final_output in float_list:
#     y_array.append(item[len(item) - 1])


# sklearn_perceptron = Perceptron(tol=1e-3, random_state=0)
# sklearn_perceptron.fit(X_array, y_array)
# sklearn_perceptron.score(X_array, y_array)
