import random
import numpy as np 

class My_Perceptron(object):
    def __init__(self, r=0.01, initial_weight=0, max_iter=5, threshold=0):
        self.r = r # learning rate, between 0 and 1
        self.initial_weight = initial_weight # initial value of weights
        self.max_iter = max_iter
        self.threshold = threshold # threshold, in this case, 0
    
    def function(self, z, weight_vector):
        '''Activation function
        
        z is the input vector Z
        weight_vector is the vector of weights
        '''
        bias = weight_vector[0]
        total_sum = 0
        for i in range(len(z) - 1):
            # f(x) = wx + b
            total_sum += z[i] * weight_vector[i+1]
        
        total_sum += bias

        return 1.0 if total_sum >= 0.0 else 0.0
    
    def train(self, dataset):
        '''Train the weights

        dataset is the input dataset
        '''
        # initialize all weights to 0
        weight_vector = [0 for i in range(len(dataset[0]))]

        total_sum_dataset = len(dataset)

        for each_iter in range(self.max_iter):
            error_count = 0
            for line in dataset:
                # Calculate the actual output
                actual_output = self.function(line, weight_vector)
                # desired output, last item in the line
                desired_output = line[len(line) - 1]
                if actual_output != desired_output:
                    # wrong output
                    error_count += 1
                
                # update the weights
                for w in range(1, len(weight_vector)):
                   weight_vector[w] = weight_vector[w] + self.r * (desired_output - actual_output) * line[w]


                # update the bias
                weight_vector[0] = weight_vector[0] + self.r * (desired_output - actual_output)
            print('Number of Errors: ' + str(error_count))
            print('Score: ' + str(self.score(error_count, total_sum_dataset)))
            random.shuffle(dataset)
        return
    
    def score(self, errors, total_number):
        return (total_number - errors) / total_number




my_perceptron = My_Perceptron()
list_of_lists = []
with open('bankdata.txt', 'r') as txt_file:
    list_of_lists = [line.strip().split(',') for line in txt_file]

float_list = [[float(item) for item in inner] for inner in list_of_lists]

my_perceptron.train(float_list)

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