import pandas as pd
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


def score(errors, total_number):
    return (total_number - errors) / total_number


class MyPerceptron(object):
    def __init__(self):
        self.weights = {}

    def function(self, features):
        """Activation function

        z is the input vector Z
        weights is the vector of weights
        """
        total_sum = self.weights['__bias__']
        for feature, weight in features.items():
            # f(x) = wx + b
            total_sum += weight * self.weights[feature]

        return 1.0 if total_sum >= 0.0 else 0.0

    def predict(self, features):
        return [self.function(instance_features) for instance_features in features]

    def train(self, features, labels, r=1.0, max_iter=5):
        """Train the weights

        dataset is the input dataset
        """
        # initialize all weights to 0

        count_training_ex = len(features)

        self.weights = defaultdict(float)

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

                    for feature, weight in instance_features.items():
                        self.weights[feature] += diff * weight

                    # update the bias
                    self.weights['__bias__'] += diff

            print('Number of Errors: ' + str(error_count))
            print('Score: ' + str(score(error_count, count_training_ex)))
        return


def bag_of_words(train_dataset, max_lines):
    """Create a bag of words features

    a dict representation of word counts
    """
    # read from the file
    tsv_file = pd.read_csv(filepath_or_buffer=train_dataset, delimiter='\t', quoting=3, nrows=max_lines, header=None)

    # using lowercase
    tsv_file.iloc[:, 1] = tsv_file.iloc[:, 1].str.lower()

    # change the sentiment, positive = 1.0, negative = 0.0
    tsv_file.iloc[:, 0] = tsv_file.iloc[:, 0].astype(float)
    for row_number in range(len(tsv_file.iloc[:, 0])):
        if tsv_file.iloc[row_number, 0] == 2:
            tsv_file.iloc[row_number, 0] = 1.0
        else:
            tsv_file.iloc[row_number, 0] = 0.0

    # create a bag of words and add to the end
    feature_list = []
    for row_number in range(len(tsv_file.iloc[:, 1])):
        feature_list.append(count_appearance(tsv_file.iloc[row_number, 1].split()))

    # tsv_file[len(tsv_file) - 1] = feature_list
    label_list = tsv_file.iloc[:, 0].to_list()

    return label_list, feature_list


def count_appearance(list_of_words):
    bag = defaultdict(int)
    for word in list_of_words:
        bag[word] = 1
    return bag


# my perceptron
print('----------')
print('My Perceptron')
train_labels, train_features = bag_of_words('yelp_sentiment_tokenized/train_tokenized.tsv', 10000)
test_labels, test_features = bag_of_words('yelp_sentiment_tokenized/test_tokenized.tsv', 10000)
my_perceptron = MyPerceptron()
my_perceptron.train(train_features, train_labels, max_iter=20)
my_perceptron.predict(test_features)
print('\naccuracy:', accuracy_score(test_labels, my_perceptron.predict(test_features)))
print('----------')

# sklearn
print('\n----------')
print('Sklearn')
vectorizer = DictVectorizer()
train_vectorized = vectorizer.fit_transform(train_features)
test_vectorized = vectorizer.transform(test_features)

sklearn_perceptron = Perceptron()
sklearn_perceptron.fit(train_vectorized, train_labels)
print("training accuracy:", sklearn_perceptron.score(train_vectorized, train_labels))
print("test accuracy:", sklearn_perceptron.score(test_vectorized, test_labels))
print('----------')

# random shuffling
