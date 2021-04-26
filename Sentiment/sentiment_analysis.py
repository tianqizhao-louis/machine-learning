import pandas as pd
from collections import defaultdict


def score(errors, total_number):
    return (total_number - errors) / total_number


class MyPerceptron(object):
    def __init__(self):
        self.weight_vector = []

    def function(self, features):
        """Activation function

        z is the input vector Z
        weight_vector is the vector of weights
        """
        total_sum = self.weight_vector[0]
        for i in range(1, len(features)):
            # f(x) = wx + b
            total_sum += features[i] * self.weight_vector[i + 1]

        return 1.0 if total_sum >= 0.0 else 0.0

    def train(self, features, labels, r=0.01, max_iter=5):
        """Train the weights

        dataset is the input dataset
        """
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
            print('Score: ' + str(score(error_count, total_sum_dataset)))
            # random.shuffle(dataset)
        return


def bag_of_words(train_dataset, test_dataset):
    """Create a bag of words features

    a dict representation of word counts
    """
    # read from the file
    tsv_file = pd.read_csv(filepath_or_buffer=train_dataset, delimiter='\t', quotechar='"', nrows=3, header=None)

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
    new_column = []
    for row_number in range(len(tsv_file.iloc[:, 1])):
        new_column.append(count_appearance(tsv_file.iloc[row_number, 1].split()))

    tsv_file[len(tsv_file) - 1] = new_column


def count_appearance(list_of_words):
    bag = defaultdict(int)
    for word in list_of_words:
        bag[word] += 1
    return bag


bag_of_words('yelp_sentiment_tokenized/train_tokenized.tsv', '')