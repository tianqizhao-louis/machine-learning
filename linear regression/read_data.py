import csv
from collections import defaultdict

def read_from_csv():
    to_return = defaultdict(list)
    with open('data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                to_return[row[2]].append(row)
                line_count += 1
    
    return to_return
        
        # for k in to_return:
        #     print(k)
        #     for m in to_return[k]:
        #         print('\t' + str(m))

def retrieve_dataset_x(dataset):
    x = []
    for individual in dataset:
        for dict_item in dataset.get(individual):
            x.append(int(dict_item[1]))
    
    return x

def retrieve_dataset_y(dataset):
    y = []
    for individual in dataset:
        for dict_item in dataset.get(individual):
            y.append(float(dict_item[0]))
    
    return y

def retrieve_training_x(dataset):
    training_x = []
    for individual in dataset:
        for dict_item in dataset.get(individual):
            if int(individual) < 338:
                training_x.append(int(dict_item[1]))
    return training_x

def retrieve_training_y(dataset):
    training_y = []
    for individual in dataset:
        for dict_item in dataset.get(individual):
            if int(individual) < 338:
                training_y.append(float(dict_item[0]))
    return training_y

def retrieve_testing_x(dataset):
    testing_x = []
    for individual in dataset:
        for dict_item in dataset.get(individual):
            if int(individual) > 338:
                testing_x.append(int(dict_item[1]))
    return testing_x

def retrieve_testing_y(dataset):
    testing_y = []
    for individual in dataset:
        for dict_item in dataset.get(individual):
            if int(individual) > 338:
                testing_y.append(float(dict_item[0]))
    return testing_y