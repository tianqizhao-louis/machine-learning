# https://realpython.com/linear-regression-in-python/

# Step 1: Import packages and classes
import numpy as np
from sklearn.linear_model import LinearRegression

from read_data import read_from_csv
from read_data import retrieve_training_x
from read_data import retrieve_training_y
from read_data import retrieve_testing_x
from read_data import retrieve_testing_y

import matplotlib.pyplot as plt

# Step 2: Provide data
dataset = read_from_csv()
training_x = retrieve_training_x(dataset)
training_y = retrieve_training_y(dataset)

testing_x = retrieve_testing_x(dataset)
testing_y = retrieve_testing_y(dataset)

x = np.array(training_x).reshape((-1, 1))
y = np.array(training_y)

print(x)
print(y)

# Step 3: Create a model and fit it
model = LinearRegression()

model.fit(x, y)

# Step 4: Get results
r_sq = score(x, y)
print('coefficient of determination:', r_sq)
print('intercept:', model.intercept_)
print('slope:', model.coef_)

# Step 5: Predict response
y_pred = model.predict(x)
print('predicted response:', y_pred[:10], sep='\n')

x_new = np.array(testing_x).reshape((-1, 1))
print(x_new)

y_new = model.predict(x_new)
print(y_new[:10])


# additional: plot the graph
join_arr_x = np.concatenate((x, x_new))
join_arr_y = np.concatenate((y, np.array(testing_y)))

plt.scatter(join_arr_x, join_arr_y, color = "green")

predicted_graph = np.array(y_new[:10])
plt.plot(predicted_graph, linestyle = 'dotted', color="red")


# naming the x axis
plt.xlabel('Days')
# naming the y axis
plt.ylabel('Reaction')
  
# giving a title to my graph
plt.title('Linear Regression')
  
# function to show the plot
plt.show()