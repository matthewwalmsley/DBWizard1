# Import config and numpy libraries
import config
import numpy as np

# Load in the data (csv file)
data = np.genfromtxt(config.csv_input_file, delimiter=",")
np.random.shuffle(data)

# Split the data in inputs and outputs (assume y is the final column)
data_split = np.hsplit(data, [data.shape[1] - 1, data.shape[1]])
X = data_split[0]
y = data_split[1]

# Split the data into training and test subsets
X_split = np.vsplit(X, [int(X.shape[0] * config.train_split), X.shape[0]])
X_train = X_split[0];
X_test = X_split[1];

y_split = np.vsplit(y, [int(y.shape[0] * config.train_split), y.shape[0]])
y_train = y_split[0];
y_test = y_split[1];
