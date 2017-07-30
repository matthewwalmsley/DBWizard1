# Data settings
csv_input_file = 'data/dbitems.csv'
num_features = 46 # The number of features in the data set

# Learning parameters
train_split = 0.80
learning_rate = 0.00001
batch_size = 10
num_classes = 2 # The number of outcomes
units_h1 = 64 # The number of units in the first hidden layer
units_h2 = 64 # The number of units in the second hidden layer
max_steps = 200 # Number of learning iterations

# Initialisation parameters
std_dev = 0.01 # Standard deviation used when generating initial weights
