# Data settings
csv_input_file = 'data/dbitems.csv'
num_features = 46 # The number of features in the data set

# Initialisation parameters
std_dev = 0.01 # Standard deviation used when generating initial weights
train_split = 0.75
num_classes = 2 # The number of outcomes

# Learning parameters - nn
nn_learning_rate = 0.00001
nn_batch_size = 10
nn_units_h1 = 64 # The number of units in the first hidden layer
nn_units_h2 = 64 # The number of units in the second hidden layer
nn_max_steps = 200 # Number of learning iterations

# Learning parameters - lr
lr_learning_rate = 0.05
lr_training_epochs = 100
lr_batch_size = 10
lr_display_step = 1
