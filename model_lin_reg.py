# Import third party libraries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# Add bias of 1 to the shape
def append_bias_reshape(features, labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l

# Load in the data (csv file)
data = np.genfromtxt('data/dbitems.csv', delimiter=",")

# Separate features and labels
features = data
data_split = np.hsplit(data, [data.shape[1] - 1, data.shape[1]])
features = data_split[0]
labels = data_split[1]

# Add biases and count the number of features
f, l = append_bias_reshape(features, labels)
n_features = f.shape[1]

# Split the data into random selection of
rnd_indices = np.random.rand(len(f)) < 0.80
X_train = f[rnd_indices]
y_train = l[rnd_indices]
X_test = f[~rnd_indices]
y_test = l[~rnd_indices]

# Learning parameters
learning_rate = 0.01
training_epochs = 200

# Initialise arrays to record metrics
cost_history = np.empty(shape=[1], dtype=float)
mse_history = np.empty(shape=[1], dtype=float)

X = tf.placeholder(tf.float32,[None, n_features])
y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.ones([n_features,1]))

init = tf.global_variables_initializer()

y_ = tf.matmul(X, W)

# Model - minimise cost using graident descent
with tf.name_scope("cost_function"):
    cost = tf.reduce_mean(tf.square(y_ - y))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    tf.summary.scalar("cost_function", cost)

sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={X: X_train, y: y_train})

    y_pred = sess.run(y_, feed_dict={X: X_test})
    mse = tf.reduce_mean(tf.square(y_pred - y_test))

    cost_history = np.append(cost_history,sess.run(cost, feed_dict={X: X_train, y: y_train}))
    mse_history = np.append(mse_history,sess.run(mse))

plt.title('Cost History')
plt.xlabel('Epoch')
plt.plot(range(len(cost_history)), cost_history, 'r--', range(len(cost_history)), mse_history, 'g^')
plt.axis([0,training_epochs,0,np.max(cost_history)])
plt.show()
