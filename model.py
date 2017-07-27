# Import the required libraries, configuration information and data
import tensorflow as tf
import config
import data


def init_random(shape, std_dev, name, histogram_name):
    """Initialise random values in a given shape and call it name
    Add a histogram so it can be tracked if required

    Args:
        shape: Dimensions of the tensor to be populated
        std_dev: The standard deviation of the random numbers
        name: The name of the tensor
        historgram_name: The name of the histogram to be created

    Returns: A tensor with random values of size shape
    """
    t = tf.Variable(tf.random_normal(shape, stddev=std_dev), name=name)
    tf.summary.histogram(histogram_name, t)
    return t


def build_model(X, w1, b1, w2, b2, wo, bo):
    """Build the model - a two hidden layer nn

    Args:
        X: Tensor of input values
        w1: Hidden layer 1 weights
        b1: Hidden layer 1 biases
        w2: Hidden layer 2 weights
        b2: Hidden layer 2 biases
        wo: Output layer weights
        bo: Hidden layer 1 biases

    Returns: the output layer tensor
    """
    with tf.name_scope("layer_1"):
        layer_1 = tf.add(tf.matmul(X, w1), b1)
        layer_1 = tf.nn.relu(layer_1)

    with tf.name_scope("layer_2"):
        layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
        layer_2 = tf.nn.relu(layer_2)

    with tf.name_scope("layer_1"):
        layer_o = tf.matmul(layer_2, wo) + bo

    return layer_o


# Set up placeholders for X & y
X = tf.placeholder(tf.float32, [None, config.num_features], name="X")
Y = tf.placeholder(tf.float32, [None, config.num_classes], name="Y")

# Set up the weights & biases for the two hiddne and one output layer
w1 = init_random([config.num_features, config.units_h1], config.std_dev, "w1", "Weight 1 Summary")
b1 = init_random([config.units_h1], config.std_dev, "b1", "Bias 1 Summary")

w2 = init_random([config.units_h1, config.units_h2], config.std_dev, "w2", "Weight 2 Summary")
b2 = init_random([config.units_h2], config.std_dev, "b2", "Bias 2 Summary")

wo = init_random([config.units_h2, config.num_classes], config.std_dev, "w3", "Output Summary")
bo = init_random([config.num_classes], config.std_dev, "b3", "Output Bias Summary")

# Set up the model
y = build_model(X, w1, b1, w2, b2, wo, bo)

with tf.name_scope("cost_function"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y))
    train_step = tf.train.AdamOptimizer(config.learning_rate).minimize(cross_entropy)
    tf.summary.scalar("cost_function", cross_entropy)

with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Ability to save all variables
saver = tf.train.Saver()

# Create a sesion and run the model
with tf.Session() as sess:

    # Set up the logs
    writer = tf.summary.FileWriter("./logs/", sess.graph)
    merged = tf.summary.merge_all(key="summaries")

    # Initialise all variables
    tf.global_variables_initializer().run()

    # Run the model
    for i in range(config.max_steps):
        if i % config.batch_size == 0: # Record summarys and accuracy
            summary, acc = sess.run([merged, accuracy], feed_dict={X: data.X_test, Y: data.y_test})
            writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:
            sess.run(train_step, feed_dict={X: data.X_train, Y: data.y_train})

    # Save the data
    save_path = saver.save(sess, "temp/model.ckpt")
    print("Model saved in: %s" % save_path)
