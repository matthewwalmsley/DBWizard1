# Import the required libraries, configuration information and data
import tensorflow as tf
import config
import data

# Set up placeholders for X & y
X_placeholder = tf.placeholder(tf.float32, [None, config.num_features], name="X")
y_placeholder = tf.placeholder(tf.float32, [None, config.num_classes], name="Y")

# Initialise weight & biases
W = tf.Variable(tf.random_normal([config.num_features, config.num_classes], stddev=config.std_dev), "W")
b = tf.Variable(tf.random_normal([config.num_classes], stddev=config.std_dev), "b")


# Set up the model using softmax
# Should this be based on a sigmoid function at this point???
pred = tf.nn.softmax(tf.matmul(X_placeholder, W) + b)

# Minimise error using cross entropy
with tf.name_scope("cost_function"):
    cost = tf.reduce_mean(-tf.reduce_sum(y_placeholder*tf.log(pred), reduction_indices=1))  # Look at notes re LR and cost
    optimizer = tf.train.GradientDescentOptimizer(config.lr_learning_rate).minimize(cost)
    tf.summary.scalar("cost_function", cost)

# Test model
with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y_placeholder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

# Histograms to allow me to visualise weights & biases
tf.summary.histogram("Weight summary", W)
tf.summary.histogram("Bias summary", b)
tf.summary.histogram("Cost summary", cost)
tf.summary.histogram("Accuracy summary", accuracy)

# Ability to save and restore all variables
saver = tf.train.Saver()

# Initialise all variables
init = tf.global_variables_initializer()

# Create a sesion and run the model
with tf.Session() as sess:

    # Initialise the session
    sess.run(init)

    writer = tf.summary.FileWriter("./Logs/", sess.graph)
    merged = tf.summary.merge_all(key="summaries")

    # Run the model
    for epoch in range(config.lr_training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict = {X_placeholder: data.X_train, y_placeholder: data.y_train})
        summary, acc = sess.run([merged, accuracy], feed_dict={X_placeholder: data.X_test, y_placeholder: data.y_test})

        # Need to make this work!!!
        writer.add_summary(summary, epoch)

        if (epoch+1) % config.lr_display_step == 0:
            print("Epoch:", '%04d' % (epoch), "Cost=", "{:.9f}".format(c), "Accuracy=", "{:.9f}".format(acc))
            # print("Accuracy:", accuracy.eval({X_placeholder: data.X_test, y_placeholder: data.y_test}))

    # Save the data
    save_path = saver.save(sess, "temp/model.ckpt")
    print("Model saved in: %s" % save_path)
