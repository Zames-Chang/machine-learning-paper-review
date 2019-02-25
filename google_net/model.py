import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn
from keras.layers import ZeroPadding2D
from google_cell import google_net_cell
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)
image_width = 28
image_height = 28
cell = google_net_cell(28,28,1)
input_x = tf.placeholder("float", [None, image_width, image_height])
Y = tf.placeholder("float", [None, num_classes])

layer1 = cell.conv(input_x)
cell2 = google_net_cell(28,28,4)
layer2 = cell2.conv(layer1)
cell3 = google_net_cell(28,28,16)
layer3 = cell3.conv(layer2)
fatten = tf.reshape(layer3,[-1,28*28*64])
weights = {
    'w1': tf.Variable(tf.random_normal([28*28*64, num_hidden])),
    'w2': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([num_hidden])),
    'b2': tf.Variable(tf.random_normal([num_classes]))
}

vec1 = tf.nn.relu(tf.matmul(fatten, weights['w1']) + biases['b1'])
vec2 = tf.matmul(vec1, weights['w2']) + biases['b2']
# Define loss and optimizer
prediction = tf.nn.softmax(vec2)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=vec2, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, image_width, image_height))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={input_x: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={input_x: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, image_width, image_height))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", 
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))