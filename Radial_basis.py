from __future__ import print_function
import numpy as np
from sklearn.metrics import confusion_matrix
from numpy import ndarray
import matplotlib.pyplot as plt
# Import MNIST data

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.0001
training_epochs = 5
batch_size = 100
display_step = 0.5
a = np.zeros(shape=(10,10))
# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#Radial Basis
def RBF(x, C):
    """Computes distance from cluster centers defined in input C
    
    Both outdim and indim should be integers.
    """
    return -tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf.expand_dims(x,2),
                                                   tf.expand_dims(C,0))),1))
# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    
    return layer_1

# Construct model
input_X = multilayer_perceptron(X)

logits = RBF(input_X,weights['out'])+biases['out']


cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
i = 0
train_accuracy = np.zeros((1,100))
avg_cost = np.zeros((1,100))
count = np.zeros((1,100))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        batch = mnist.train.next_batch(100)
        

        if i%100 == 0:
            j = int(i/100)
            print(j)
            count[0][j-1] = j
            train_accuracy[0][j-1] = accuracy.eval(feed_dict={
                X:batch[0], Y: batch[1]})
            _,c = sess.run([train_step,cross_entropy],feed_dict={X:batch[0], Y:batch[1]})
            total_batches = j
            avg_cost[0][j-1] = c/total_batches 
            print("\rstep %d, training accuracy %g"%(i, train_accuracy[0][j-1]), end="" if i%10 else "\n")
        train_step.run(feed_dict={X: batch[0], Y: batch[1]})
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))
    cm = tf.contrib.metrics.confusion_matrix(tf.argmax(Y,1),tf.argmax(logits,1))
    print('Confusion Matrix: \n\n', tf.Tensor.eval(cm,feed_dict={X: mnist.test.images, Y: mnist.test.labels}, session=None))

for i in range(60):
    print(count)
    print(train_accuracy[0][i]*100)
    print(avg_cost[0][i])


#Printing plots 
plt.plot(count, train_accuracy, 'ro')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
plt.show()

plt.plot(count, avg_cost, 'bo')
plt.xlabel('epoch')
plt.ylabel('training loss')
plt.show()
