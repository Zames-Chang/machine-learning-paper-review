import tensorflow as tf
from keras.layers import ZeroPadding2D,Dense
import math
import numpy as np
import os
from keras import backend as K
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

class Siamese(object):
    def __init__(self,size_x,size_y):
        self.network = ""
        self.feature_size = size_x
        self.lable_size = size_y
        self.learning_rate = 0.001
        self.loss = 0
        self.train_op = ""
        self.x1 = tf.placeholder("float",name='x1',shape = [None, self.feature_size])
        self.x2 = tf.placeholder("float", name='x2',shape = [None, self.feature_size])
        self.y1 = tf.placeholder("int32", name='y1',shape = [None, self.lable_size])
        self.y2 = tf.placeholder("int32",name='y2', shape = [None, self.lable_size])
        self.output1 = ""
        self.output2 = ""
        self.condition = ""
        self.case_loss = ""
        self.test = ""
        self.sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0},allow_soft_placement=True, log_device_placement=True))
        self.build_network()
    def build_network(self):
        num_hidden = 32
        c = 1
        num_classes = 10
        weights = {
            'w1': tf.Variable(tf.random_normal([self.feature_size, num_hidden])),
            'w2': tf.Variable(tf.random_normal([num_hidden, num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([num_hidden])),
            'b2': tf.Variable(tf.random_normal([num_classes]))
        }
        output1 = tf.nn.relu(tf.matmul(self.x1,weights['w1']) + biases['b1'])
        output1 = tf.matmul(output1,weights['w2']) + biases['b2']
        output2 = tf.nn.relu(tf.matmul(self.x2,weights['w1']) + biases['b1'])
        output2 = tf.matmul(output2,weights['w2']) + biases['b2']
        self.output1 = output1
        self.output2 = output2
        self.condition = tf.placeholder(tf.bool, shape=[], name="condition")
        def if_True():
            return tf.losses.mean_squared_error(output1, output2)
        def if_False():
            return tf.maximum(c - tf.losses.mean_squared_error(output1, output2),0)
        self.case_loss = tf.cond(self.condition, if_True , if_False)
        self.loss = self.loss + self.case_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
    def pair(self):
        pair_x = []
        pair_y = []
        for i,j in zip(self.train_X,self.train_Y):
            temp_x = []
            temp_y = []
            for x,y in zip(self.train_X,self.train_Y):
                temp_x.append([i,x])
                temp_y.append([j,y])
            pair_x.append(temp_x)
            pair_y.append(temp_y)
        return pair_x,pair_y
    def train(self,training_steps):
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()
        batch_size = 32
        display_step = 100
        saver = tf.train.Saver()
    # Start training

        # Run the initializer
        self.sess.run(init)

        for step in range(1, training_steps+1):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop)
            for i,j in zip(batch_x,batch_y):
                for x,y in zip(batch_x,batch_y):
                    i = i.reshape([1,self.feature_size])
                    j= j.reshape([1,self.lable_size])
                    x = x.reshape([1,self.feature_size])
                    y = y.reshape([1,self.lable_size])
                    index1, = np.where(j[0] == 1)
                    index2, = np.where(y[0] == 1)
                    #print(index1,index2)
                    if(index1[0] == index2[0]):
                        cond = True
                    else:
                        cond = False
                    #print(cond)
                    _temp = self.sess.run([self.train_op,self.loss], feed_dict={self.x1 : i, self.x2 : x,self.y1 : j, self.y2 : y,self.condition:cond})
                    #print(temp)
            if 1:
                # Calculate batch loss and accuracy
                total_loss = 0
                for i,j in zip(batch_x,batch_y):
                    for x,y in zip(batch_x,batch_y):
                        i = i.reshape([1,self.feature_size])
                        j= j.reshape([1,self.lable_size])
                        x = x.reshape([1,self.feature_size])
                        y = y.reshape([1,self.lable_size])
                        index1, = np.where(j[0] == 1)
                        index2, = np.where(y[0] == 1)
                        #print(index1,index2)
                        if(index1[0] == index2[0]):
                            cond = True
                        else:
                            cond = False
                        loss = self.sess.run([self.loss],feed_dict={self.x1 : i, self.x2 : x,self.y1 : j, self.y2 : y,self.condition:cond})
                        total_loss += loss[0]
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(total_loss))

        print("Optimization Finished!")
        save_path = saver.save(self.sess, "/tmp/model/siamese.ckpt")
    def predict(self,a,b):
        #tf.reset_default_graph()
        a = a.reshape([1,self.feature_size])
        b = b.reshape([1,self.feature_size])
        temp1,temp2 = self.sess.run([self.output1,self.output2], feed_dict={self.x1 : a, self.x2 : b})
        return np.sum(abs(temp1-temp2))
batch_x, batch_y = mnist.train.next_batch(32)
siamese_network = Siamese(batch_x.shape[1], batch_y.shape[1])
sess = siamese_network.train(50)
class SpectralNet(object):
    def __init__(self,mnist,siamese_network):
        self.network = ""
        self.learning_rate = 0.001
        self.loss = 0
        self.train_op = ""
        self.batch_size = 4
        batch_x, batch_y = mnist.train.next_batch(self.batch_size)
        self.feature_size = batch_x[0].shape[0]
        self.x = tf.placeholder("float",name='x',shape = [None, self.feature_size])
        batch_x, batch_y = mnist.train.next_batch(32)
        self.siamese_network = siamese_network
        self.build_network()
        self.sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0},allow_soft_placement=True, log_device_placement=True))
    def build_network(self):
        num_hidden = 32
        num_classes = 10
        weights = {
            'w1': tf.Variable(tf.random_normal([self.feature_size, num_hidden])),
            'w2': tf.Variable(tf.random_normal([num_hidden, num_classes])),
            'orth': tf.Variable(tf.random_normal([num_classes, num_classes]),name='orth')
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([num_hidden])),
            'b2': tf.Variable(tf.random_normal([num_classes]))
        }
        output = tf.nn.relu(tf.matmul(self.x,weights['w1']) + biases['b1'])
        output = tf.nn.softmax(tf.matmul(output,weights['w2']) + biases['b2'])
        #weights['orth'] = self.orthonorm_op(output)
        self.before_orth = output
        self.W = tf.placeholder("float",name='W',shape = [self.batch_size,self.batch_size])
        output = tf.matmul(output,self.orthonorm_op(output))
        self.embed_y = output
        for i in range(self.batch_size):
            for j in range(self.batch_size):
                self.loss += tf.reduce_sum((tf.abs(output[i]-output[j])))*self.W[i][j]
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss)
    def aff_matrix(self,train_x):
        w = []
        for i in train_x:
            temp_w = []
            for j in train_x:
                temp = self.siamese_network.predict(i,j)
                temp_w.append(temp)
            w.append(temp_w)
        return w
    def orthonorm_op(self,x, epsilon=1e-7):
        x_2 = K.dot(K.transpose(x), x)
        x_2 += K.eye(K.int_shape(x)[1])*epsilon
        print(x_2)
        L = tf.cholesky(x_2)
        ortho_weights = tf.transpose(tf.matrix_inverse(L)) * tf.sqrt(tf.cast(tf.shape(x)[0], dtype=K.floatx()))
        return ortho_weights
    def train(self,training_steps):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        
        # Start training
        # Run the initializer
        self.sess.run(init)
        for step in range(1, training_steps+1):
            batch_x, batch_y = mnist.train.next_batch(self.batch_size)
            feed_dict={
                self.x : batch_x,
                self.W : self.aff_matrix(batch_x)
            }
            _,loss = self.sess.run([self.train_op,self.loss],feed_dict =feed_dict)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))
    def predict(self,test_x): 
        return self.sess.run([self.embed_y], feed_dict={self.x : test_x})
S_network = SpectralNet(mnist,siamese_network)
S_network.train(50)