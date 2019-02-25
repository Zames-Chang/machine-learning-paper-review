from module.DeepAutoEncoder import DeepAutoEncoder
import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist #load minst data

(x_train, y_train),(x_test, y_test) = mnist.load_data() #load minst data
x_train = x_train/255 #normalize
np.random.shuffle(x_train) #shuffle
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2]) #flatten
x_test = x_test/255 #normalize
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2]) #flatten
np.random.shuffle(x_test) #shuffle

model = DeepAutoEncoder(x_train[:1000],x_test[:1000]) #build model

trainLoss,testLoss = model.train(0.01,0.1,[64,16,4],1000,32) #train model