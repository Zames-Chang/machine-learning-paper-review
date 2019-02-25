import tensorflow as tf
import numpy as np

class DeepAutoEncoder:
    def __init__(self,trainX,testX):
        """
        define data and shape
        """
        self.trainX = trainX
        self.testX = testX
        self.shape_train = trainX.shape
        self.shape_test = testX.shape
    def makeBatch(self,batchSize,total):
        """
        my batch create function
        """
        nums = []
        i = 0
        while(1):
            if(i*batchSize > total):
                nums.append(total)
                break
            else:    
                nums.append(i*batchSize)
            i = i+1
        return nums
    def train(self,lr,beta,layerDim,iters,batchSize,log = True):
        #batchNums = self.shape_train[0] // batchSize
        #input
        data = tf.placeholder(tf.float32, shape=(None, self.shape_train[1]), name="input")
        trainBatch = self.makeBatch(batchSize,self.shape_train[0])
        w = tf.Variable(tf.random_normal([self.shape_train[1],layerDim[0]]),name = "w_0")
        b = tf.Variable(tf.random_normal([layerDim[0]]),name = "b_0")
        z = tf.nn.relu(tf.matmul(data,w) + b)
        w_list = [w]
        b_list = [b]
        z_list = [z]
        # encoder layer
        for i in range(1,len(layerDim)):
            w = tf.Variable(tf.random_normal([layerDim[i-1],layerDim[i]]),name = "w_{}".format(i))
            b = tf.Variable(tf.random_normal([layerDim[i]]),name = "b_{}".format(i))
            z = tf.nn.relu(tf.matmul(z_list[-1],w) + b)
            w_list.append(w)
            b_list.append(b)
            z_list.append(z) 
        layerDim.reverse()
        # decoder layer
        for i in range(1,len(layerDim)):
            w = tf.Variable(tf.random_normal([layerDim[i-1],layerDim[i]]),name = "Dw_{}".format(i))
            b = tf.Variable(tf.random_normal([layerDim[i]]),name = "Db_{}".format(i))
            z = tf.nn.relu(tf.matmul(z_list[-1],w) + b)
            w_list.append(w)
            b_list.append(b)
            z_list.append(z)
        w = tf.Variable(tf.random_normal([layerDim[-1],self.shape_train[1]]),name = "w_out")
        b = tf.Variable(tf.random_normal([self.shape_train[1]]),name = "b_out")
        z = tf.nn.sigmoid(tf.matmul(z_list[-1],w) + b)
        w_list.append(w)
        b_list.append(b)
        z_list.append(z)
        #define loss
        loss =  tf.reduce_mean(tf.nn.l2_loss(z - data))
        # add l2 regulation
        for w in w_list:
            regularizer = tf.nn.l2_loss(w)
            loss = tf.reduce_mean(loss + beta* regularizer)
        # use adam optimizer
        optimizer = tf.train.AdamOptimizer(lr)
        # minmize loss
        train_op = optimizer.minimize(loss)
        trainLoss = []
        testLoss = []
        # training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            for i in range(iters):
                for index in range(1,len(trainBatch)):
                    start = trainBatch[index-1]
                    end = trainBatch[index]
                    feed_dict = {data:self.trainX[start:end]}
                    _, loss_t = sess.run([train_op, loss],feed_dict)
                    trainLoss.append(loss_t)
                    feed_dict = {data:self.testX}
                    loss_test = sess.run([loss],feed_dict)
                    testLoss.append(loss_test[0])
                if(log):
                    print("training loss = ",loss_t,"    testing loss = ",loss_test[0])
            save_path = saver.save(sess, "/result/model_{}_{}.ckpt".format(iters,beta)) # save model
        return trainLoss,testLoss