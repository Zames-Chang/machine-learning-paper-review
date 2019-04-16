import tensorflow as tf

class google_net_cell(object):
    def __init__(self,image_width,image_height,channel):
        self.a = 0
        self.width = image_width
        self.height = image_height
        self.channel = channel
        self.filter_number = [channel,channel,channel,channel]
    def get_padding(self,tensor,shape):
        width = shape[0]
        height = shape[1]
        width2 = width // 2
        height2 = height // 2
        top = ((height - height2)//2)
        bottom = (height - height2 - top)
        left = ((width - width2) //2)
        right = (width - width2 - left)
        #print(right)
        paddings = [[top,bottom,],[left,right]]
        return ZeroPadding2D(paddings)(tensor)
    def conv(self,input_data):
        input_layer = tf.reshape(input_data, [-1, self.width, self.height, self.channel])
        conv1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=self.filter_number[0],
          kernel_size=[1, 1],
          padding="same",
          activation=tf.nn.relu)
        conv2_1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=self.filter_number[1],
          kernel_size=[1, 1],
          padding="same",
          activation=tf.nn.relu)
        conv2_2 = tf.layers.conv2d(
          inputs=conv2_1,
          filters=self.filter_number[1],
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)
        conv3_1 = tf.layers.conv2d(
          inputs=input_layer,
          filters=self.filter_number[2],
          kernel_size=[1, 1],
          padding="same",
          activation=tf.nn.relu)
        conv3_2 = tf.layers.conv2d(
          inputs=conv3_1,
          filters=self.filter_number[2],
          kernel_size=[5, 5],
          padding="same",
          activation=tf.nn.relu)
        pool1 = tf.layers.average_pooling2d(inputs=input_layer, pool_size=[3, 3], strides=2)
        padding_pool = self.get_padding(pool1,[self.width,self.height])
        conv4 = tf.layers.conv2d(
          inputs=conv2_1,
          filters=self.filter_number[0],
          kernel_size=[1, 1],
          padding="same",
          activation=tf.nn.relu)
        return tf.concat([conv1,conv2_2,conv3_2,conv4],-1)