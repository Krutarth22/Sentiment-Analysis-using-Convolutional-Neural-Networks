import tensorflow as tf
import numpy as np

class Word_Embedding(object):
    """
    Here we will see an embedding layer, which is followed by a convolutional layer.
    The convolutional layer will be followed by max pooling and finally softmax layer.

    """

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes,
        num_filters, l2_reg_lambda = 0.0):

        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32,[None, num_classes], name = 'input_y')
        self.droput_keep_probability = tf.placeholder(tf.float32, name = "dropout_keep_probability")

        #An optional setting here is to keep track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        #Here we create an embedding layer
        with tf.device('Krutarth’s MacBook Pro'), tf.name_scope("embedding"):
            self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name='W')
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)


        #Here we create a convolutional layer with max pooling for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_sizes):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape = [num_filters]), name='b')

                conv = tf.nn.conv2d(self.embedded_chars, W, strides=[1,1,1,1], padding='VALID',
                                    name='conv')

                #Here RELU function is used since it introduces non-linearity the best
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
                #After introducing non-linearity, we max pool over the outputs
                pooled = tf.nn.max_pool(h, ksize = [1, sequence_length - filter_size+1, 1, 1],
                                        strides=[1,1,1,1], padding="VALID", name="pool")

                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        #Here we add a dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.droput_keep_probability)

        #Final scores and predictions

        with tf.name_scope("output"):
            W = tf.get_variable("W", shape=[num_filters_total, num_classes],
                                initializer=tf.contrib.layers.xavier_intializer())

            b = tf.Variable(tf.constant(0.1), shape=[num_classes],name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss+= tf.nn.l2_loss(b)

            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            self.predictions = tf.argmax(self.scores, 1, name = 'predictions')

        #Calculate mean entropy loss

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels = self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda*l2_loss

        #Accuracy
        with tf.name_scope("Accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")





