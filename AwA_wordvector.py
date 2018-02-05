import tensorflow as tf
import numpy as np, h5py
import scipy.io as sio
import sys
import random
import kNN_cosine
import re
from numpy import *   

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def compute_accuracy(test_word, test_visual, test_id, test_label):
    global left_w1     
    word_pre = sess.run(left_w1, feed_dict={word_features: test_word})
    test_id = np.squeeze(np.asarray(test_id))
    outpre = [0]*6180  
    test_label = np.squeeze(np.asarray(test_label))
    test_label = test_label.astype("float32")
    for i in range(6180):  
        outputLabel = kNN_cosine.kNNClassify(test_visual[i,:], word_pre, test_id, 1) 
        outpre[i] = outputLabel
    correct_prediction = tf.equal(outpre, test_label)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={
                                           word_features: test_word, visual_features: test_visual})
    return result


# # data

f=sio.loadmat('./data/AwA_data/wordvector/train_word.mat')
word=np.array(f['train_word'])
word.shape

f=sio.loadmat('./data/AwA_data/train_googlenet_bn.mat')
x=np.array(f['train_googlenet_bn'])
x.shape

f=sio.loadmat('./data/AwA_data/test_googlenet_bn.mat')
x_test=np.array(f['test_googlenet_bn'])
x_test.shape

f=sio.loadmat('./data/AwA_data/test_labels.mat')
test_label=np.array(f['test_labels'])
test_label.shape

f=sio.loadmat('./data/AwA_data/testclasses_id.mat')
test_id=np.array(f['testclasses_id'])
test_id.shape

f=sio.loadmat('./data/AwA_data/wordvector/test_vectors.mat')
word_pro=np.array(f['test_vectors'])
word_pro.shape



# # data shuffle
def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(x))
        np.random.shuffle(idxs)
        shuf_visual = x[idxs]
        shuf_word = word[idxs]
        batch_size = 64
        for batch_idx in range(0, len(x), batch_size):
            visual_batch = shuf_visual[batch_idx:batch_idx+batch_size]
            visual_batch = visual_batch.astype("float32")
            word_batch = shuf_word[batch_idx:batch_idx+batch_size]
            yield word_batch, visual_batch 
            



# # Placeholder
# define placeholder for inputs to network
word_features = tf.placeholder(tf.float32, [None, 1000])
visual_features = tf.placeholder(tf.float32, [None, 1024])


# # Network
# AwA 1000 1024 ReLu, 1e-3 * regularisers, 64 batch, 0.0001 Adam
W_left_w1 = weight_variable([1000, 1024])
b_left_w1 = bias_variable([1024])
left_w1 = tf.nn.relu(tf.matmul(word_features, W_left_w1) + b_left_w1)


# # loss
loss_w = tf.reduce_mean(tf.square(left_w1 - visual_features))

# L2 regularisation for the fully connected parameters.             
regularisers_w = (tf.nn.l2_loss(W_left_w1) + tf.nn.l2_loss(b_left_w1))
                  
                  
# Add the regularisation term to the loss.
loss_w += 1e-3 * regularisers_w


train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_w)



sess = tf.Session()
sess.run(tf.global_variables_initializer())


# # Run
iter_ = data_iterator()
for i in range(1000000):
    word_batch_val, visual_batch_val = iter_.next()
    sess.run(train_step, feed_dict={word_features: word_batch_val, visual_features: visual_batch_val})
    if i % 1000 == 0:
        print(compute_accuracy(word_pro, x_test, test_id, test_label))



