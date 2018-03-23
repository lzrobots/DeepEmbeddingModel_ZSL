import tensorflow as tf
import numpy as np
import scipy.io as sio
import kNN
import kNN_cosine
from numpy import *
from sklearn.metrics import accuracy_score


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def compute_accuracy(test_att, test_visual, test_id, test_label):
    global left_a2
    att_pre = sess.run(left_a2, feed_dict={att_features: test_att})
    test_id = np.squeeze(np.asarray(test_id))
    outpre = [0] * test_visual.shape[0]  # CUB 2933
    test_label = np.squeeze(np.asarray(test_label))
    test_label = test_label.astype("float32")
    for i in range(test_visual.shape[0]):  # CUB 2933
        outputLabel = kNN.kNNClassify(test_visual[i, :], att_pre, test_id, 1)
        outpre[i] = outputLabel
    # compute averaged per class accuracy
    outpre = np.array(outpre, dtype='int')
    unique_labels = np.unique(test_label)
    acc = 0
    for l in unique_labels:
        idx = np.nonzero(test_label == l)[0]
        acc += accuracy_score(test_label[idx], outpre[idx])
    acc = acc / unique_labels.shape[0]
    return acc


dataroot = './data/'
dataset = 'SUN_data'
image_embedding = 'res101'
class_embedding = 'att' # original_att

matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + image_embedding + ".mat")
feature = matcontent['features'].T
label = matcontent['labels'].astype(int).squeeze() - 1
matcontent = sio.loadmat(dataroot + "/" + dataset + "/" + class_embedding + "_splits.mat")
# numpy array index starts from 0, matlab starts from 1
trainval_loc = matcontent['trainval_loc'].squeeze() - 1
test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

attribute = matcontent['original_att'].T # att

x = feature[trainval_loc]
train_label = label[trainval_loc].astype(int)
att = attribute[train_label]
print(att.shape)
x_test = feature[test_unseen_loc]
test_label = label[test_unseen_loc].astype(int)
x_test_seen = feature[test_seen_loc]
test_label_seen = label[test_seen_loc].astype(int)
test_id = np.unique(test_label)
att_pro = attribute[test_id]


def data_iterator():
    """ A simple data iterator """
    batch_idx = 0
    while True:
        # shuffle labels and features
        idxs = np.arange(0, len(x))
        np.random.shuffle(idxs)
        shuf_visual = x[idxs]
        shuf_att = att[idxs]
        batch_size = 64
        for batch_idx in range(0, len(x), batch_size):
            visual_batch = shuf_visual[batch_idx:batch_idx + batch_size]
            visual_batch = visual_batch.astype("float32")
            att_batch = shuf_att[batch_idx:batch_idx + batch_size]
            yield att_batch, visual_batch


# # Placeholder
# define placeholder for inputs to network
att_features = tf.placeholder(tf.float32, [None, 102])
visual_features = tf.placeholder(tf.float32, [None, 2048])

# # Network
# AwA 85 1600 2048 ReLu, 1e-3 * regularisers, 64 batch, 0.0001 Adam
W_left_a1 = weight_variable([102, 1600])
b_left_a1 = bias_variable([1600])
left_a1 = tf.nn.relu(tf.matmul(att_features, W_left_a1) + b_left_a1)

W_left_a2 = weight_variable([1600, 2048])
b_left_a2 = bias_variable([2048])
left_a2 = tf.nn.relu(tf.matmul(left_a1, W_left_a2) + b_left_a2)

# # loss
loss_a = tf.reduce_mean(tf.square(left_a2 - visual_features))

# L2 regularisation for the fully connected parameters.
regularisers_a = (tf.nn.l2_loss(W_left_a1) + tf.nn.l2_loss(b_left_a1)
                  + tf.nn.l2_loss(W_left_a2) + tf.nn.l2_loss(b_left_a2))

loss_a += 1e-5 * regularisers_a

train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_a)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# # Run
iter_ = data_iterator()
for i in range(1000000):
    att_batch_val, visual_batch_val = iter_.next()
    sess.run(train_step, feed_dict={att_features: att_batch_val, visual_features: visual_batch_val})
    if i % 1000 == 0:
        acc_zsl = compute_accuracy(att_pro, x_test, test_id, test_label)
        acc_seen_gzsl = compute_accuracy(attribute, x_test_seen, np.arange(717), test_label_seen)
        acc_unseen_gzsl = compute_accuracy(attribute, x_test, np.arange(717), test_label)
        H = 2 * acc_seen_gzsl * acc_unseen_gzsl / (acc_seen_gzsl + acc_unseen_gzsl)
        print('zsl:', acc_zsl)
        print('gzsl: seen=%.4f, unseen=%.4f, h=%.4f' % (acc_seen_gzsl, acc_unseen_gzsl, H))






