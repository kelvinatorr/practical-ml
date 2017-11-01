import tensorflow as tf
import pickle
# read pickle

print 'Running'

class Batcher:
    def __init__(self, data, name):
        self.data = data
        self.name = name
        self.last_batch = None
        self.new_batch = None
        self.num_examples = len(data)

    def next_batch(self, batch_size):
        if self.last_batch is None:
            self.last_batch = 0
            self.new_batch = batch_size
        else:
            self.last_batch = self.new_batch
            self.new_batch += batch_size

        batch = self.data[self.last_batch: self.new_batch]
        # batch_y = self.data[self.last_batch: self.new_batch][0][1]
        return batch

    def reset_next_last_batch(self):
        self.last_batch = None

# #test Batcher

# data = [
#     [0, 1, 1, 0],
#     [0, 1, 2, 0],
#     [0, 3, 1, 0],
#     [0, 4, 1, 0],
# ]
#
# data = ['apple', 'bannana', 'carrot', 'peach']

# print data[2:4]
#
# batched_data = Batcher(data, 'test_data')
# batch_size = 1
#
# print batched_data.name
# for _ in range(int(len(data)/ batch_size)):
#     e_x = batched_data.next_batch(batch_size)
#     print e_x



def get_data():
    data = pickle.load(open('sentiment_set.pickle'))
    # data = pickle.load(open('sentdex-sentiment_set.pickle'))
    train_x = data[0]
    train_y = data[1]
    test_x = data[2]
    test_y = data[3]
    return Batcher(train_x, 'train_x'), Batcher(train_y, 'train_y'), Batcher(test_x, 'test_x'), Batcher(test_y, 'test_y')


train_x, train_y, test_x, test_y = get_data()

data_features = len(test_x.data[0])
n_classes = 2

print 'The number of training examples is:', train_x.num_examples
print 'The number of test examples is:', test_x.num_examples
print 'N classes is:', n_classes
print 'Number of features is:', data_features

batch_size = 100

print 'The number of batch loops is:', int(train_x.num_examples / batch_size)

n_nodes_hl1 = 846


x = tf.placeholder(tf.float32, [None, data_features])
y_ = tf.placeholder(tf.float32, [None, n_classes])

layer1_weights = tf.Variable(tf.random_normal([data_features, n_nodes_hl1]))
layer1_bias = tf.Variable(tf.random_normal([n_nodes_hl1]))

layer_1 = tf.matmul(x, layer1_weights) + layer1_bias
layer_1 = tf.nn.sigmoid(layer_1)

output_weights = tf.Variable(tf.random_normal([n_nodes_hl1, n_classes]))
output_bias = tf.Variable(tf.random_normal([n_classes]))

output = tf.matmul(layer_1, output_weights) + output_bias
# output = tf.nn.sigmoid(output)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))

optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

epochs = 100

sess = tf.Session()

with sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        epoch_loss = 0
        for _ in range(int(train_x.num_examples / batch_size)):
            epoch_x = train_x.next_batch(batch_size)
            epoch_y = train_y.next_batch(batch_size)
            # print 'This should be 100:', len(epoch_x)
            # print 'This should be 423:', len(epoch_x[0])
            _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y_: epoch_y})
            epoch_loss += c
            train_x.reset_next_last_batch()
            train_y.reset_next_last_batch()
        print 'Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss

    correct = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    # print 'This is y[0]:', sess.run(y)
    # print 'This is len(y[0]):', len(y[0])
    print 'Accuracy:', accuracy.eval({x: test_x.data, y_: test_y.data})

print 'Done'
