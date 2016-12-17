"""
Initial meta test:
- learn a meta network that outputs the weights of another network
    + the meta network takes as input the same input as the low-level network
    + the meta network outputs the mean and sigma of all the low-level weights
    + the weights are sampled and backprop is through reparameterization trick
- the low-level network objective is to autoencode its input
    + the input is gaussian noise of variable dimension
        * this can optionally be pos/neg 1
    + loss is squared error
"""

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True, precision=5)
import tensorflow as tf

# dataset, network, training constants
num_samples = 10000
batch_size = 32

input_dim = 5
output_dim = 5
meta_hidden_dim = 64
hidden_dim = 32
weight_dim = input_dim * hidden_dim + hidden_dim + hidden_dim * output_dim + output_dim
data_type = 'pos_neg_one'

num_epochs = 10000
learning_rate = 0.001

# generate data
if data_type == 'gaussian':
    inputs = np.random.randn(num_samples * input_dim)
elif data_type == 'pos_neg_one':
    inputs = np.random.randn(num_samples * input_dim)
    inputs[inputs < 0] = -1
    inputs[inputs > 0] = +1
else:
    raise(ValueError('invalid data type: {}'.format(data_type)))
inputs = inputs.reshape(-1, input_dim)
targets = inputs

# create a session and computational graph
with tf.Session() as sess:

    # trainable weights
    w = tf.get_variable('w', (input_dim, meta_hidden_dim), 
        initializer=tf.contrib.layers.variance_scaling_initializer(
        factor=2.0, mode='FAN_IN', uniform=False))
    b = tf.get_variable('b', (meta_hidden_dim,), 
        initializer=tf.constant_initializer(0.1))
    mu_w = tf.get_variable('mu_w', (meta_hidden_dim, weight_dim), 
        initializer=tf.contrib.layers.variance_scaling_initializer(
        factor=1.0, mode='FAN_AVG', uniform=False))
    mu_b = tf.get_variable('mu_b', (weight_dim,), 
        initializer=tf.constant_initializer(0.0))
    sigma_w = tf.get_variable('sigma_w', (meta_hidden_dim, weight_dim), 
        initializer=tf.contrib.layers.variance_scaling_initializer(
        factor=1.0, mode='FAN_AVG', uniform=False))
    sigma_b = tf.get_variable('sigma_b', (weight_dim,), 
        initializer=tf.constant_initializer(0.0))

    # placeholders
    input_ph = tf.placeholder(tf.float32,
                    shape=(None, input_dim),
                    name="input_ph")
    target_ph = tf.placeholder(tf.float32,
                    shape=(None, output_dim),
                    name="target_ph")
    noise_ph = tf.placeholder(tf.float32,
                    shape=(None, weight_dim),
                    name="noise_ph")

    # get output weights, and reshape to matrix form for each batch
    meta_h = tf.nn.relu(tf.matmul(input_ph, w) + b)
    mu = tf.matmul(meta_h, mu_w) + mu_b
    log_variance = tf.matmul(meta_h, sigma_w) + sigma_b
    sigma = tf.sqrt(tf.exp(log_variance))
    weights = noise_ph * sigma + mu

    # slice out the weights and biases for each layer
    start = 0
    size = input_dim * hidden_dim
    hidden_weights = tf.slice(weights, [0, start], [batch_size, size])
    hidden_weights = tf.reshape(hidden_weights, (batch_size, input_dim, hidden_dim))
    start += size

    size = hidden_dim
    hidden_bias = tf.slice(weights, [0, start], [batch_size, size])
    hidden_bias = tf.reshape(hidden_bias, (batch_size, 1, hidden_dim))
    start += size

    size = hidden_dim * output_dim
    output_weights = tf.slice(weights, [0, start], [batch_size, size])
    output_weights = tf.reshape(output_weights, (batch_size, hidden_dim, output_dim))
    start += size

    size = output_dim
    output_bias = tf.slice(weights, [0, start], [batch_size, size])
    output_bias = tf.reshape(output_bias, (batch_size, 1, output_dim))
    start += size

    # get output predictions of shape = (batch_size, output_dim)
    prediction_input = tf.reshape(input_ph, (batch_size, 1, input_dim))
    h = tf.batch_matmul(prediction_input, hidden_weights) + hidden_bias
    h = tf.nn.relu(h)
    scores = tf.batch_matmul(h, output_weights) + output_bias
    scores = tf.squeeze(scores, [1])

    # mse
    loss = tf.reduce_sum((scores - target_ph) ** 2)

    # training op
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # intialize variables
    sess.run(tf.initialize_all_variables())

    # run the training
    for epoch in range(num_epochs):
        epoch_losses = []

        # shuffle the data
        idxs = np.random.permutation(num_samples)
        inputs = inputs[idxs]
        targets = targets[idxs]

        # run training batches
        num_batches = int(num_samples / batch_size)
        for bidx in range(num_batches):

            # gather a training batch
            s = bidx * batch_size
            e = s + batch_size
            noise = np.random.randn(batch_size * weight_dim).reshape(
                batch_size, weight_dim)

            # run the batch 
            feed = {input_ph: inputs[s:e, :],
                    target_ph: targets[s:e, :],
                    noise_ph: noise}
            graph_outputs = [train_op, loss]
            _, loss_val = sess.run(graph_outputs, feed_dict=feed)

            # collect outputs
            epoch_losses.append(loss_val)

        print('epoch: {} loss: {}'.format(epoch, np.mean(epoch_losses)))

