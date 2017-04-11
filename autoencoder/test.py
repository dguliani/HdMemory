import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
import tensorflow as tf
import pickle

import constants as c
from train import autoencoder

def encode(layer_sizes, data, encoders, biases):
    pl = data
    encoder_weights = encoders

    for i, ls in enumerate(layer_sizes[1:]):
        w = weights[i,:,:]
        b = biases[i,:]
        pl = tf.sigmoid(tf.matmul(pl, w) + b)

    return pl

def decode(layer_sizes, data, encoders, biases):
    pl = data
    encoder_weights = encoders

    for w, b, ls in zip(reversed(encoder_weights), reversed(biases), reversed(layer_sizes[:-1])):

        # return np.array([0])
        b = tf.zeros([ls])
        pl = tf.sigmoid(tf.matmul(pl, tf.transpose(w)) + b)

    return pl

plt.rcParams['image.cmap'] = 'gray'
lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
s = lfw_people.images.shape

data = np.reshape(lfw_people.images,(s[0], s[1]*s[2]))
data = data/255.

ae_load = pickle.load( open( "ae.p", "rb" ) )
layers = ae_load[c.PCKL_LAYERS_KEY]
weights = np.array(ae_load[c.PCKL_WEIGHTS_KEY])
enc_biases = np.array(ae_load['enc_biases'])
dec_biases = np.array(ae_load['dec_biases'])

tf.reset_default_graph()
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

idx = np.random.randint(len(data))
idx = 4
encoded = encode(layers, data[idx][None,:], weights, enc_biases)
decoded = decode(layers, encoded, weights, dec_biases)

cost = tf.reduce_mean(tf.square(data[idx] - decoded))
print "Cost: {}".format(sess.run(cost))

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(data[idx].reshape(s[1], s[2]))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(sess.run(decoded).reshape(s[1], s[2]))
plt.axis('off')

plt.show()
