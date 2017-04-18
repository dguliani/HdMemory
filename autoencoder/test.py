import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
import tensorflow as tf
import pickle

import constants as c
import nengo
from nengo.dists import Uniform

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
lfw_people = fetch_lfw_people(min_faces_per_person=150, resize=0.65)
s = lfw_people.images.shape

# print s
# print lfw_people.target_names
# exit()

data = np.reshape(lfw_people.images,(s[0], s[1]*s[2]))
data = data/255.

ae_load = pickle.load( open( "ae.p", "rb" ) )
layers = ae_load[c.PCKL_LAYERS_KEY]
weights = np.array(ae_load[c.PCKL_WEIGHTS_KEY])
enc_biases = np.array(ae_load['enc_biases'])
dec_biases = np.array(ae_load['dec_biases'])
costs = np.array(ae_load['costs'])

tf.reset_default_graph()
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

idx = np.random.randint(len(data))
# idx = 4
encoded = encode(layers, data[idx][None,:], weights, enc_biases)
print sess.run(encoded).shape

# N = 4000
# tau_rc = 0.02
# tau_ref = 0.002
# lif_model = nengo.LIF(tau_rc=tau_rc, tau_ref=tau_ref)
# T = 1.5
# reccur_syn = 0.02
# in_syn = 0.005
# probe_syn = 0.01
# dim = 64
# model = nengo.Network(label="Neuron")
# with model:
#     # ab
#     # stim = nengo.Node(lambda t: 0.9 if (t > 0.04 and t < 1) else 0)
#     # c
#     # stim = nengo.Node(lambda t: 0.9 if (t > 0.04 and t < 0.16) else 0)
#     # d
#     # stim = nengo.Node(lambda t: (0.9/0.45)*t if t < 0.45 else 0)
#     # e
#     stim = nengo.Node(lambda t: np.array(sess.run(encoded))[0,:])

#     ensA = nengo.Ensemble(N, dimensions=dim,
#                           max_rates = Uniform(100,200),
#                           intercepts= Uniform(-1,1),
#                           neuron_type=lif_model)

#     ensB = nengo.Ensemble(N, dimensions=dim,
#                           max_rates = Uniform(100,200),
#                           intercepts= Uniform(-1,1),
#                           neuron_type=lif_model)

#     nengo.Connection(stim, ensA, function=lambda x: reccur_syn*x, synapse=in_syn)
#     nengo.Connection(ensA, ensA, synapse=reccur_syn)
#     # nengo.Connection(stim, ensA)

#     stim_p = nengo.Probe(stim, synapse=probe_syn)
#     ensA_p = nengo.Probe(ensA, synapse=probe_syn)



# sim = nengo.Simulator(model)
# sim.run(T)

# decode_me  = tf.cast(sim.data[ensA_p][-1,:][None,:], tf.float32)
# t = sim.trange()

decoded_nomem = decode(layers, encoded, weights, dec_biases)
# decoded = decode(layers, decode_me, weights, dec_biases)

# cost = tf.reduce_mean(tf.square(data[idx] - decoded))
cost_nomem = tf.sqrt(tf.reduce_mean(tf.square(data[idx] - decoded_nomem)))
print "Cost: {}".format(sess.run(cost_nomem))

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(data[idx].reshape(s[1], s[2]))
plt.axis('off')
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(sess.run(decoded_nomem).reshape(s[1], s[2]))
plt.axis('off')
plt.title('Image After Auto-Encoder')


plt.figure(figsize=(5, 5))
plt.plot(costs)
plt.title('RMSE vs Training Epoches')
plt.xlabel('Epoch')
plt.ylabel('Average RMSE')
plt.show()
