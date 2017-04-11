import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_lfw_people
import tensorflow as tf
import constants as c
import pickle

def run_epochs(n_epoch=10, batch_size=32):

    n_batch = int(len(data)/batch_size)
    print("Num Batches: {}".format(n_batch))
    costs = []

    for i_epoch in range(n_epoch):

        indexes = np.arange(len(data))
        np.random.shuffle(indexes)

        batch_costs = []

        for i_batch in range(n_batch):

            sindx = indexes[i_batch*batch_size:(i_batch+1)*batch_size]

            c, _ = sess.run([ae['cost'], grad_op], feed_dict={
                ae['x']: data[sindx]
            })

            batch_costs.append(c)

        mean_cost = np.mean(batch_costs)
        costs.append(mean_cost)

        print 'Epoch %i, Average Loss: %f' % (i_epoch, mean_cost)

    return costs

def autoencoder(layer_sizes=[784, 512, 256, 64]):
    """
    Input:
        layer_sizes: Number of neurons in encoding model of AE
    Output:
        Dictionary of input tensor (x), encoded layer tensor (z),
        output layer tensor (y) and cost function (cost)
    """
    x = tf.placeholder(tf.float32, [None, layer_sizes[0]])

    ae = {'x': x}

    pl = x
    encoder_weights = []
    enc_biases = []
    dec_biases = []
    for i, ls in enumerate(layer_sizes[1:]):

        w = tf.Variable(tf.random_uniform([layer_sizes[i], ls],
                                     -1./np.sqrt(layer_sizes[i]), 1./np.sqrt(layer_sizes[i])))
        encoder_weights.append(w)
        b = tf.Variable(tf.zeros([ls]))
        enc_biases.append(b)

        pl = tf.sigmoid(tf.matmul(pl, w) + b)

    ae['z'] = pl
    ae['encoder_w'] = encoder_weights
    ae['enc_biases'] = enc_biases

    for w, ls in zip(reversed(encoder_weights), reversed(layer_sizes[:-1])):

        b = tf.Variable(tf.zeros([ls]))
        pl = tf.sigmoid(tf.matmul(pl, tf.transpose(w)) + b)
        dec_biases.append(b)

    ae['y'] = pl
    ae['dec_biases'] = dec_biases

    cost = tf.reduce_mean(tf.square(ae['x'] - ae['y']))
    ae['cost'] = cost

    return ae

if __name__ == "__main__":

    plt.rcParams['image.cmap'] = 'gray'
    lfw_people = fetch_lfw_people(min_faces_per_person=50, resize=0.4)
    s = lfw_people.images.shape

    data = np.reshape(lfw_people.images,(s[0], s[1]*s[2]))
    data = data/255.

    layer_sizes = [data.shape[1], 128]
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    ae = autoencoder(layer_sizes=layer_sizes)
    grad_op = tf.train.AdamOptimizer(0.003).minimize(ae['cost'])

    init = tf.global_variables_initializer()
    sess.run(init)

    costs = run_epochs(220, 20)

    # Saving
    encoder_weights = sess.run(ae['encoder_w'])
    enc_biases = sess.run(ae['enc_biases'])
    dec_biases = sess.run(ae['dec_biases'])
    save_me = {'weights': encoder_weights,
               'costs': costs,
               'layers': layer_sizes,
               'enc_biases': enc_biases,
               'dec_biases': dec_biases}

    pickle.dump( save_me, open( "ae.p", "wb" ) )


    # Showing Results
    # rec_ae = sess.run(ae['y'], feed_dict={ae['x']: data})
    # idx = np.random.randint(len(rec_ae))

    # plt.figure(figsize=(4, 4))

    # plt.subplot(1, 2, 1)
    # plt.imshow(data[idx].reshape(s[1], s[2]))
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(rec_ae[idx].reshape(s[1], s[2]))
    # plt.axis('off')

    # plt.show()

