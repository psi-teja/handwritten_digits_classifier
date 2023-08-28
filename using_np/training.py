import keras
import pickle as pkl

from tqdm import trange
from utils import *

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

N = 60000
epochs = 10000
batch_size = 1000

Y_train = np.zeros((N, 10))

for i in range(N): Y_train[i][y_train[i]] = 1

n_in = 784
n_hl1 = 16
n_hl2 = 16
n_out = 10

learning_rate = 0.1

# random initialization of weights
wh1 = 2 * np.random.random((n_in, n_hl1)) - 1
bh1 = 2 * np.random.random((1, n_hl1)) - 1
wh2 = 2 * np.random.random((n_hl1, n_hl2)) - 1
bh2 = 2 * np.random.random((1, n_hl2)) - 1
wout = 2 * np.random.random((n_hl2, n_out)) - 1
bout = 2 * np.random.random((1, n_out)) - 1

epoch_list = []

for j in trange(epochs, desc="Training In Progress", unit="epochs"):
    i = 0
    while i + batch_size < int(N):
        # for k in range((i*n):((i+1)*n)):

        Z = X_train[i:i + batch_size]
        Z = (255 - Z) / 255
        Z[Z > 0] = 1
        X = Z.reshape((batch_size, 784))

        bh1_copy = bh1.copy()
        bh2_copy = bh2.copy()
        bout_copy = bout.copy()

        hl1_in = np.dot(X, wh1)
        hl1_a = sigmoid(hl1_in)

        hl2_in = np.dot(hl1_a, wh2)
        hl2_a = sigmoid(hl2_in)

        out_in = np.dot(hl2_a, wout)
        out_a = softmax(out_in)

        E = Y_train[i:i + batch_size] - out_a

        # print(j, i, E.sum(axis=0))

        slope_out = d_sigmoid(out_in)
        slope_hl1 = d_sigmoid(hl1_in)
        slope_hl2 = d_sigmoid(hl2_in)

        # d_out = E * slope_out
        d_out = np.multiply(E, slope_out)

        Error_at_hl2 = np.dot(d_out, wout.transpose())

        d_hl2 = Error_at_hl2 * slope_hl2

        Error_at_hl1 = np.dot(d_hl2, wh2.transpose())

        d_hl1 = Error_at_hl1 * slope_hl1

        wout = wout + np.dot(hl2_a.transpose(), d_out) * learning_rate
        wh2 = wh2 + np.dot(hl1_a.transpose(), d_hl2) * learning_rate
        wh1 = wh1 + np.dot(X.transpose(), d_hl1) * learning_rate

        bout = bout + d_out.sum(axis=0) * learning_rate
        bh2 = bh2 + d_hl2.sum(axis=0) * learning_rate
        bh1 = bh1 + d_hl1.sum(axis=0) * learning_rate

        i += batch_size

pkl.dump(wh1, open('weights/wh1.pkl', 'wb'))
pkl.dump(wh2, open('weights/wh2.pkl', 'wb'))
pkl.dump(wout, open('weights/wout.pkl', 'wb'))
pkl.dump(bh2, open('weights/bh2.pkl', 'wb'))
pkl.dump(bh1, open('weights/bh1.pkl', 'wb'))
pkl.dump(bout, open('weights/bout.pkl', 'wb'))
