import numpy as np
import pickle as pkl
from tqdm import trange
from keras.datasets import mnist
from utils import preprocess_data, sigmoid, softmax, d_sigmoid


# loading MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

N = len(X_train)
epochs = 1000
batch_size = 1000
learning_rate = 0.1

# designing MLPs
n_in = 784
n_hl1 = 16
n_hl2 = 16
n_out = 10

# Random initialization of weights and biases
w1 = 2 * np.random.random((n_in, n_hl1)) - 1
b1 = 2 * np.random.random((1, n_hl1)) - 1
w2 = 2 * np.random.random((n_hl1, n_hl2)) - 1
b2 = 2 * np.random.random((1, n_hl2)) - 1
wo = 2 * np.random.random((n_hl2, n_out)) - 1
bo = 2 * np.random.random((1, n_out)) - 1

# preprocessing training data
X_train, Y_train = preprocess_data(X_train, y_train)


epoch_list = []

for j in trange(epochs, desc="Training In Progress", unit="epochs"):
    i = 0
    while i + batch_size < N:
        X_batch = X_train[i:i + batch_size]

        hl1_in = np.dot(X_batch, w1) + b1
        hl1_a = sigmoid(hl1_in)

        hl2_in = np.dot(hl1_a, w2) + b2
        hl2_a = sigmoid(hl2_in)

        out_in = np.dot(hl2_a, wo) + bo
        out_a = softmax(out_in)

        E = Y_train[i:i + batch_size] - out_a

        slope_out = d_sigmoid(out_a)
        slope_hl1 = d_sigmoid(hl1_a)
        slope_hl2 = d_sigmoid(hl2_a)

        d_out = E * slope_out
        Error_at_hl2 = np.dot(d_out, wo.transpose())
        d_hl2 = Error_at_hl2 * slope_hl2
        Error_at_hl1 = np.dot(d_hl2, w2.transpose())
        d_hl1 = Error_at_hl1 * slope_hl1

        wo = wo + np.dot(hl2_a.transpose(), d_out) * learning_rate
        w2 = w2 + np.dot(hl1_a.transpose(), d_hl2) * learning_rate
        w1 = w1 + np.dot(X_batch.transpose(), d_hl1) * learning_rate

        bo = bo + d_out.sum(axis=0) * learning_rate
        b2 = b2 + d_hl2.sum(axis=0) * learning_rate
        b1 = b1 + d_hl1.sum(axis=0) * learning_rate

        i += batch_size

# Save the trained weights and biases
weights = [w1, w2, wo, b1, b2, bo]
pkl.dump(weights, open('using_np/weights/model_weights.pkl', 'wb'))
