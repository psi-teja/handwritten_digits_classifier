import numpy as np
import pickle as pkl
from tqdm import trange
from keras.datasets import mnist
from utils import preprocess_data_for_training, sigmoid, softmax, d_sigmoid


# loading MNIST dataset
(X_train, y_train), (_, _) = mnist.load_data()

N = len(X_train)
epochs = 100
batch_size = 1000
learning_rate = 0.001
lamda_l2 = 0.002

# designing MLPs
n_in = 784
n_hl1 = 128
n_hl2 = 128
n_out = 10

# Random initialization of weights and biases
w1 = 2 * np.random.random((n_in, n_hl1)) - 1
b1 = 2 * np.random.random((1, n_hl1)) - 1
w2 = 2 * np.random.random((n_hl1, n_hl2)) - 1
b2 = 2 * np.random.random((1, n_hl2)) - 1
wo = 2 * np.random.random((n_hl2, n_out)) - 1
bo = 2 * np.random.random((1, n_out)) - 1

# load pre-trained model 
# model_weights_path = "weights/model_weights.pkl"
# with open(model_weights_path, 'rb') as file:
#     weights = pkl.load(file)
#     w1, w2, wo, b1, b2, bo = weights

# preprocessing training data
X_train, Y_train = preprocess_data_for_training(X_train, y_train)

epoch_list = []

for j in trange(epochs, desc="Training In Progress", unit="epochs"):
    i = 0
    error = 0
    np.random.shuffle(X_train)
    while i + batch_size < N:
        X_batch = X_train[i:i + batch_size]

        # forward pass
        hl1_in = np.dot(X_batch, w1) + b1
        hl1_a = sigmoid(hl1_in)

        hl2_in = np.dot(hl1_a, w2) + b2
        hl2_a = sigmoid(hl2_in)

        out_in = np.dot(hl2_a, wo) + bo
        out_a = softmax(out_in)

        # compute loss
        E = (Y_train[i:i + batch_size] - out_a)

        error += np.sum(np.square(E))

        slope_out = d_sigmoid(out_a)
        slope_hl1 = d_sigmoid(hl1_a)
        slope_hl2 = d_sigmoid(hl2_a)

        # Backpropagation and gradient descent
        d_out = E * slope_out
        Error_at_hl2 = np.dot(d_out, wo.transpose())
        d_hl2 = Error_at_hl2 * slope_hl2
        Error_at_hl1 = np.dot(d_hl2, w2.transpose())
        d_hl1 = Error_at_hl1 * slope_hl1

        grad_wo = np.dot(hl2_a.transpose(), d_out)   - 2*lamda_l2*wo
        grad_w2 = np.dot(hl1_a.transpose(), d_hl2)   - 2*lamda_l2*w2
        grad_w1 = np.dot(X_batch.transpose(), d_hl1) - 2*lamda_l2*w1

        grad_bo = d_out.sum(axis=0)   - 2*lamda_l2*bo
        grad_b2 = d_hl2.sum(axis=0)   - 2*lamda_l2*b2
        grad_b1 = d_hl1.sum(axis=0)   - 2*lamda_l2*b1

        wo += grad_wo * learning_rate
        w2 += grad_w2 * learning_rate
        w1 += grad_w1 * learning_rate

        bo += grad_bo * learning_rate
        b2 += grad_b2 * learning_rate
        b1 += grad_b1 * learning_rate

        i += batch_size
    print(error)

# Save the trained weights and biases
weights = [w1, w2, wo, b1, b2, bo]
pkl.dump(weights, open('np_model/weights/model_weights.pkl', 'wb'))
