import numpy as np

def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    y = np.zeros((len(x), len(x.transpose())))
    for i in range(len(x)):
        for j in range(len(x.transpose())):
            if x[i][j] > 0:
                y[i][j] = 1
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x):
    S = np.exp(x)
    T = np.sum(S, axis=1, keepdims=True)
    y = S / T
    return y


def d_softmax(x):
    S = softmax(x)
    np.diag(S)
    S_vector = S.reshape(S.shape[0], 1)
    S_matrix = np.tile(S_vector, S.shape[0])
    return np.diag(S) - (S_matrix * np.transpose(S_matrix))
