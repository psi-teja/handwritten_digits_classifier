import numpy as np
import pickle as pkl
from PIL import Image

class np_model:
    def __init__(self):
        self.wh1 = pkl.load(open('using_np/weights/wh1.pkl', 'rb'))
        self.wh2 = pkl.load(open('using_np/weights/wh2.pkl', 'rb'))
        self.bh1 = pkl.load(open('using_np/weights/bh1.pkl', 'rb'))
        self.bh2 = pkl.load(open('using_np/weights/bh2.pkl', 'rb'))
        self.wout = pkl.load(open('using_np/weights/wout.pkl', 'rb'))
        self.bout = pkl.load(open('using_np/weights/bout.pkl', 'rb'))

    def predict(self, X):

        X = X.reshape((1, 784))

        hl1 = np.dot(X, self.wh1) + self.bh1
        hl1 = 1 / (1 + np.exp(-hl1))
        hl2 = np.dot(hl1, self.wh2) + self.bh2
        hl2 = 1 / (1 + np.exp(-hl2))

        out = np.dot(hl2, self.wout) + self.bout
        S = np.exp(out)
        T = np.sum(S, axis=1, keepdims=True)
        out = S / T

        return out
