import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

model_weights_path = "np_model/weights/model_weights.pkl"
with open(model_weights_path, 'rb') as file:
    weights = pkl.load(file)
    w1, w2, wo, b1, b2, bo = weights

arr = wo

# Display the 2D array as an image
plt.imshow(arr, cmap='gray')
plt.axis('off')  # Turn off axis labels
plt.show()


