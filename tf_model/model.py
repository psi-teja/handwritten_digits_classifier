from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

tf_model = models.Sequential()
# First Convolutional Layer
tf_model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding="same" ,input_shape=(28, 28, 1)))
tf_model.add(layers.MaxPooling2D((2, 2)))

tf_model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding="same"))
tf_model.add(layers.MaxPooling2D((2, 2)))
tf_model.add(layers.Flatten())
tf_model.add(layers.Dense(128, activation='relu'))
tf_model.add(layers.Dense(10, activation='softmax'))



