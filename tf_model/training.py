from dataset import *
from model import model

tf_model = model

tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# tf_model.optimizer.lr.assign(0.005)
tf_model.fit(X_train, y_train, epochs=10)

tf_model.save('tf_model')

print(tf_model.predict(X_test[:1]))

