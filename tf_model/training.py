from dataset import X_train, y_train
from model import tf_model

model = tf_model

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model for 10 epochs
model.fit(X_train, y_train, epochs=10)

# Save the trained model
model.save('pretrained_model')

# Make a prediction on a test sample
prediction = model.predict(X_train[:1])
print(prediction)