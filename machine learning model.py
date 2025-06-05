import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Show the 1st training image
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()

# Normalize the data to range 0-1
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten the data (28x28 images to 784-length vectors)
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

# Build the model
model = keras.Sequential([
    keras.layers.Dense(32, activation="relu", input_shape=(784,)),  # hidden layer
    keras.layers.Dense(10, activation="softmax")  # output layer for 10 classes (digits 0-9)
])

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Get epoch amount from user input
epoch = int(input("Epoch amount: "))

# Train the model
model.fit(x_train, y_train, epochs=epoch, batch_size=32)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")

# Make a prediction on the first test image
image = x_test[0].reshape(1, 784)
prediction = model.predict(image)
label = np.argmax(prediction)

print(f"Predicted: {label}")
print(f"Actual: {y_test[0]}")

# Show the test image with the model prediction
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Model prediction: {label}")
plt.axis('off')
plt.show()

if label == y_test[0]:
    print("Correct :)")
else:
    print("Incorrect :(")
