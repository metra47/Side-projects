#need to practice more python before fully jumping in
# find a tutorial (chatgpt or yt???)
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
# these are used to save the exit number and then add 1 to it for the next training sequenece
import sys
import os
# this should load the number dataset
mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#this shows the 1st training image
plt.imshow(x_train[0],cmap = 'gray')
plt.title(f"Label: {y_train[0]}")
plt.axis('off')
plt.show()
# these steps divide it down to a range between 0-1 which is easier for python
# aka normalising the data
x_train = x_train/255
x_test = x_test/255
# flattens the data so it can interept it in a 1d style
#and the [0] is used so the code works with the database being any size
x_train = x_train.reshape(x_train.shape[0], 784)
#this does the same thing
x_test = x_test.reshape(x_test.shape[0], 784)
model = keras.Sequential([#sequential = it goes on after another
    #this is the neurons the num at the start is the number of neurons
    keras.layers.Dense(32, activation="relu", input_shape=(784,)),#hidden layer
    # relu is the function it goes through to help it learn better
    keras.layers.Dense(10, activation="softmax")#output layer we have 10 since there will be values from 0-9
    #softmax turns it into a probablilty ( in percentages)
])
# this defines how it will learn by classification, I think multiclass.
# it isn't linear regression , bcs the numbers aren't continous like temp
# Plus we know how many classes we have (10) and not an infinite amount
model.compile(
    loss="sparse_categorical_crossentropy",# loss tells it how wrong it is
    optimizer="adam",# this is just an algorithm that automatically changes the weights -> to reduce loss.
    metrics=["accuracy"] # this just tells us the accuracy -> in percentages
)
epoch = int(input("Epoch amount"))
model.fit(x_train, y_train, epochs = epoch, batch_size = 32)# this runs the training - 5 cycles with 32 samples per cycle
#tells me how well it has done
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")
#this finds the last exit code num
try:
    with open("exit_code.txt", "r") as file:
        last = int(file.read())
except:
    last = 0  # If no file exists, start at 0

new_index = last + 1
print("Current index:", new_index)

# Step 3: Save it for next time
with open("exit_code.txt", "w") as file:
    file.write(str(new_index))

# this gets the image and we don't need to normalise it cause we normalised it before
image = x_test[new_index]
# reshape (1 sample for 784 values)
image = image.reshape(1, 784)
# 3.gets the prediction with a array of 10 probabilites
prediction = model.predict(image.reshape(1, 784))
label = np.argmax(prediction)# this just picks the highest one = most likely one
# 5. Print prediction and compare to actual label
print(f"Predicted: {label}")# the f is just to make it easier with variables
print(f"Actual: {y_test[0]}")
#this displays the image so I can see it
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Model prediction: {label}")
plt.axis('off')
plt.show()
if label == y_test[0]:
    print("Correct:)")
else:
    print("Incorrect :(")
sys.exit(new_index)# this just exits with the index so we don't have to start from zero


