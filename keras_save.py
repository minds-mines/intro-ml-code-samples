""" Saving and loading keras models
"""

from keras import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils import to_categorical

# Load Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28
num_classes = 10 # 10 digits

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print(x_train.shape)
print(x_train.shape[1:])

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

my_layers = [
    Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=input_shape),
    Conv2D(64, kernel_size=(3,3), activation="relu"),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(num_classes, activation="softmax") # Number of classes as output units
]

model = Sequential(my_layers)

model.compile(loss='categorical_crossentropy',optimizer="adam", metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.2)

scores = model.evaluate(x_test, y_test)
print(f"Our model is able to predict with an accuracy of {scores[1]:.2f}.")

model.save("models/keras-mnist.h5")

loaded_model = load_model("models/keras-mnist.h5")
loaded_model_scores = loaded_model.evaluate(x_test, y_test)
print(f"Our loaded model is able to predict with an accuracy of {loaded_model_scores[1]:.2f}.")
