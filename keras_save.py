""" Saving and loading keras models
"""

from keras import Sequential
from keras.datasets import boston_housing
from keras.layers import Dense
from keras.models import load_model

# Load Dataset
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

my_layers = [
    Dense(64, input_shape=x_train.shape[1:], activation="relu"),
    Dense(64, activation="relu"),
    Dense(64, activation="relu"),
    Dense(1) # We are predicting 1 target
]

model = Sequential(my_layers)

model.compile(loss='mse',optimizer="adam", metrics=['mae'])

model.fit(x_train, y_train, epochs=500, validation_split=0.2)

scores = model.evaluate(x_test, y_test)
print(f"Our model is able to predict with a mean squared error of {scores[0]:.2f} and a mean absolute error of {scores[1]:.2f}.")

model.save("models/keras-boston.h5")

loaded_model = load_model("models/keras-boston.h5")
loaded_model_scores = loaded_model.evaluate(x_test, y_test)
print(f"Our loaded model is able to predict with a mean squared error of {loaded_model_scores[0]:.2f} and a mean absolute error of {loaded_model_scores[1]:.2f}.")