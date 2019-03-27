
"""
Algebraic representation

By:
- Kevin Nijmeijer
- Wiebe van Breukelen

Used the Keras MMIST example as inspiration: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

The network should exists out of:
- 784 inputs (in case of a 28 by 28 pixel image)
- 10 outputs (one of each digit/class)

The network may be used without hidden layers.

We've tested various type of layers, dropouts and activation functions as shown in the table below:

+----------------------------+-----------------+------------------+-------+----------+
| Layers (excl. input layer) |     Dropout     | Layer Activation | Loss  | Accuracy |
+----------------------------+-----------------+------------------+-------+----------+
| 3 (MLP)                    | 0.25, 0.5, 0.75 | ReLU             | 2.658 |    0.834 |
| 2 (MLP)   0.25, 0.5        | ReLU            | 1.178            | 0.921 |          |
| 2 (MLP)                    | 0.25, 0.25      | ReLU             | 1.104 |    0.930 |
| 1 (SLP)                    | 0.25            | sigmoid          | 0.185 |    0.944 | <-- BEST PERFORMANCE/ACCURACY RATIO
| 2 (MLP)                    | 0.25, 0.25      | tanh             | 0.171 |    0.947 |
| 2 (MLP)                    | 0.25, 0.25      | sigmoid          | 0.163 |    0.949 | <-- BEST ACCURACY
+----------------------------+-----------------+------------------+-------+----------+


We can conclude that the use of the Sigmoid activation in combination with one hidden layer and a dropout of gives the best accuracy.
However, one may consider using a single layer network for better performance and slightly worse accuracy. Note that in all tests softmax is used
as activation function for the output layer.

Note: RMSprop is used for optimalisation and resulted in a time decrease higher than 90%.

"""
from __future__ import print_function
import keras

# Import dependencies.

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

batch_size = 128
num_classes = 10
epochs = 12
layer_activation = 'sigmoid'  # Other options are for example relu and sigmoid

# Input image dimensions.
img_rows, img_cols = 28, 28

# Split data between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape to the amount of pixels, for example: 28 * 28 = 784. These are the input neurons.
x_train = x_train.reshape(len(x_train), img_rows * img_cols)
x_test = x_test.reshape(len(x_test), img_rows * img_cols)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Initialize the sequential neural network.
model = Sequential()
model.add(Dense(128, activation=layer_activation,
                input_shape=((img_rows * img_cols),)))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

# Generate the underlying TensorFlow model.
model.compile(loss=keras.losses.categorical_crossentropy,
              # Use RMSprop optimizer for faster computation.
              optimizer=RMSprop(),
              metrics=['accuracy'])

# Train the model.
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Display the results upon the user's screen.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
