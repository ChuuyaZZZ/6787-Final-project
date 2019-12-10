from keras.utils import np_utils
from keras.models import Sequential
from keras import models, layers, optimizers
from keras import backend as K
import keras
import matplotlib.pyplot as plt


def AlexNetModel():
    #Instantiate an empty model
    model = Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224,224,3), padding="same"))
    # S2 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # C3 Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"))
    # S4 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    # C5 Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))

    # C6 Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))

    # C7 Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
    # S2 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

    model.add(layers.Flatten())

    # FC8 Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    # FC9 Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))

    #Output Layer with softmax activation
    model.add(layers.Dense(1000, activation='softmax'))
    model.add(layers.Dropout(0.5))
    model.summary()
    # Compile the model
    # sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.99, nesterov=True)
    # adam = optimizers.Adam(0.001, 0.99, 0.999)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model


AlexNetModel()
