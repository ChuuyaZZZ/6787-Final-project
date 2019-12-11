from keras.utils import np_utils
from keras.models import Sequential
from keras import models, layers, optimizers
from keras import backend as K
from keras.layers.normalization import BatchNormalization
import keras
import matplotlib.pyplot as plt
from load_data import loadData
import tensorflow as tf
from load_data import plotImages
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

def randomModel():
    model_fine = tf.keras.models.Sequential()
    pre_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3), alpha=1.0, include_top=False, weights='imagenet')
    model_fine.add(pre_model)
    model_fine.add(tf.keras.layers.Flatten())
    model_fine.add(tf.keras.layers.Dense(64, activation='relu'))
    model_fine.add(tf.keras.layers.Dropout(0.4))
    model_fine.add(tf.keras.layers.Dense(32, activation='relu'))
    model_fine.add(tf.keras.layers.Dense(7, activation='softmax'))
    model_fine.summary()
    return model_fine

def AlexNetModel():
    #Instantiate an empty model
    model = Sequential()

    # C1 Convolutional Layer
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224,224,3), padding="valid"))
    # S2 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # C3 Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(11, 11), strides=(1, 1), activation='relu', padding="valid"))
    # S4 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    # C5 Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid"))
    model.add(BatchNormalization())

    # C6 Convolutional Layer
    model.add(layers.Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid"))
    model.add(BatchNormalization())

    # C7 Convolutional Layer
    model.add(layers.Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="valid"))
    # S2 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(BatchNormalization())

    model.add(layers.Flatten())

    # FC8 Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(BatchNormalization())

    # FC9 Fully Connected Layer
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(BatchNormalization())

    #Output Layer with softmax activation
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.4))
    model.add(BatchNormalization())

    model.add(layers.Dense(7, activation='softmax'))  

    model.summary()
    # Compile the model
    # sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.99, nesterov=True)
    # adam = optimizers.Adam(0.001, 0.99, 0.999)
    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model

train_generator, valid_generator = loadData(224)
model = AlexNetModel()
model.compile(optimizer="adagrad", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
label_map1 = (train_generator.class_indices)
label_map2 = (valid_generator.class_indices)
print(label_map1)
print(label_map2)
model.fit_generator(generator=train_generator, validation_data=valid_generator,epochs=1, steps_per_epoch=20, validation_steps=20)
sample_training_images, _ = next(valid_generator)
plotImages(sample_training_images[5:10])
pred=model.predict(sample_training_images[5:10])
print(pred.shape)
print(pred)
