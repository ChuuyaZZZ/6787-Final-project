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
import torchvision.models as models

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


def LENET5model():
    #Instantiate an empty model
    model = Sequential()
    # C1 Convolutional Layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=(64,64,3), padding="same"))
    # S2 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    # C3 Convolutional Layer
    model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"))
    # S4 Pooling Layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model.add(layers.Flatten())
    # FC5 Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    #Output Layer with softmax activation
    model.add(layers.Dense(7, activation='softmax'))
    # Compile the model
    sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.99, nesterov=True)
    adam = optimizers.Adam(0.001, 0.99, 0.999)
    return model

def plot(epochs_range, acc, val_acc, loss, val_loss):

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def getData():
    epochs_range = []
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    i = 0
    with open("file") as f:
        for line in f:
            line = line.split(":")
            if len(line) == 5:
                epochs_range.append(i)
                i += 1
                acc.append(float(line[2].split()[0]))
                val_acc.append(float(line[4]))
                loss.append(float(line[1].split()[0]))
                val_loss.append(float(line[3].split()[0]))
    return acc, val_acc, loss, val_loss, epochs_range


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
    # model.add(BatchNormalization())

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
    # model.add(BatchNormalization())

    model.add(layers.Dense(7, activation='softmax'))  

    model.summary()
    # Compile the model
    # sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.99, nesterov=True)
    # adam = optimizers.Adam(0.001, 0.99, 0.999)
    # model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
    return model

# acc, val_acc, loss, val_loss, epochs_range = getData()
# accN, val_accN, lossN, val_lossN, epochs_rangeN = [], [], [], [], []
# for i in range(len(acc)//4):
#     epochs_rangeN.append(4*i)
#     accN.append(sum(acc[4*i:4*i+4])/4)
#     val_accN.append(sum(val_acc[4*i:4*i+4])/4)
#     lossN.append(sum(loss[4*i:4*i+4])/4)
#     val_lossN.append(sum(val_loss[4*i:4*i+4])/4)

# plot(epochs_rangeN, accN, val_accN, lossN, val_lossN)
model = models.alexnet(pretrained=True)
train_generator, valid_generator = loadData(224)
# model = AlexNetModel()
# model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=False)
# model = models.alexnet(pretrained=True)
model.compile(optimizer="adagrad", loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit_generator(generator=train_generator, validation_data=valid_generator,epochs=200, steps_per_epoch=50, validation_steps=20)

