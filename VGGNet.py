from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Model
import argparse
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from load_data import *

tf.keras.backend.set_session = tf.compat.v1.Session()

batch_size = 100
epochs = 100
IMG_SHAPE = 224 

pre_model = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape = (224,224,3), pooling=None)

x = tf.keras.layers.Flatten()(pre_model.output)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dropout(0.4)(x)
# if want to test without batch normalization, just comment the following line
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
predictions = tf.keras.layers.Dense(7, activation = 'softmax')(x)

#create graph of your new model
model = Model(inputs = pre_model.input, outputs = predictions)
model.summary()

train_data_gen, val_data_gen = loadData(IMG_SHAPE)

model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

history = model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs = 50, steps_per_epoch=20, validation_steps=20)

score = model.evaluate_generator(val_data_gen)
print(score)

acc = history.history['categorical_accuracy']
val_acc = history.history['val_categorical_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

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
