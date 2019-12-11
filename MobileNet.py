import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import itertools
from load_data import *

tf.keras.backend.set_session = tf.compat.v1.Session()

batch_size = 100
epochs = 100
IMG_SHAPE = 224 

pre_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape = (224,224,3), alpha = 1.0, include_top = False, weights = 'imagenet')

# Setting the layers of pre-trained model to be non trainable for transfer learning
for layer in pre_model.layers:
    layer.trainable = False
    
# Setting all layers of the pre-trained model to be trainable
# if want to test for non trainable, just comment the following two lines
'''
for layer in pre_model.layers:
    layer.trainable = True
'''
train_data_gen, val_data_gen = loadData(IMG_SHAPE)

model = tf.keras.models.Sequential()
model.add(pre_model)
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax'))
model.summary()

model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

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

pre_model.summary()
