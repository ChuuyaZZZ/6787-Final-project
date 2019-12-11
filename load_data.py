from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

def loadData(dimension):
    datagen = ImageDataGenerator(
            rescale=1./255, 
            rotation_range=45,
            width_shift_range=0.15,
            height_shift_range=0.15,
            horizontal_flip=True,
            fill_mode='nearest',
            shear_range=0.2,
            zoom_range=0.5
            )

    cwd = os.getcwd()
    train_dir = os.path.join(cwd, '500_data', 'train')
    valid_dir = os.path.join(cwd, '500_data', 'valid')

    train_generator = datagen.flow_from_directory(
                                                directory=train_dir, 
                                                target_size=(dimension,dimension),
                                                batch_size=32, 
                                                shuffle=True,
                                                seed=42, 
                                                class_mode='categorical')

    valid_generator = datagen.flow_from_directory(
                                                directory=valid_dir, 
                                                target_size=(dimension,dimension),
                                                batch_size=16, 
                                                shuffle=False, 
                                                class_mode='categorical')
    return train_generator, valid_generator                                      
