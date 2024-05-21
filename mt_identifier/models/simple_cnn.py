import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
#import hyperopt as hp
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append("../../python/")
import general_purpose_libs as gpl
import regression_libs as rl

def build_model(buid_parameters, train, validation, output_folder, input_shape):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(buid_parameters['n_filters'], (buid_parameters['kernel_size'], 1), activation='relu', input_shape=input_shape))

    for i in range(buid_parameters['n_conv_layers']):
        model.add(layers.Conv2D(buid_parameters['n_filters']//(i+1), (buid_parameters['kernel_size'], 1), activation='relu'))
        model.add(layers.LeakyReLU(alpha=0.05))
        model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    for i in range(buid_parameters['n_dense_layers']):
        model.add(layers.Dense(buid_parameters['n_dense_units']//(i+1), activation='relu'))
        model.add(layers.LeakyReLU(alpha=0.05))
    
    model.add(layers.Dense(1, activation='sigmoid'))  

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=buid_parameters['learning_rate'],
        decay_steps=10000,
        decay_rate=buid_parameters['decay_rate'])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='binary_crossentropy',
        metrics=['accuracy'])   

    # Stop training when `val_loss` is no longer improving
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            verbose=1)
    ]    

    history = model.fit(train, 
                        epochs=200,  
                        validation_data=validation, 
                        callbacks=callbacks,
                        verbose=1)

    return model, history

def create_and_train_model(model_parameters, train, validation, output_folder, model_name):
    input_shape = model_parameters['input_shape']
    build_parameters = model_parameters['build_parameters'] 
    model, history = build_model(build_parameters, train, validation, output_folder, input_shape)

    return model, history



