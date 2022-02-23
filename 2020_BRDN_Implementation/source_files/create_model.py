import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Activation, Lambda, Subtract, concatenate
from tensorflow.keras.layers import Add
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def BRDNet(input_shape, lr, epsilon=1e-3, axis=-1, momentum=0.99,
           r_max_value=3., d_max_value=5., t_delta=1e-3, weights=None, beta_init='zero',
                 gamma_init='one', gamma_regularizer=None, beta_regularizer=None):
    input = Input(shape = input_shape)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
              padding='same')(input)
#     x = BatchRenormalization(axis = -1, epsilon=1e-3)(x)
    x = BatchNormalization(axis = axis, 
                          scale=True,
                          momentum = momentum,
                          epsilon = epsilon,
                          renorm = True,
                          renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
#                           renorm_momentum = 0.9,
                          beta_regularizer=beta_regularizer,
                          gamma_regularizer=gamma_regularizer,
                          gamma_initializer = gamma_init)(x)
    x = Activation('relu')(x)
    
    # following lines will create 15 layers (Conv+BN+ReLU)
    for i in range(7):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis = axis, 
                          scale=True,
                          momentum = momentum,
                          epsilon = epsilon,
                          renorm = True,
                          renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
#                           renorm_momentum = 0.9,
                          beta_regularizer=beta_regularizer,
                          gamma_regularizer=gamma_regularizer,
                          gamma_initializer = gamma_init)(x)
        x = Activation('relu')(x)
        
        
    for i in range(8):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis = axis, 
                          scale=True,
                          momentum = momentum,
                          epsilon = epsilon,
                          renorm = True,
                          renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
#                           renorm_momentum = 0.9,
                          beta_regularizer=beta_regularizer,
                          gamma_regularizer=gamma_regularizer,
                          gamma_initializer = gamma_init)(x)
        x = Activation('relu')(x)
        
    x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([input, x])
    y = Conv2D(filters=64, kernel_size = (3,3), strides=(1,1), padding = 'same')(input)
    y = BatchNormalization(axis = axis, 
                          scale=True,
                          momentum = momentum,
                          epsilon = epsilon,
                          renorm = True,
                          renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
#                           renorm_momentum = 0.9,
                          beta_regularizer=beta_regularizer,
                          gamma_regularizer=gamma_regularizer,
                          gamma_initializer = gamma_init)(y)
    y = Activation('relu')(y)
    for i in range(7):
        y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),dilation_rate=(2,2), padding='same')(y)
        y = Activation('relu')(y)   
    y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(y)
    y = BatchNormalization(axis = axis, 
                          scale=True,
                          momentum = momentum,
                          epsilon = epsilon,
                          renorm = True,
                          renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
#                           renorm_momentum = 0.9,
                          beta_regularizer=beta_regularizer,
                          gamma_regularizer=gamma_regularizer,
                          gamma_initializer = gamma_init)(y)
    
    y = Activation('relu')(y) 
    for i in range(6):
        y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),dilation_rate=(2,2), padding='same')(y)
        y = Activation('relu')(y)
    y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(y)
    y = BatchNormalization(axis = axis, 
                          scale=True,
                          momentum = momentum,
                          epsilon = epsilon,
                          renorm = True,
                          renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
#                           renorm_momentum = 0.9,
                          beta_regularizer=beta_regularizer,
                          gamma_regularizer=gamma_regularizer,
                          gamma_initializer = gamma_init)(y)
    y = Activation('relu')(y)    
    y = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(y)#gray is 1 color is 3
    y = Subtract()([input, y])   # input - noise
    o = concatenate([x,y],axis=-1)
    z = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(o)#gray is 1 color is 3
    z=  Subtract()([input, z])
    model = Model(inputs=input, outputs=z)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr),
                 loss = tf.keras.losses.MeanSquaredError())
    return model

if __name__ == "__main__":
    print("Creating a Model with Default Values...")
    print(__name__)
    model = BRDNet()