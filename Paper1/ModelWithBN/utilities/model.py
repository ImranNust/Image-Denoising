# import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Activation, Lambda, Subtract, concatenate,PReLU
from tensorflow.keras.layers import Add
import tensorflow.keras.backend as K



def rdb_block(inputs, numLayers = 3):
    channels = inputs.get_shape()[-1]
    storedOutputs = [inputs]
    for _ in range(numLayers):
        localConcat = tf.concat(storedOutputs, axis=-1)
        out = Conv2D(filters=channels, kernel_size=3, padding="same")(localConcat)
        out=PReLU()(out)
        storedOutputs.append(out)
        finalConcat = tf.concat(storedOutputs, axis=-1)
        finalOut = Conv2D(filters=inputs.get_shape()[-1], kernel_size=1,
                          padding="same")(finalConcat)
        final=PReLU()(finalOut)
    finalOut = Add()([final, inputs])
    return finalOut

def BRDNet(input_shape = (64, 64, 3),epsilon=1e-3, axis=-1, momentum=0.99,
           r_max_value=3., d_max_value=5., t_delta=1e-3, weights=None, beta_init='zero',
           gamma_init='one', gamma_regularizer=None, beta_regularizer=None,
          BatchReNormalization = False):
    input = Input(shape = input_shape)
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1),
              padding='same')(input)
    x = PReLU()(x)
    for i in range(7):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis = axis, 
                               scale=True,
                               momentum = momentum,
                               epsilon = epsilon,
                               renorm = BatchReNormalization,
                               renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
                               #                           renorm_momentum = 0.9,
                               beta_regularizer=beta_regularizer,
                               gamma_regularizer=gamma_regularizer,
                               gamma_initializer = gamma_init)(x)
      
        x = PReLU()(x)
        x = rdb_block(x)
    for i in range(8):
        x = BatchNormalization(axis = axis, 
                               scale=True,
                               momentum = momentum,
                               epsilon = epsilon,
                               renorm = BatchReNormalization,
                               renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
                               #                           renorm_momentum = 0.9,
                               beta_regularizer=beta_regularizer,
                               gamma_regularizer=gamma_regularizer,
                               gamma_initializer = gamma_init)(x)
        x = PReLU()(x)
        x = rdb_block(x)
    
    x = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([x, x])
    y = Conv2D(filters=64, kernel_size = (3,3), strides=(1,1), padding = 'same')(input)
    y = PReLU()(y)
    for i in range(7):
        y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',dilation_rate=(2,2))(y)
        y = BatchNormalization(axis = axis, 
                               scale=True,
                               momentum = momentum,
                               epsilon = epsilon,
                               renorm = BatchReNormalization,
                               renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
                               #                           renorm_momentum = 0.9,
                               beta_regularizer=beta_regularizer,
                               gamma_regularizer=gamma_regularizer,
                               gamma_initializer = gamma_init)(y)
        y = PReLU()(y)
        y = rdb_block(y)
    y = Conv2D(filters=64, kernel_size = (3,3), strides=(1,1), padding = 'same')(y)
    y = PReLU()(y)
    for i in range(6):
        y = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same',dilation_rate=(2,2))(y)
        y = BatchNormalization(axis = axis, 
                               scale=True,
                               momentum = momentum,
                               epsilon = epsilon,
                               renorm = BatchReNormalization,
                               renorm_clipping = {'rmax':r_max_value, 'dmax':d_max_value},
                               #                           renorm_momentum = 0.9,
                               beta_regularizer=beta_regularizer,
                               gamma_regularizer=gamma_regularizer,
                               gamma_initializer = gamma_init)(y)
        y = PReLU()(y) 
        y = rdb_block(y)
    y = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(y)#gray is 1 color is 3
    y = Subtract()([input, y])   # input - noise
    o = concatenate([x,y],axis=-1)
    z = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding='same')(o)#gray is 1 color is 3
    z=  Subtract()([input, z])
    model = Model(inputs=input, outputs=z)
    return model