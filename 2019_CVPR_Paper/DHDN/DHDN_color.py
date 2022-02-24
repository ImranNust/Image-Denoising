import tensorflow as tf
from tensorflow.keras.layers import Conv2D, PReLU, concatenate, add
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.models import Model

def DCR_block(x, channel_in):
    """This function will implement the DCR Block"""
    out = Conv2D(filters=int(channel_in / 2.),
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 data_format="channels_last")(x)
    out = PReLU()(out)
    conc = concatenate([x, out])
    out = Conv2D(filters=int(channel_in / 2.),
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 data_format="channels_last")(conc)
    out = PReLU()(out)
    conc = concatenate([conc, out])
    out = Conv2D(filters=int(channel_in),
                 kernel_size=3,
                 strides=1,
                 padding='same',
                 data_format="channels_last")(conc)
    out = PReLU()(out)
    out = add([out, x])
    return out

def Down(x, channel_in):
    """This function will downsample the input"""
    out = MaxPooling2D()(x)

    out = Conv2D(filters=2 * channel_in,
                 kernel_size=1,
                 strides=1,
                 padding='valid',
                 data_format='channels_last')(out)
    out = PReLU()(out)
    return out


def Up(x, channel_in):
    """This function will upsample the input"""

    out = Conv2D(filters= channel_in,
                 kernel_size=1,
                 strides=1,
                 padding='valid',
                 data_format='channels_last')(x)
    out = PReLU()(out)
    out = tf.nn.depth_to_space(out, block_size=2)
    return out

def model_creation(input_shape = (64, 64, 3)):
    inputs = Input(shape=input_shape)
    out = Conv2D(filters=128,
                 kernel_size=1,
                 strides=1,
                 padding='valid',
                 data_format="channels_last")(inputs)
    out = PReLU()(out)
    out = DCR_block(out, channel_in=128)
    conc1 = DCR_block(out, channel_in=128)
    out = Down(conc1, channel_in=128)

    out = DCR_block(out, channel_in=256)
    conc2 = DCR_block(out, channel_in=256)
    out = Down(conc2, channel_in=256)

    out = DCR_block(out, channel_in=512)
    conc3 = DCR_block(out, channel_in=512)
    conc4 = Down(conc3, channel_in=512)

    out = DCR_block(conc4, channel_in=1024)
    out = DCR_block(out, channel_in=1024)
    out = concatenate([conc4, out])
    out = Up(out, channel_in = 2048)

    out = concatenate([conc3, out])
    out = DCR_block(out, channel_in=1024)
    out = DCR_block(out, channel_in=1024)
    out = Up(out, channel_in = 1024)

    out = concatenate([conc2, out])
    out = DCR_block(out, channel_in=512)
    out = DCR_block(out, channel_in=512)
    out = Up(out, channel_in = 512)

    out = concatenate([conc1, out])
    out = DCR_block(out, channel_in=256)
    out = DCR_block(out, channel_in=256)

    out = Conv2D(filters = 3, 
                 kernel_size=1, 
                 strides=1, 
                 padding='valid',
                 data_format="channels_last")(out)
    out = PReLU()(out)

    outputs = add([inputs, out])
    model = Model(inputs=inputs, outputs=outputs)
    
    return model