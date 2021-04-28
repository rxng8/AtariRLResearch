
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from .utils import *
IMG_SHAPE = (105, 80)
def atari_model(action_size, input_shape=(*IMG_SHAPE, 1)):
    
    inp = layers.Input(shape=input_shape)
    tensor = down_sample_layer(16)(inp)
    tensor = down_sample_layer(32)(tensor)
    tensor = layers.Flatten()(tensor)
    tensor = layers.Dense(256, activation='relu')(tensor)
    tensor = layers.Dense(64, activation='relu')(tensor)
    tensor = layers.Dense(16, activation='relu')(tensor)
    out = layers.Dense(action_size, activation='sigmoid')(tensor)

    model = tf.keras.Model(inp, out)
    return model


def atari_model_2(n_actions, ATARI_SHAPE=(105, 80, 4)):
    # We assume a theano backend here, so the "channels" are first.
    
    # With the functional API we need to define the inputs.
    frames_input = keras.layers.Input(ATARI_SHAPE, name='frames')
    
    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = keras.layers.convolutional.Convolution2D(
        16, 8, 8, subsample=(4, 4), activation='relu'
    )(frames_input)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = keras.layers.convolutional.Convolution2D(
        32, 4, 4, subsample=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = keras.layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = keras.layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = keras.layers.Dense(n_actions)(hidden)

    model = keras.models.Model(input=frames_input, output=output)
    optimizer = optimizer=keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
    model.compile(optimizer, loss='mse')
    return model