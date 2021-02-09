# %%

import gym
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras import layers

from core.memory import ReplayBuffer
from core.utils import *

IMG_SHAPE = (105, 80)
BATCH_SIZE = 32


# Computation

def compute_epsilon(iteration: int) -> float:
    # Hard code!
    return 0.1

def choose_best_action(model, state) -> int:
    model
    return 0

def fit_batch(model, batch):
    pass

def conv_layer(out_channels, strides=1, activation='relu', padding='same'):
    layer = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(4, 4),
        activation=activation,
        padding=padding
    )
    return layer

def dropout_layer(rate=0.5):
    return layers.Dropout(rate)

def max_pooling_layer():
    return layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )

def down_sample_layer(out_channels):
    return tf.keras.Sequential([
        conv_layer(out_channels),
        max_pooling_layer(),
        dropout_layer()
    ])

def atari_model(action_size, input_shape=(*IMG_SHAPE, 1)):
    # Sai
    inp = layers.Input(shape=input_shape)
    tensor = down_sample_layer(16)(inp)
    tensor = down_sample_layer(32)(tensor)
    tensor = layers.Flatten()(tensor)
    tensor = layers.Dense(256, activation='relu')(tensor)
    tensor = layers.Dense(64, activation='relu')(tensor)
    tensor = layers.Dense(16, activation='relu')(tensor)
    out = layers.Dense(action_size, activation='sigmoid')(tensor)

    model = tf.keras.Model(inp, out)
    model.compile(optimizer='adam', loss='huber_loss', metrics=['acc'])

    return model

def atari_env(env_name='BreakoutDeterministic-v4'):
    return gym.make(env_name)

def q_step(env, 
        model, 
        state: np.ndarray, 
        iteration: int, 
        memory: ReplayBuffer, 
        batch_size :int=32):

    # use the iteration to compute 
    epsilon = compute_epsilon(iteration)

    # Choose action
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = choose_best_action(model, state)

    # Step in the environment
    observation, reward, terminal, info = env.step(action)

    # Add to the replay memory
    memory.add(state, action, reward, observation, terminal)

    # Create batch
    batch = memory.batch(batch_size)

    # Train batch
    fit_batch(model, batch)

# %%

# Test game
env = atari_env()

for i_episode in range(20):
    observation = env.reset()
    reward = 0
    for t in range(100):
        clear_output(wait=True)
        print("Game:", i_episode)
        # env.render()
        show_img(observation)
        print("Reward: ", reward)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        time.sleep(0.01)
env.close()

# %%





# %%

env = atari_env()

# %%

model = atari_model(env.action_space.n, (*IMG_SHAPE, 1))

# %%

model.summary()



# Train




