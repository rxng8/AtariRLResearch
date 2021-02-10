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
from core.data import *
from core.agent import *
from core.models import *

IMG_SHAPE = (105, 80)
BATCH_SIZE = 32


def atari_env(env_name='BreakoutDeterministic-v4'):
    return gym.make(env_name)


def compute_epsilon(iteration: int) -> float:
    # Hard code!
    return 0.1

def choose_best_action(model, state) -> int:
    model
    return 0

def fit_batch(agnet, batch):
    pass



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




