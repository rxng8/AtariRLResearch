# %%

from PIL.Image import new
import gym
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras import layers

from core.data import *
from core.agent import DQNAgent
from core.models import atari_model
from core.memory import ReplayBuffer

IMG_SHAPE = (105, 80)
BATCH_SIZE = 32

def eps_function(step):
    return 0.2

def atari_env(env_name='BreakoutDeterministic-v4'):
    return gym.make(env_name)

# env = atari_env(env_name='BreakoutNoFrameskip-v4')

env = atari_env()
agent = DQNAgent(env.action_space.n, eps_function)

# %%

# Test game
# for i_episode in range(20):
#     observation = env.reset()
#     reward = 0
#     for t in range(100):
#         clear_output(wait=True)
#         print("Game:", i_episode)
#         # env.render()
#         show_img(observation)
#         print("Reward: ", reward)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
#         time.sleep(0.01)
# env.close()


# Train

memory = agent.train(env, epochs=30, steps=1000)


# %%

memory = agent.train(env, epochs=1, steps=1000, render=True)

