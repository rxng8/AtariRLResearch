

from core import models
import random
import numpy as np
import tensorflow as tf

from ..memory import ReplayBuffer
from ..models import atari_model

IMG_SHAPE = (105, 80)

class BaseAgent:
    def __init__(self) -> None:
        pass

class DQNAgent(BaseAgent):
    def __init__(self, model, action_size: int, eps_func) -> None:
        super().__init__()
        self.model = model
        self.eps_func = eps_func
        self.action_size = action_size

    def best_action(self, state) -> int:
        return self.model([state])[0]

    def fit_batch(self, batch):

        pass
    
    def q_step(self,
        env, # One agent can play in many env, env does not need to belong to an agent
        state: np.ndarray, # An agent can have different current state.
        iteration: int,
        model,
        memory: ReplayBuffer, # An agent can have different memory
        batch_size :int=32 # Batch size can be varied at times.
    ):
        # use the iteration to compute 
        epsilon = self.eps_func(iteration)

        # Choose action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = self.best_action(state)

        # Step in the environment
        observation, reward, terminal, info = env.step(action)

        # Add to the replay memory
        memory.add_experience(action, state, reward, terminal)

        # Create batch
        batch = memory.get_minibatch(batch_size)

        # Train batch
        self.fit_batch(batch, model)

        return observation, terminal

    def train(self, env, epochs: int=1, steps: int=200):
        memory = ReplayBuffer(input_shape=IMG_SHAPE)
        model = atari_model(action_size=self.action_size, input_shape=IMG_SHAPE)
        for epoch in range(epochs):
            observation = env.reset()
            terminal = False
            reward = 0
            for step in range(steps):
                if not terminal:
                    observation, terminal = self.q_step(
                        env,
                        observation,
                        step,
                        model,
                        memory
                    )
                else:
                    break
