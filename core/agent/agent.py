
from core import models
import random
import numpy as np
import tensorflow as tf

from ..memory import ReplayBuffer

class BaseAgent:
    def __init__(self) -> None:
        pass

class DQNAgent(BaseAgent):
    def __init__(self, model, action_space_size: int, eps_func) -> None:
        super().__init__()
        self.model = model
        self.eps_func = eps_func
        self.action_space_size = action_space_size

    def best_action(self, state) -> int:
        return self.model([state])[0]

    def train_agent(self):
        pass

    def fit_batch(self, batch):
        pass
    
    def q_step(self,
        env, # One agent can play in many env, env does not need to belong to an agent
        state: np.ndarray, # An agent can have different current state.
        iteration: int, 
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
        self.fit_batch(batch)


