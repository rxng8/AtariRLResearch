

from core.models.utils import mse
from tensorflow.python.keras.engine import training
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
    def __init__(self, action_size: int, eps_func) -> None:
        super().__init__()
        self.eps_func = eps_func
        self.action_size = action_size
        self.optimizer = tf.keras.optimizers.Adam()

    def best_action(self, model, state) -> int:
        q_vector = model([state])[0]
        return tf.argmax(q_vector)

    def fit_batch(self, model, batch, discount_rate: float=0.97):
        (states, actions, rewards, new_states, terminals), \
            importance, indices = batch

        with tf.device('/device:GPU:0'):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                pred_q_vec = model(states, training=True)  # Logits for this minibatch
                pred_q_value = pred_q_vec[actions]

                # Compute the loss value for this minibatch.
                true_q_value = rewards
                if not terminals:
                    true_q_value += discount_rate * model(new_states, training=False)
                loss = mse(true_q_value, pred_q_value)

                # print(f"loss for this batch at step: {step + 1}: {loss }")

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            self.optimizer.apply_gradients(zip(grads, model.trainable_weights))

    
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
        # (A tuple of states, actions, rewards, new_states, and terminals),
        # importance, indices.
        batch = memory.get_minibatch(batch_size)

        # Train batch
        self.fit_batch(model, batch)

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
