
from IPython.display import clear_output
from core.data.utils import preprocess
from core.models.utils import mse, huber
from tensorflow.python.keras.engine import training
from core import models
import random
import numpy as np
import tensorflow as tf
import time

from ..memory import ReplayBuffer
from ..models import atari_model
from core.data import show_img

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
        self.model = atari_model(action_size=self.action_size, input_shape=(*IMG_SHAPE, 1))

    def best_action(self, model, state) -> int:
        processed_state = tf.expand_dims(preprocess(state), axis=0)
        # processed_state = tf.broadcast_to(processed_state, [*processed_state.shape[0:3], 4])
        q_vector = model(processed_state, training=False)[0]
        return tf.argmax(q_vector)

    def fit_batch(self, model, batch, discount_rate: float=0.97):
        (states, action, reward, new_states, terminal), \
            importance, indices = batch
        batch_size = states.shape[0]
        # states = tf.expand_dims(states, axis=-1)
        # new_states = tf.expand_dims(new_states, axis=-1)
        assert states.shape == (batch_size, 105, 80, 1), "wrong shape"
        with tf.device('/device:GPU:0'):
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                pred_q_values = model(states, training=True)  # Logits for this minibatch
                # Create a mask so we only calculate loss on the updated Q-values
                one_hot_actions = tf.keras.utils.to_categorical(action, self.action_size, dtype=np.float32)
                # one_hot_actions = tf.one_hot(action_sample, num_actions)
                
                Q = tf.reduce_sum(tf.multiply(pred_q_values, one_hot_actions), axis=1)
                assert Q.shape == (batch_size), "Wrong shape"
                # Compute the loss value for this minibatch.
                # true_q_values = tf.broadcast_to(reward, [batch_size])

                future_rewards = model(new_states, training=False)
                # Q value = reward + discount factor * expected future reward
                updated_q_values = reward + discount_rate * tf.reduce_max(
                    future_rewards, axis=1
                )
                # If final frame set the last value to -1
                updated_q_values = updated_q_values * (1 - terminal) - terminal

                # loss = mse(true_q_values, Q)
                loss = huber(updated_q_values, Q)

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
        batch_size :int=32, # Batch size can be varied at times.
    ):
        # use the iteration to compute 
        epsilon = self.eps_func(iteration)

        # Choose action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = self.best_action(model, state)

        observations = np.empty(shape=(*state.shape[0:2], 0))
        # Step in the environment
        # for i in range(4):
        observation, reward, terminal, info = env.step(action)
        observation = preprocess(observation)
        observations = tf.concat([observations, observation], axis=-1)

        observations = tf.squeeze(observations, axis=2)
        assert observations.shape == (105,80), observations.shape
        # if state.shape[-1] == 4:
        #     for i in range(4):
                # Add to the replay memory
                # memory.add_experience(action, state[...,i:i+1], reward, terminal)
        assert state.shape == (105,80), state.shape
        memory.add_experience(action, state, reward, terminal)

        if iteration + 1 > batch_size:
            # Create batch
            # (A tuple of states, actions, rewards, new_states, and terminals),
            # importance, indices.
            batch = memory.get_minibatch(batch_size)

            # Train batch
            self.fit_batch(model, batch)

        return observations, terminal

    def q_eval(self, env, state, step, model):
        # use the iteration to compute 
        epsilon = self.eps_func(step)
        # Choose action
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = self.best_action(model, state)
        observations = np.empty(shape=(*state.shape[0:2], 0))
        # Step in the environment
        # for i in range(4):
        observation, reward, terminal, info = env.step(action)
        observation = preprocess(observation)
        observations = tf.concat([observations, observation], axis=-1)
        return observations, reward, terminal, info

    def train(self, env, epochs: int=1, steps: int=1000, render=False):
        memory = ReplayBuffer(input_shape=IMG_SHAPE, history_length=1)
        for epoch in range(epochs):
            if not render:
                print(f"Starting training epoch {epoch + 1}")
            observations = preprocess(env.reset())
            observations = tf.squeeze(observations, axis=2)
            assert observations.shape == (105,80), observations.shape
            terminal = False
            reward = 0
            for step in range(steps):
                if render:
                    clear_output(wait=True)
                    print(f"Starting training epoch {epoch + 1}")
                    # env.render()
                    show_img(observations)
                if not terminal:
                    observations, terminal = self.q_step(
                        env,
                        observations,
                        step,
                        self.model,
                        memory
                    )
                else:
                    break
                # time.sleep(0.01)
        return memory
    
    # deprecated
    def play(self, env, num_games=1, max_step=500, show=True):
        for game in range(num_games):
            observations = preprocess(env.reset())
            observations = tf.broadcast_to(observations, [*observations.shape[0:2], 1])
            terminal = False
            reward = 0
            for step in range(max_step):
                clear_output(wait=True)
                print("Game:", game)
                # env.render()
                show_img(observations[...,-1])
                if not terminal:
                    observations, reward, terminal, info = self.q_eval(
                        env,
                        observations,
                        step,
                        self.model
                    )
                else:
                    break