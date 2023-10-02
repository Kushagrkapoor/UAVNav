import numpy as np
import tensorflow as tf
import gym

# Define the actor and critic networks
class Actor(tf.keras.Model):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='tanh')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.output_layer(x)
