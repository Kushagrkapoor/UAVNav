import numpy as np
import tensorflow as tf

# Define the DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_high):
        self.actor = Actor(action_dim)
        self.target_actor = Actor(action_dim)
        self.critic = Critic()
        self.target_critic = Critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.002)

        self.action_high = action_high
        self.buffer = []

    # Define the get_action method for exploration
    def get_action(self, state):
        state = np.expand_dims(state, axis=0)
        action = self.actor(state).numpy()
        action += np.random.normal(0, 0.1, size=action.shape)
        action = np.clip(action, -1, 1)
        return action[0] * self.action_high

    # Define the remember method for experience replay
    def remember(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # Define the train method for updating the actor and critic networks
    def train(self, batch_size=64, gamma=0.99, tau=0.001):
        if len(self.buffer) < batch_size:
            return

        minibatch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            target_q_values = rewards + gamma * (1 - dones) * target_q_values

            predicted_q_values = self.critic(states, actions)
            critic_loss = tf.reduce_mean(tf.square(target_q_values - predicted_q_values))

        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic(states, actions_pred))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target_networks(tau)

    # Define the update_target_networks method for slowly updating target networks
    def update_target_networks(self, tau):
        self.target_actor.set_weights(
            [tau * actor_weight + (1 - tau) * target_actor_weight
             for actor_weight, target_actor_weight in zip(self.actor.get_weights(), self.target_actor.get_weights())]
        )

        self.target_critic.set_weights(
            [tau * critic_weight + (1 - tau) * target_critic_weight
             for critic_weight, target_critic_weight in zip(self.critic.get_weights(), self.target_critic.get_weights())]
        )
