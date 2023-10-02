import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt

# Create the environment
env = gym.make('Your_Env_Name_Here')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_high = env.action_space.high[0]

# Create the DDPG agent
agent = DDPGAgent(state_dim, action_dim, action_high)

# Training loop
num_episodes = 100  # Modify as needed
rewards_history = []

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(200):  # Modify max time steps per episode as needed
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        agent.train()
        agent.update_target_networks(tau)
        total_reward += reward
        state = next_state

        if done:
            break

    rewards_history.append(total_reward)

    # Plot the rewards over episodes
    plt.plot(rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('DDPG Learning Progress')
    plt.show()
