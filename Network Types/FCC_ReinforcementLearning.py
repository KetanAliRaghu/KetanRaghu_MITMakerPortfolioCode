"""
Free Code Camp Reinforcement Learning
(Specifically Q-Learning)
"""

import gym # OpenAI Reinforcement Learning library
import numpy as np
import time
import matplotlib.pyplot as plt

"""
env = gym.make('FrozenLake-v1')
print(env.observation_space.n) # get number of states
print(env.action_space.n) # get number of actions

env.reset() # resets environment back to beginning state

action = env.action_space.sample() # get random action
observation , reward , done , info = env.step(action) # performs the action in the environment (observation is new state)
# env.render() # renders GUI for environment
"""

env = gym.make('FrozenLake-v1')
STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES , ACTIONS))

EPISODES = 10000 # how many times to run the environment
MAX_STEPS = 100 # max number of steps allowed for each run of environment

LEARNING_RATE = 0.81
GAMMA = 0.96

epsilon = 0.9 # start with 90% chance of random action
"""
# code to pick action
if np.random.uniform(0 , 1) < epsilon:
    action = env.action_space.sample()
else:
    action = np.argmax(Q[state, :]) # Idk either, state just isn't defined
"""

RENDER = False

rewards = []

for episode in range(EPISODES):

    state = env.reset()
    for _ in range(MAX_STEPS):
        if RENDER:
            env.render()

        if np.random.uniform(0 , 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        next_state , reward , done , _ = env.step(action)
        Q[state , action] = Q[state , action] + LEARNING_RATE * (reward + GAMMA * np.max(Q[next_state, :]) - Q[state , action])
        state = next_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break

print(Q)
print(f'Average Reward:', sum(rewards)/len(rewards))

avg_rewards = []
for i in range(0 , len(rewards) , 100):
    avg_rewards.append(sum(rewards[i : i + 100])/len(rewards[i : i + 100]))

plt.plot(avg_rewards)
plt.ylabel('average rewards')
plt.xlabel('episodes (100\'s')
plt.show()
