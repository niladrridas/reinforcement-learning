import numpy as np
import random
import gym

# Initialize the gym environment
env = gym.make('CartPole-v1')

# Initialize the Q-table to zeros
q_table = np.zeros([500, env.action_space.n])  # Discretized state space

def discretize_state(state):
    intervals = [0.25, 0.5, 0.5, 0.25]
    bins = [np.digitize(state[i], np.arange(-2.4, 2.4, intervals[i])) for i in range(len(state))]
    return sum([bins[i] * (len(intervals) ** i) for i in range(len(intervals))])

# Set hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

# Training loop
for episode in range(episodes):
    state = discretize_state(env.reset())
    done = False
    while not done:
        # Exploration-exploitation trade-off
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Take action and observe the outcome
        next_state, reward, done, _ = env.step(action)
        next_state = discretize_state(next_state)

        # Update Q-value
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training finished.\n")
