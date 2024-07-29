import gym
import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_decay):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.q_table = self.initialize_q_table()

    def initialize_q_table(self):
        q_table = {}
        for state in range(self.env.observation_space.n):
            q_table[state] = [0.0] * self.env.action_space.n
        return q_table

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        q_value = self.q_table[state][action]
        next_q_value = max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * next_q_value - q_value)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
            self.epsilon *= self.epsilon_decay
            print(f"Episode {episode+1}, Epsilon: {self.epsilon:.2f}")

    def play(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            rewards = 0
            while not done:
                action = np.argmax(self.q_table[state])
                state, reward, done, _ = self.env.step(action)
                rewards += reward
            print(f"Episode {episode+1}, Reward: {rewards}")

if __name__ == "__main__":
    env = gym.make("FrozenLake-v0")
    agent = QLearningAgent(env, alpha=0.1, gamma=0.9, epsilon=1.0, epsilon_decay=0.99)
    agent.train(episodes=1000)
    agent.play(episodes=10)