import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

class DQNAgent:
    def __init__(self, env, gamma, epsilon, epsilon_decay, learning_rate):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.env.observation_space.shape[0], activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.env.action_space.n, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def learn(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
        target_f = self.model.predict(state)
        target_f[0][action] = target
        self.model.fit(state, target_f, epochs=1, verbose=0)

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.env.observation_space.shape[0]])
                self.learn(state, action, reward, next_state, done)
                state = next_state
            self.epsilon *= self.epsilon_decay
            print(f"Episode {episode+1}, Epsilon: {self.epsilon:.2f}")

    def play(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
            done = False
            rewards = 0
            while not done:
                action = np.argmax(self.model.predict(state)[0])
                state, reward, done, _ = self.env.step(action)
                            state = np.reshape(state, [1, self.env.observation_space.shape[0]])
                rewards += reward
            print(f"Episode {episode+1}, Reward: {rewards}")

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = DQNAgent(env, gamma=0.99, epsilon=1.0, epsilon_decay=0.99, learning_rate=0.001)
    agent.train(episodes=1000)
    agent.play(episodes=10)