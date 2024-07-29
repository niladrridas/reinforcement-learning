#### **Robotics AI Project**
##### "Empowering robots to navigate the world, one step at a time."

**Libraries Used:**
* ![Gym](https://img.shields.io/badge/Gym-0047AB?style=for-the-badge&logo=gym)
* ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy)
* ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow)
* ![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras)
* ![AI](https://img.shields.io/badge/AI-009688?style=for-the-badge&logo=ai)

# Reinforcement Learning Library
=====================================================

A project demonstrating the implementation of reinforcement learning algorithms using Python and the OpenAI Gym library.

## Table of Contents

- [RL](#rl)
- [Mathematical](#mathematical)
- [Introduction](#introduction)
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Markov Decision Process (MDP)
A Markov Decision Process (MDP) is a mathematical framework used to model decision-making problems in situations where outcomes are partially random. It's a key concept in reinforcement learning.

## Components of an MDP

**States (S)**
Represents the possible situations an agent can be in.

**Actions (A)**
Represents the possible actions an agent can take.

**Transition Function (P)**
Defines the probability of transitioning to a new state given the current state and action.

**Reward Function (R)**
Represents the expected reward received after transitioning to a new state given the current state and action.

**Policy (π)**
Defines the agent's behavior, determining the probability of taking each action in each state.

## Project Objectives
- Implement reinforcement learning algorithms to solve real-world problems
- Evaluate the performance of different algorithms on various MDPs
- Develop a framework for applying RL to real-life projects

## Project Structure
The project is organized into the following directories:

- `docs`: contains documentation for the project
- `src`: contains the source code for the project
- `tests`: contains unit tests for the project
- `examples`: contains example use cases for the project

## Reinforcement Learning Algorithms
The project implements the following reinforcement learning algorithms:

- Q-Learning
- SARSA
- Deep Q-Networks (DQN)
- Policy Gradient Methods

## RL
Python with the OpenAI Gym library to apply RL concepts:

1. **Install OpenAI Gym:**

```bash
pip install gym
```

2. **Creating an Environment:**

```python
import gym

env = gym.make('CartPole-v1')
state = env.reset()
```

3. **Define a Simple Policy:**

```python
def simple_policy(state):
    return 0 if state[2] < 0 else 1
```

4. **Run the Environment with the Policy:**

```python
for _ in range(1000):
    env.render()
    action = simple_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        state = env.reset()
env.close()
```

5. **Implementing a Q-Learning Algorithm:**

```python
import numpy as np
import random

q_table = np.zeros([env.observation_space.shape[0], env.action_space.n])

alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 1000

for episode in range(episodes):
    state = env.reset()
    done = False
    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        next_state, reward, done, _ = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("Training finished.\n")
```

6. **Testing the Trained Policy:**

```python
state = env.reset()
for _ in range(1000):
    env.render()
    action = np.argmax(q_table[state])
    state, reward, done, _ = env.step(action)
    if done:
        break
env.close()
```

## Training the Agent
To train the agent, call the train method:

```python
agent.train(num_episodes, max_steps)
```

## Evaluating the Agent
To evaluate the agent, call the evaluate method:

```python
agent.evaluate(num_episodes, max_steps)
```

## Example Use Case
Here is an example use case for the Q-Learning algorithm:

```python
import gym
from src.q_learning import QLearning

env = gym.make('CartPole-v0')
agent = QLearning(env, alpha=0.1, gamma=0.9, epsilon=0.1)

agent.train(num_episodes=1000, max_steps=1000)
agent.evaluate(num_episodes=100, max_steps=1000)
```

This code trains a Q-Learning agent on the CartPole environment for 1000 episodes, and then evaluates its performance for 100 episodes.

## Mathematical
### Markov Decision Process (MDP)

- **States ((S))**: Represents the possible situations an agent can be in.
- **Actions ((A))**: Represents the possible actions an agent can take.
- **Transition Function ((P))**: Defines the probability of transitioning to state (s') from state (s) by taking action (a). $$P(s', s, a) = \mathbb{P}(s_{t+1} = s' | s_t = s, a_t = a)$$
- **Reward Function ((R))**: Represents the expected reward received after transitioning from state (s) to state (s') by taking action (a). $$R(s, a) = \mathbb{E}[r_{t+1} | s_t = s, a_t = a]$$
- **Policy ((\pi)**)**: Defines the agent's behavior, the probability of taking action (a) in state (s). $$\pi(a|s) = \mathbb{P}(a_t = a | s_t = s)$$

### Value Functions

- **State-Value Function ((V^\pi(s)))**: The expected return when starting in state (s) and following policy (\pi) $$V^\pi(s) = \mathbb{E}\pi [ \sum{t=0}^\infty \gamma^t r_{t+1} | s_0 = s ]$$.
- **Action-Value Function ((Q^\pi(s, a)))**: The expected return when starting in state (s), taking action (a), and following policy (\pi) $$Q^\pi(s, a) = \mathbb{E}\pi [ \sum{t=0}^\infty \gamma^t r_{t+1} | s_0 = s, a_0 = a ]$$.

### Bellman Equations

- **Bellman Expectation Equation for (V^\pi)**: $$V^\pi(s) = \sum_{a \in A} \pi(a|s) \sum_{s' \in S} P(s'|s, a) [ R(s, a, s') + \gamma V^\pi(s') ]$$.
- **Bellman Expectation Equation for (Q^\pi)**: $$Q^\pi(s, a) = \sum_{s' \in S} P(s'|s, a) [ R(s, a, s') + \gamma \sum_{a' \in A} \pi(a'|s') Q^\pi(s', a') ]$$.

### Bellman Optimality Equations

- **Optimal State-Value Function (V^*)**: $$V^(s) = \max_a \sum_{s' \in S} P(s'|s, a) [ R(s, a, s') + \gamma V^(s') ]$$.
- **Optimal Action-Value Function (Q^*)**: $$Q^(s, a) = \sum_{s' \in S} P(s'|s, a) [ R(s, a, s') + \gamma \max_{a'} Q^(s', a') ]$$.

## Introduction
This project includes implementations of various reinforcement learning algorithms such as Q-learning and Deep Q-Networks (DQNs). The primary goal is to provide a comprehensive resource for understanding and applying RL algorithms to real-life projects.

## Installation
Follow these steps to set up the project:

```bash
# Clone the repository
git clone https://github.com/niladrridas/rl-library.git

# Navigate to the project directory
cd rl-library

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Features
- Implementation of basic RL algorithms (Q-learning, Deep Q-Networks)
- Easy-to-follow code examples
- Comprehensive documentation and tutorials
- Integration with OpenAI Gym environments

## Usage
To use the project, import the desired algorithm and create an instance of it:

```python
from src.q_learning import QLearning

agent = QLearning(env, alpha, gamma, epsilon)
```

## Guidelines
- Make sure to follow the project's coding style and conventions.
- Write unit tests for any new code.
- Document any new features or changes.

## Contributing
We welcome contributions from the community. To contribute:

1. Fork the repository
2. Create a new branch (git checkout -b feature-branch)
3. Commit your changes (git commit -am 'Add new feature')
4. Push to the branch (git push origin feature-branch)
5. Create a new Pull Request

## Code Review
All pull requests will be reviewed by the project maintainers. We will check for:

- Code quality and style.
- Unit tests.
- Documentation.

## Issues
If you find a bug or have a feature request, please open an issue on the project's issue tracker.

## License
This project is licensed under the MIT License - see the [LICENSE](/LICENSE) file for details.

## Contact
Niladri Das - [Gmail](mailto:ndas1262000@gmail.com)

Project Link: https://github.com/niladrridas/rl-library