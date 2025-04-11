import numpy as np

class QLearningAgent:
    def __init__(self, grid_size, epsilon=1.0, alpha=0.1, gamma=0.9, epsilon_decay=0.995, min_epsilon=0.1):
        self.grid_size = grid_size
        self.q_table = {}
        self.actions = [0, 1, 2, 3]  # up, down, left, right
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))

        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions))
        if next_key not in self.q_table:
            self.q_table[next_key] = np.zeros(len(self.actions))

        old_value = self.q_table[state_key][action]
        next_max = np.max(self.q_table[next_key])

        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state_key][action] = new_value

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
