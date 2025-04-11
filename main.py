from environment import GridWorld
from q_learning import QLearningAgent
import matplotlib.pyplot as plt
import numpy as np
import pygame

GRID_SIZE = 10
EPISODES = 500
RENDER_EVERY = 50

env = GridWorld()
agent = QLearningAgent(grid_size=GRID_SIZE)

for episode in range(EPISODES):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if episode % RENDER_EVERY == 0:
            env.render()

    agent.decay_epsilon()
    print(f"Episode {episode + 1}/{EPISODES} | Total Reward: {total_reward} | Epsilon: {agent.epsilon:.3f}")

pygame.quit()

# Visualize Q-table
def plot_q_table(q_table):
    X, Y, U, V = [], [], [], []
    action_map = {
        0: (0, -1),   # up
        1: (0, 1),    # down
        2: (-1, 0),   # left
        3: (1, 0)     # right
    }

    for state, actions in q_table.items():
        x, y = state
        best_action = np.argmax(actions)
        dx, dy = action_map[best_action]
        X.append(x)
        Y.append(y)
        U.append(dx)
        V.append(-dy)  # invert y for plot direction

    plt.figure(figsize=(6, 6))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='blue')
    plt.xlim(-1, GRID_SIZE)
    plt.ylim(-1, GRID_SIZE)
    plt.gca().invert_yaxis()
    plt.grid(True)
    plt.title("Learned Policy (Q-Table Arrows)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

plot_q_table(agent.q_table)
