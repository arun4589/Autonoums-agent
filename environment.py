import pygame

CELL_SIZE = 50
GRID_SIZE = 10
WIDTH = HEIGHT = CELL_SIZE * GRID_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class GridWorld:
    def __init__(self):
        pygame.init()
        self.win = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Autonomous Agent - Q Learning")

        self.clock = pygame.time.Clock()
        self.grid_size = GRID_SIZE
        self.start_pos = [0, 0]
        self.goal_pos = [GRID_SIZE - 1, GRID_SIZE - 1]
        self.agent_pos = self.start_pos.copy()
        self.agent_path = []

        self.obstacles = [
            (3, 3), (3, 4), (3, 5),
            (6, 6), (7, 6), (8, 6),
            (5, 2), (5, 3)
        ]

    def reset(self):
        self.agent_pos = self.start_pos.copy()
        self.agent_path = []
        return self.agent_pos.copy()

    def step(self, action):
        x, y = self.agent_pos
        if action == 0 and y > 0: y -= 1
        elif action == 1 and y < self.grid_size - 1: y += 1
        elif action == 2 and x > 0: x -= 1
        elif action == 3 and x < self.grid_size - 1: x += 1

        next_pos = [x, y]

        if tuple(next_pos) in self.obstacles:
            reward = -5
        else:
            self.agent_pos = next_pos
            reward = -1

        self.agent_path.append(self.agent_pos.copy())

        done = self.agent_pos == self.goal_pos
        if done:
            reward = 10

        return self.agent_pos.copy(), reward, done

    def render(self):
        self.win.fill(WHITE)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(self.win, BLACK, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

        for obs in self.obstacles:
            pygame.draw.rect(self.win, GRAY, (obs[0] * CELL_SIZE, obs[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for pos in self.agent_path:
            pygame.draw.rect(self.win, (255, 150, 150), (
                pos[0] * CELL_SIZE + CELL_SIZE // 4,
                pos[1] * CELL_SIZE + CELL_SIZE // 4,
                CELL_SIZE // 2,
                CELL_SIZE // 2
            ))

        pygame.draw.rect(self.win, RED, (self.agent_pos[0] * CELL_SIZE, self.agent_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.draw.rect(self.win, GREEN, (self.goal_pos[0] * CELL_SIZE, self.goal_pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        pygame.display.flip()
        self.clock.tick(60)
