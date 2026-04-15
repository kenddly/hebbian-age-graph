import numpy as np
from collections import deque

# ── constants ─────────────────────────────────────────────
GRID = 10
MAX_STEPS = 200

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
TURN_R = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
TURN_L = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}

FEATURE_NAMES = [
    "danger_front", "danger_right", "danger_left",
    "food_front", "food_back", "food_left", "food_right",
    "bias"
]


class SnakeEnv:
    def __init__(self, grid=GRID):
        self.grid = grid
        self.REWARD_DEATH = -10.0
        self.REWARD_FOOD = 10.0
        self.reset()

    def reset(self):
        mid = self.grid // 2
        self.snake = deque([(mid, mid), (mid, mid - 1)])
        self.direction = RIGHT
        self.food = self._place_food()
        self.steps = 0
        self.done = False

        self.min_dist_to_food = self._distance(self.snake[0], self.food)
        return self._state()

    def _distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _place_food(self):
        cells = [(r, c) for r in range(self.grid) for c in range(self.grid)
                 if (r, c) not in self.snake]
        return cells[np.random.randint(len(cells))]


    def _state(self):
        """
        Returns a 10x10 grid representation for the visual encoder.
        0.0 = Empty
        0.5 = Snake Body
        1.0 = Snake Head
        -1.0 = Food
        """
        grid = np.zeros((self.grid, self.grid), dtype=np.float32)
        
        # 1. Draw the food
        fr, fc = self.food
        grid[fr, fc] = -1.0
        
        # 2. Draw the body
        for i, (r, c) in enumerate(self.snake):
            if i == 0:
                grid[r, c] = 1.0  # Head is brightest
            else:
                grid[r, c] = 0.5  # Body is dimmer
                
        return grid


    def step(self, action):
        if action == 1:
            self.direction = TURN_R[self.direction]
        elif action == 2:
            self.direction = TURN_L[self.direction]

        dr, dc = DELTAS[self.direction]
        new_head = (self.snake[0][0] + dr, self.snake[0][1] + dc)

        if (new_head[0] < 0 or new_head[0] >= self.grid or
            new_head[1] < 0 or new_head[1] >= self.grid or
            new_head in self.snake):
            return self._state(), self.REWARD_DEATH, True

        self.snake.appendleft(new_head)

        if new_head == self.food:
            reward = self.REWARD_FOOD
            self.food = self._place_food()
            self.min_dist_to_food = self._distance(new_head, self.food)
        else:
            self.snake.pop()
            new_dist = self._distance(new_head, self.food)

            if new_dist < self.min_dist_to_food:
                self.min_dist_to_food = new_dist
                reward = 1.0
            else:
                reward = -0.5

        self.steps += 1
        return self._state(), reward, self.steps >= MAX_STEPS
