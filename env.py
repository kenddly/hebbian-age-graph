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
        head = self.snake[0]
        d = self.direction

        def is_hazard(pos):
            r, c = pos
            return r < 0 or r >= self.grid or c < 0 or c >= self.grid or pos in self.snake

        def step(pos, delta):
            return (pos[0] + delta[0], pos[1] + delta[1])

        # dangers
        danger = [
            is_hazard(step(head, DELTAS[d])),
            is_hazard(step(head, DELTAS[TURN_R[d]])),
            is_hazard(step(head, DELTAS[TURN_L[d]]))
        ]

        # food relative
        fr, fc = self.food
        hr, hc = head
        dr, dc = fr - hr, fc - hc

        if d == UP:
            f = [dr < 0, dr > 0, dc < 0, dc > 0]
        elif d == DOWN:
            f = [dr > 0, dr < 0, dc > 0, dc < 0]
        elif d == LEFT:
            f = [dc < 0, dc > 0, dr > 0, dr < 0]
        else:  # RIGHT
            f = [dc > 0, dc < 0, dr < 0, dr > 0]

        return np.array([*map(float, danger), *map(float, f), 1.0], dtype=np.float32)

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
                reward = -0.2

        self.steps += 1
        return self._state(), reward, self.steps >= MAX_STEPS
