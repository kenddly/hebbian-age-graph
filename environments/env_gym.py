import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SnakeGymWrapper(gym.Env):
    def __init__(self, env):
        super().__init__()

        self.env = env

        # spaces (inferred from your env)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        obs = self.env.reset()
        return obs, {}

    def step(self, action):
        obs, reward, done = self.env.step(action)

        # Your env only has "done"
        terminated = done
        truncated = False  # you already encode max_steps inside done

        return obs, reward, terminated, truncated, {}

    def render(self):
        # optional: hook into your own visualization later
        pass

    def close(self):
        pass
