from environments.env import SnakeEnv, GRID

class ReversalEnv(SnakeEnv):
    def __init__(self, grid=GRID, max_steps=1000, seed=42):
        super().__init__(grid, max_steps, seed)
        self.is_reversed = False
        self.reset()
        self.steps = 0

    def reverse_controls(self, reverse=True):
        print(f"Reversing controls: {reverse}")
        self.is_reversed = reverse

    def step(self, action):
        # If reversed, the agent's motor controls invert (Right turns Left, Left turns Right)
        if self.is_reversed:
            if action == 1:
                action = 2
            elif action == 2:
                action = 1

        return super().step(action)
