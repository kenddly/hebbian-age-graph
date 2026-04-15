import numpy as np
from models.graph import BipartiteGraph

class AgeingGraph(BipartiteGraph):
    def __init__(self, num_inputs, num_outputs,
                 age=0.0,
                 trace_decay=0.85,
                 base_lr=0.005,
                 crystallization_threshold=1.2,
                 rigidity=0.1,
                 baseline_lr=0.05,
                 ageing_threshold=100.0,
                 seed=42):
        super().__init__(num_inputs, num_outputs, age, trace_decay, base_lr,
                         crystallization_threshold, rigidity, baseline_lr, seed)

        self.clock = 0.0  # Simulated time in steps
        self.ageing_threshold = ageing_threshold  # Time steps after which ageing effects kick in

    def forward(self, state):
        action = super().forward(state)
        self.clock += 1.0
        if self.clock >= self.ageing_threshold:
            self.age_one_step()
            self.clock = 0.0
        return action

    def _update_age(self):
        age_factor = np.exp(-0.05 * self.age)
        self.lr = self.lr * age_factor
        self.da_scale = self.da_scale * age_factor
        self.temperature = 1.0 + 2.0 * np.exp(-0.1 * self.age)


    def reset_age(self):
        self.age = 0.0
        self._update_age()


    def age_one_step(self):
        self.age += 1
        self._update_age()


    def set_age(self, age):
        self.age = age
        self._update_age()

