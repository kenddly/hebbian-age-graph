import numpy as np

SEED = 42

class BipartiteGraph:
    def __init__(self, num_inputs, num_outputs,
                 age=0.0,
                 trace_decay=0.85,
                 base_lr=0.003,
                 seed=SEED):
        self.generator = np.random.default_rng(seed)
        self.age = age
        self.weights = self.generator.uniform(-0.1, 0.1, (num_inputs, num_outputs))
        self.traces = np.zeros_like(self.weights)
        self.crystallization = np.zeros_like(self.weights, dtype=bool)
        self.trace_decay = trace_decay
        self.reward_baseline = 0.0

        # Age related parameters
        age_factor = np.exp(-0.05 * age)
        self.lr = base_lr * age_factor
        self.da_scale = age_factor

        self.temperature = 1.0 + 2.0 * np.exp(-0.1 * age)  # range: ~1.0 (old) to ~3.0 (young)
        self.crystallization_threshold = max(0.5, 4.0 - 0.3 * self.age)
        
        # Rigidity (Young = Takes full punishment, Old = 90% resistant)
        # Age 0 -> 1.0 (Fluid). Age 10 -> 0.1 (Stubborn)
        self.rigidity = max(0.1, 1.0 - 0.05 * self.age)


    def seed(self, seed):
        self.generator = np.random.default_rng(seed)

    def _bound_state(self, state):
        return np.tanh(state)

    def forward(self, state):
        bounded_state = self._bound_state(state)
        logits = bounded_state @ self.weights
        probs = self._softmax(logits * self.temperature)
        action = self.generator.choice(len(probs), p=probs)

        self.traces *= self.trace_decay
        self.traces[:, action] += bounded_state
        return action

    def predict(self, state):
        bounded_state = self._bound_state(state)
        return np.argmax(bounded_state @ self.weights)

    def apply_reward(self, reward):
        # Advantage: only signal above running average drives plasticity
        advantage = reward - self.reward_baseline
        # Asymmetric tracking: catch upward spikes fast, decay slowly on bad runs
        self.reward_baseline += 0.01 * advantage
        self.reward_baseline = np.clip(self.reward_baseline, -1.0, 1.0) 

        dopamine = np.tanh(self.da_scale * advantage)

        # Normalize traces to prevent instability
        col_norms = np.linalg.norm(self.traces, axis=0, keepdims=True) + 1e-8
        normalized_traces = self.traces / col_norms

        delta = self.lr * dopamine * normalized_traces

        # If dopamine is negative, apply rigidity to crystallized weights
        if dopamine < 0:
            delta = np.where(self.crystallization, delta * self.rigidity, delta)

        self.weights = np.clip(self.weights + delta, -5.0, 5.0)

        # Update crystallization status
        self.crystallization = np.abs(self.weights) > self.crystallization_threshold


    def reset_traces(self):
        self.traces.fill(0)

    def diagnostics(self):
        return {
            "crystallized": int(self.crystallization.sum()),
            "w_mean": float(self.weights.mean()),
            "w_max": float(self.weights.max()),
            "w_min": float(self.weights.min()),
            "reward_baseline": float(self.reward_baseline),
            "temperature": float(self.temperature)
        }

    def get_weights(self):
        return (self.weights)

    def set_weights(self, weights):
        self.weights = weights

    @staticmethod
    def _softmax(x):
        x = np.clip(x, -20.0, 20.0)
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

