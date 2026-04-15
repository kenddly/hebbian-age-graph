import numpy as np


def top_k_sparse(x, k):
    if k >= len(x):
        return x

    idx = np.argpartition(x, -k)[-k:]
    out = np.zeros_like(x)
    out[idx] = x[idx]
    return out


class AgeingBipartiteGraph:
    def __init__(self, num_inputs, num_outputs,
                 age=0.0,
                 trace_decay=0.85,
                 base_lr=0.005,
                 crystallization_threshold=1.2,
                 rigidity=0.1,
                 baseline_lr=0.05):

        self.age = age
        self.weights = np.random.uniform(-0.1, 0.1, (num_inputs, num_outputs))
        self.traces = np.zeros_like(self.weights)
        self.crystallization = np.zeros_like(self.weights, dtype=bool)

        self.trace_decay = trace_decay
        self.crystallization_threshold = crystallization_threshold
        self.rigidity = rigidity

        age_factor = np.exp(-0.05 * age)
        self.lr = base_lr * age_factor
        self.da_scale = age_factor

        self.reward_baseline = 0.0
        self.baseline_lr = baseline_lr

        self.temperature = 1.0 + 2.0 * np.exp(-0.1 * age)

    def _bound_state(self, state):
        return np.tanh(state)

    def forward_action(self, state):
        """Used for output layer (samples action)."""
        bounded_state = self._bound_state(state)
        logits = bounded_state @ self.weights
        probs = self._softmax(logits * self.temperature)
        action = np.random.choice(len(probs), p=probs)

        # eligibility traces (action-specific)
        self.traces *= self.trace_decay
        self.traces[:, action] += bounded_state

        return action, logits

    def forward_hidden(self, state):
        """Hidden layer forward (no sampling)."""
        bounded_state = self._bound_state(state)
        h = bounded_state @ self.weights

        return h, bounded_state

    def apply_reward(self, reward):
        advantage = reward - self.reward_baseline
        self.reward_baseline += 0.01 * advantage

        dopamine = np.tanh(self.da_scale * advantage)

        col_norms = np.linalg.norm(self.traces, axis=0, keepdims=True) + 1e-8
        normalized_traces = self.traces / col_norms

        delta = self.lr * dopamine * normalized_traces
        delta = np.where(self.crystallization, delta * self.rigidity, delta)

        self.weights = np.clip(self.weights + delta, -5.0, 5.0)
        self.crystallization = np.abs(self.weights) > self.crystallization_threshold

    def reset_traces(self):
        self.traces.fill(0)

    @staticmethod
    def _softmax(x):
        x = np.clip(x, -20.0, 20.0)
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    def get_weights(self):
        return self.weights.copy()

    def set_weights(self, weights):
        self.weights = weights.copy()


class AgeingDeepGraph:
    def __init__(self, input_dim, hidden_dim, output_dim,
                 k=5,
                 age=0.0):

        self.age = age
        self.layer1 = AgeingBipartiteGraph(input_dim, hidden_dim, age)
        self.layer2 = AgeingBipartiteGraph(hidden_dim, output_dim, age)

        self.k = k

        # stored for credit assignment
        self.last_hidden = None
        self.last_hidden_raw = None
        self.last_action = None
        self.last_state = None

    def predict(self, state):
        return self.forward(state)

    def forward(self, state):
        self.last_state = state

        # ---- Layer 1 ----
        h_raw, bounded_state = self.layer1.forward_hidden(state)

        # 🔥 Top-k sparsity
        h_sparse = top_k_sparse(h_raw, self.k)

        self.last_hidden = h_sparse
        self.last_hidden_raw = h_raw

        # ---- Layer 2 ----
        action, _ = self.layer2.forward_action(h_sparse)
        self.last_action = action

        # ---- Update Layer1 traces (ONLY for active neurons) ----
        mask = (h_sparse != 0).astype(float)
        self.layer1.traces *= self.layer1.trace_decay
        self.layer1.traces += np.outer(bounded_state, mask)

        return action

    def apply_reward(self, reward):
        # ---- Layer 2 learns normally ----
        self.layer2.apply_reward(reward)

        # ---- Compute influence of hidden neurons on chosen action ----
        action = self.last_action
        h = self.last_hidden

        weights_to_action = self.layer2.weights[:, action]

        # influence score
        influence = weights_to_action * h

        # normalize influence
        norm = np.linalg.norm(influence) + 1e-8
        influence = influence / norm

        # gate: only neurons that positively contributed
        gate = (influence > 0).astype(float)

        # ---- Apply gate to layer1 traces ----
        self.layer1.traces *= gate

        # ---- Layer 1 learns (scaled reward helps stability) ----
        self.layer1.apply_reward(0.5 * reward)

    def reset_traces(self):
        self.layer1.reset_traces()
        self.layer2.reset_traces()

    def diagnostics(self):
        return {
            "layer1_crystallized": int(self.layer1.crystallization.sum()),
            "layer2_crystallized": int(self.layer2.crystallization.sum()),
            "hidden_active": int((self.last_hidden != 0).sum()) if self.last_hidden is not None else 0
        }

    def get_weights(self):
        return {
                "layer1": self.layer1.weights.copy(),
                "layer2": self.layer2.weights.copy()
                }

    def set_weights(self, weights):
        self.layer1.weights = weights["layer1"].copy()
        self.layer2.weights = weights["layer2"].copy()

