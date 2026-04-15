import numpy as np

class AgeingBipartiteGraphNonlinear:
    def __init__(self, num_inputs, num_outputs,
                 num_hidden=8,
                 age=0.0,
                 trace_decay=0.92,
                 lr=0.001,
                 crystallization_threshold=1.5,
                 rigidity=0.2,
                 baseline_lr=0.05):

        # ---------------- core params ----------------
        self.trace_decay = trace_decay
        self.lr = lr
        self.rigidity = rigidity
        self.crystallization_threshold = crystallization_threshold

        # Stable (no chaos from age)
        self.temperature = 1.2
        self.da_scale = 1.0

        # Reward baseline
        self.reward_baseline = 0.0
        self.baseline_lr = baseline_lr

        # ---------------- weights ----------------
        self.W1 = np.random.randn(num_inputs, num_hidden) * 0.1
        self.W2 = np.random.randn(num_hidden, num_outputs) * 0.1

        # ---------------- traces ----------------
        self.T1 = np.zeros_like(self.W1)
        self.T2 = np.zeros_like(self.W2)

        # ---------------- crystallization ----------------
        self.C1 = np.zeros_like(self.W1, dtype=bool)
        self.C2 = np.zeros_like(self.W2, dtype=bool)

    # --------------------------------------------------
    def _bound(self, x):
        return np.tanh(x)

    def _softmax(self, x):
        x = np.clip(x, -20, 20)
        x = x - np.max(x)
        e = np.exp(x)
        return e / np.sum(e)

    # --------------------------------------------------
    def forward(self, state):
        s = self._bound(state)

        # forward pass
        h = np.tanh(s @ self.W1)
        logits = h @ self.W2
        probs = self._softmax(logits / self.temperature)

        action = np.random.choice(len(probs), p=probs)

        # ---------------- traces update ----------------
        self.T1 *= self.trace_decay
        self.T2 *= self.trace_decay

        # policy gradient signal
        one_hot = np.zeros_like(probs)
        one_hot[action] = 1.0
        delta_out = one_hot - probs

        # W2 traces (clean)
        self.T2 += np.outer(h, delta_out)

        # W1 traces (proper credit assignment)
        backprop = self.W2 @ delta_out
        dh = (1 - h**2) * backprop   # tanh derivative
        self.T1 += np.outer(s, dh)

        return action

    # --------------------------------------------------
    def predict(self, state):
        s = self._bound(state)
        h = np.tanh(s @ self.W1)
        return int(np.argmax(h @ self.W2))

    # --------------------------------------------------
    def apply_reward(self, reward):
        # Advantage (variance reduction)
        advantage = reward - self.reward_baseline
        self.reward_baseline += self.baseline_lr * advantage

        dopamine = np.tanh(self.da_scale * advantage)

        # ---------------- W2 update ----------------
        col_norms = np.linalg.norm(self.T2, axis=0, keepdims=True) + 1e-8
        norm_T2 = self.T2 / col_norms

        delta_W2 = self.lr * dopamine * norm_T2
        delta_W2 = np.where(self.C2, delta_W2 * self.rigidity, delta_W2)

        self.W2 += delta_W2
        np.clip(self.W2, -5.0, 5.0, out=self.W2)

        # ---------------- W1 update (NO normalization) ----------------
        delta_W1 = self.lr * dopamine * self.T1
        delta_W1 = np.where(self.C1, delta_W1 * self.rigidity, delta_W1)

        self.W1 += delta_W1
        np.clip(self.W1, -5.0, 5.0, out=self.W1)

        # ---------------- crystallization ----------------
        self.C1 = np.abs(self.W1) > self.crystallization_threshold
        self.C2 = np.abs(self.W2) > self.crystallization_threshold

    # --------------------------------------------------
    def reset_traces(self):
        self.T1.fill(0)
        self.T2.fill(0)

    # --------------------------------------------------
    def diagnostics(self):
        return {
            "W1_mean": float(self.W1.mean()),
            "W2_mean": float(self.W2.mean()),
            "crystallized_W1": int(self.C1.sum()),
            "crystallized_W2": int(self.C2.sum()),
            "baseline": float(self.reward_baseline),
        }

    def get_weights(self):
        return (self.W1, self.W2)
    
    def set_weights(self, weights):
        self.W1, self.W2 = weights
