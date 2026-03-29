import numpy as np

class AgeingBipartiteGraph:
    def __init__(self, num_inputs, num_outputs,
                 age=0.0,
                 trace_decay=0.9,
                 base_lr=0.1,
                 crystallization_threshold=2.0,
                 rigidity=0.5):

        self.age = age
        # 1. Allow initial weights to be negative (inhibitory and excitatory)
        self.weights = np.random.uniform(-0.1, 0.1, (num_inputs, num_outputs))
        self.traces = np.zeros_like(self.weights)
        self.crystallization = np.zeros_like(self.weights, dtype=bool)

        self.trace_decay = trace_decay
        self.crystallization_threshold = crystallization_threshold
        self.rigidity = rigidity

        age_factor = np.exp(-0.05 * age)
        self.lr = base_lr * age_factor
        self.da_scale = age_factor

    def _bound_state(self, state):
        # 2. Squash the raw latent features into a predictable [-1.0, 1.0] range
        return np.tanh(state)

    def forward(self, state):
        bounded_state = self._bound_state(state)
        
        logits = bounded_state @ self.weights
        # 3. Reduce the temperature multiplier. Without boolean inputs, 
        # a smaller multiplier (or removing it entirely) prevents softmax saturation.
        probs = self._softmax(logits * 2.0) 
        action = np.random.choice(len(probs), p=probs)

        self.traces *= self.trace_decay
        self.traces[:, action] += bounded_state
        return action

    def predict(self, state):
        bounded_state = self._bound_state(state)
        return np.argmax(bounded_state @ self.weights)

    def apply_reward(self, reward):
        dopamine = np.tanh(self.da_scale * reward)
        norm = np.linalg.norm(self.traces) + 1e-8

        delta = self.lr * dopamine * (self.traces / norm)

        if dopamine < 0:
            delta = np.where(self.crystallization, delta * self.rigidity, delta)

        # 4. Allow the network to learn strong inhibitory weights (-5.0)
        self.weights = np.clip(self.weights + delta, -5.0, 5.0)
        
        # 5. Check crystallization against the absolute magnitude of the weight
        self.crystallization = np.abs(self.weights) > self.crystallization_threshold

    def reset_traces(self):
        self.traces.fill(0)

    def diagnostics(self):
        return {
            "crystallized": int(self.crystallization.sum()),
            "w_mean": float(self.weights.mean()),
            "w_max": float(self.weights.max()),
            "w_min": float(self.weights.min())
        }

    @staticmethod
    def _softmax(x):
        # Safety bound to prevent exp() overflow just in case
        x = np.clip(x, -20.0, 20.0) 
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

class RBFHebbianNetwork:
    def __init__(self, num_inputs, num_outputs, num_rbf=50, age=0,):
        # RBF centers spread across input space
        self.centers = np.random.uniform(-1, 1, (num_rbf, num_inputs))
        self.spread = 0.5  # or learnable
        
        # Your existing Hebbian structure now operates on RBF features
        self.hebbian = AgeingBipartiteGraph(num_rbf, num_outputs, age=age)
    
    def _to_rbf(self, state):
        # Nonlinear transformation: state → RBF features
        dist = np.linalg.norm(state - self.centers, axis=1)
        return np.exp(-(dist ** 2) / (2 * self.spread ** 2))
    
    def forward(self, state):
        rbf_features = self._to_rbf(state)  # Nonlinear expansion
        return self.hebbian.forward(rbf_features)  # Unchanged Hebbian rules
    
    def apply_reward(self, reward):
        # No change needed—traces already captured in hebbian object
        self.hebbian.apply_reward(reward)

    def reset_traces(self):
        self.hebbian.reset_traces()

    def predict(self, state):
        rbf_features = self._to_rbf(state)
        return self.hebbian.predict(rbf_features)
