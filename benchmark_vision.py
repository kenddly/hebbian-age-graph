import numpy as np
import torch

from vision_encoder import VisionEncoder
from env_vision import SnakeEnv, MAX_STEPS
from graph import AgeingBipartiteGraph
from plot import plot_results
from snake_visualizer import watch_agent

SEED = 42
N_EPISODES = 2000
EVAL_EVERY = 50
EVAL_EPS = 20

# Initialize and load the frozen encoder globally
encoder = VisionEncoder(latent_dim=10)
encoder.encoder.load_state_dict(torch.load("weights/vision_encoder.pth"))
frozen_encoder = encoder.encoder
frozen_encoder.eval()

# Create generic labels for the 10 latent dimensions for your plots
LATENT_FEATURE_NAMES = [f"Latent_{i}" for i in range(10)]


def encode_state(raw_grid, current_direction):
    """Combines the 10D visual latent vector with a 4D heading vector."""
    # 1. Get Visual Latents
    tensor_grid = torch.tensor(raw_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        latent_features = frozen_encoder(tensor_grid).squeeze().numpy()
        
    # 2. One-hot encode the heading
    heading = np.zeros(4, dtype=np.float32)
    heading[current_direction] = 1.0 # UP=0, DOWN=1, LEFT=2, RIGHT=3
    
    # 3. Concatenate into a 14-element vector
    return np.concatenate([latent_features, heading])


def run_episode(agent, env, train=True):
    raw_state = env.reset()
    # Pass direction here
    state = encode_state(raw_state, env.direction) 
    total_r, food = 0.0, 0

    for _ in range(MAX_STEPS):
        action = agent.forward(state) if train else agent.predict(state)
        
        raw_state, reward, done = env.step(action)
        # And pass direction here
        state = encode_state(raw_state, env.direction)

        if train:
            agent.apply_reward(reward)

        total_r += reward
        if reward >= 10:
            food += 1

        if done:
            break

    if train:
        agent.reset_traces()

    return total_r, env.steps, food


def evaluate(agent):
    env = SnakeEnv()
    rs, ls, fs = [], [], []

    for _ in range(EVAL_EPS):
        r, l, f = run_episode(agent, env, train=False)
        rs.append(r)
        ls.append(l)
        fs.append(f)

    return np.mean(rs), np.mean(ls), np.mean(fs)


def train_agent(age, label, seed_offset=0):
    np.random.seed(SEED + seed_offset)

    env = SnakeEnv()
    
    # Updated to 10 inputs to match your latent_dim=10
    agent = AgeingBipartiteGraph(14, 3, age=age)

    results = {
        "label": label,
        "age": age,
        "eval_x": [],
        "eval_rewards": [],
        "eval_lengths": [],
        "eval_foods": [],
        "ep_rewards": [],
        "agent": agent
    }

    for ep in range(1, N_EPISODES + 1):
        r, _, _ = run_episode(agent, env, train=True)
        results["ep_rewards"].append(r)

        if ep % EVAL_EVERY == 0:
            er, el, ef = evaluate(agent)

            results["eval_x"].append(ep)
            results["eval_rewards"].append(er)
            results["eval_lengths"].append(el)
            results["eval_foods"].append(ef)

            print(f"{label} | ep {ep} | reward={er:.2f} food={ef:.2f}")

    return results


def main():
    configs = [
        (2, "Young", 0),
        (5, "Middle", 1),
        (8, "Old", 2),
    ]

    results = [train_agent(*cfg) for cfg in configs]

    # Pass the new dynamic feature names to the plotter
    plot_results(results, LATENT_FEATURE_NAMES)

    # Watch the brains play (using a lambda to encode the grid for the visualizer)
    best_agent = results[1]["agent"]
    original_predict = best_agent.predict
    best_agent.predict = lambda raw_grid: original_predict(encode_state(raw_grid, env.direction))
    
    env = SnakeEnv()
    watch_agent(best_agent, env, cell_size=100)


if __name__ == "__main__":
    main()
