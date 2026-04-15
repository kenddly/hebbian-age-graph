import numpy as np
from environments.env import SnakeEnv, MAX_STEPS, FEATURE_NAMES
from models.graph import BipartiteGraph
from plot import plot_results
from snake_visualizer import watch_agent

N_EPISODES = 2000
EVAL_EVERY = 50
EVAL_EPS = 20


def run_episode(agent, env, train=True):
    state = env.reset()
    total_r, food = 0.0, 0

    for _ in range(MAX_STEPS):
        action = agent.forward(state) if train else agent.predict(state)
        state, reward, done = env.step(action)

        if train:
            if reward < 0:
                reward *= 0.1
            agent.apply_reward(reward)

        total_r += reward
        if reward >= 10:
            food += 1

        if done:
            break

    return total_r, env.steps, food


def evaluate(agent, seed=None):
    env = SnakeEnv()
    env.seed(seed)

    rs, ls, fs = [], [], []

    for _ in range(EVAL_EPS):
        r, l, f = run_episode(agent, env, train=False)
        rs.append(r)
        ls.append(l)
        fs.append(f)

    return np.mean(rs), np.mean(ls), np.mean(fs)


def train_agent(age, label, seed):
    env = SnakeEnv()
    env.seed(seed)
    agent = BipartiteGraph(8, 3, age=age)

    results = {
        "label": label,
        "age": age,
        "eval_x": [],
        "eval_rewards": [],
        "eval_lengths": [],
        "eval_foods": [],
        "ep_rewards": [],
        "agent": agent,
        "best_weights": None
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
            if results["best_weights"] is None or er > max(results["eval_rewards"][:-1]):
                results["best_weights"] = agent.get_weights()

    return results


def main():
    configs = [
        (0, "Young", 0),
        (15, "Middle", 0),
        (30, "Old", 0),
    ]

    results = [train_agent(*cfg) for cfg in configs]

    plot_results(results, FEATURE_NAMES, out_path="outputs/plots/snake_benchmark.png")

    # watch the brains play
    result = results[0]
    env = SnakeEnv(max_steps=1000)
    env.seed(42)
    result["agent"].set_weights(result["best_weights"])
    watch_agent(result["agent"], env, cell_size=100)


if __name__ == "__main__":
    main()
