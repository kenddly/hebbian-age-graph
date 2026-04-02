import numpy as np

from reversal_env import ReversalEnv, MAX_STEPS, FEATURE_NAMES
from graph import AgeingBipartiteGraph
from plot import plot_results
from snake_visualizer import watch_agent


SEED = 42
N_EPISODES = 4000
EVAL_EVERY = 50
EVAL_EPS = 20


def run_episode(agent, env, train=True):
    state = env.reset()
    total_r, food = 0.0, 0

    for _ in range(MAX_STEPS):
        action = agent.forward(state) if train else agent.predict(state)
        state, reward, done = env.step(action)

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
    env = ReversalEnv()
    rs, ls, fs = [], [], []

    for _ in range(EVAL_EPS):
        r, l, f = run_episode(agent, env, train=False)
        rs.append(r)
        ls.append(l)
        fs.append(f)

    return np.mean(rs), np.mean(ls), np.mean(fs)


def train_agent(age, label, seed_offset=0):
    np.random.seed(SEED + seed_offset)

    env = ReversalEnv()
    agent = AgeingBipartiteGraph(8, 3, age=age)

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
        # (2, "Young", 0),
        (5, "Middle", 1),
        # (8, "Old", 2),
    ]

    results = [train_agent(*cfg) for cfg in configs]

    plot_results(results, FEATURE_NAMES)

    # watch the brains play
    env = ReversalEnv()
    watch_agent(results[0]["agent"], env, cell_size=100)


if __name__ == "__main__":
    main()
