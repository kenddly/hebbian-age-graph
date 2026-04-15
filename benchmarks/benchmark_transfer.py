import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

from environments.env import SnakeEnv, MAX_STEPS, FEATURE_NAMES
from models.graph import BipartiteGraph
from snake_visualizer import watch_agent

N_EPISODES = 3000
EVAL_EVERY = 50
EVAL_EPS = 20
SWITCH_EPISODE = 1000

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

    if train:
        agent.reset_traces()

    return total_r, env.steps, food


def evaluate(agent, seed=None):
    env = SnakeEnv()
    env.seed(seed)
    rs, ls, fs = [], [],[]

    for _ in range(EVAL_EPS):
        r, l, f = run_episode(agent, env, train=False)
        rs.append(r)
        ls.append(l)
        fs.append(f)

    return np.mean(rs), np.mean(ls), np.mean(fs)


def train_baseline_agent(age, label, seed):
    print(f"\n--- Training Baseline: {label} (Age {age}) ---")
    env = SnakeEnv()
    env.seed(seed)
    agent = BipartiteGraph(8, 3, age=age, seed=seed)

    results = {
        "label": label,
        "eval_x":[], "eval_rewards": [], "eval_lengths": [], "eval_foods":[], 
        "eval_crysts": [],
        "ep_rewards":[], "agent": agent, "best_weights": None
    }

    for ep in range(1, N_EPISODES + 1):
        r, _, _ = run_episode(agent, env, train=True)
        results["ep_rewards"].append(r)

        if ep % EVAL_EVERY == 0:
            er, el, ef = evaluate(agent)
            cryst = agent.diagnostics()["crystallized"]
            
            results["eval_x"].append(ep)
            results["eval_rewards"].append(er)
            results["eval_lengths"].append(el)
            results["eval_foods"].append(ef)
            results["eval_crysts"].append(cryst)

            print(f"  {label} | ep {ep:>4} | reward={er:>6.2f} food={ef:>4.2f} | cryst={cryst}")
            if results["best_weights"] is None or er > max(results["eval_rewards"][:-1], default=-float('inf')):
                results["best_weights"] = agent.get_weights().copy()

    return results


def train_transfer_agent(age_start, age_transfer, label, seed):
    print(f"\n--- Training Transfer: {label} (Age {age_start} -> {age_transfer} at Ep {SWITCH_EPISODE}) ---")
    env = SnakeEnv()
    env.seed(seed)
    
    agent = BipartiteGraph(8, 3, age=age_start, seed=seed)

    results = {
        "label": label,
        "eval_x":[], "eval_rewards": [], "eval_lengths": [], "eval_foods":[], 
        "eval_crysts": [],
        "ep_rewards":[], "agent": None, "best_weights": None
    }

    for ep in range(1, N_EPISODES + 1):
        if ep == SWITCH_EPISODE + 1:
            print(f"\n  >>> EPISODE {ep}: AGEING TRIGGERED! Transferring weights to Age {age_transfer} <<<")
            new_agent = BipartiteGraph(8, 3, age=age_transfer)
            
            new_agent.set_weights(agent.get_weights().copy())
            new_agent.crystallization = agent.crystallization.copy()
            new_agent.reward_baseline = agent.reward_baseline
            
            agent = new_agent

        r, _, _ = run_episode(agent, env, train=True)
        results["ep_rewards"].append(r)

        if ep % EVAL_EVERY == 0:
            er, el, ef = evaluate(agent)
            cryst = agent.diagnostics()["crystallized"]

            results["eval_x"].append(ep)
            results["eval_rewards"].append(er)
            results["eval_lengths"].append(el)
            results["eval_foods"].append(ef)
            results["eval_crysts"].append(cryst)

            print(f"  {label} | ep {ep:>4} | reward={er:>6.2f} food={ef:>4.2f} | cryst={cryst}")
            if results["best_weights"] is None or er > max(results["eval_rewards"][:-1], default=-float('inf')):
                results["best_weights"] = agent.get_weights().copy()

    results["agent"] = agent
    return results


def plot_dynamic_results(results, switch_ep=None, out_path="outputs/plots/transfer_benchmark.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    fig = plt.figure(figsize=(24, 20))
    fig.patch.set_facecolor('#F8F8F6')
    
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.25)
    
    ax_train_r = fig.add_subplot(gs[0, 0])
    ax_eval_r  = fig.add_subplot(gs[0, 1])
    ax_food    = fig.add_subplot(gs[0, 2])
    ax_len     = fig.add_subplot(gs[1, 0])
    ax_cryst   = fig.add_subplot(gs[1, 1])
    
    ax_heatmaps = [fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1]), fig.add_subplot(gs[2, 2])]

    line_axes =[ax_train_r, ax_eval_r, ax_food, ax_len, ax_cryst]
    for ax in line_axes:
        ax.set_facecolor('#F8F8F6')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    colors = plt.cm.Set1.colors 

    for idx, res in enumerate(results):
        color = colors[idx % len(colors)]
        label = res["label"]

        w = 30
        ep_r = res["ep_rewards"]
        s_ep_r = np.convolve(ep_r, np.ones(w)/w, mode='valid') if len(ep_r) >= w else ep_r

        ax_train_r.plot(s_ep_r, color=color, label=label, alpha=0.8)
        ax_eval_r.plot(res["eval_x"], res["eval_rewards"], color=color, label=label, lw=2)
        ax_food.plot(res["eval_x"], res["eval_foods"], color=color, label=label, lw=2)
        ax_len.plot(res["eval_x"], res["eval_lengths"], color=color, label=label, lw=2)
        ax_cryst.plot(res["eval_x"], res["eval_crysts"], color=color, label=label, lw=2)

        # Plot Heatmap for this result
        ax_hm = ax_heatmaps[idx]
        weights = res["agent"].get_weights()
        crystallization = res["agent"].crystallization
        
        im = ax_hm.imshow(weights, aspect='auto', cmap='YlGnBu')
        plt.colorbar(im, ax=ax_hm, shrink=0.8, pad=0.02)
        
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                if crystallization[i, j]:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                    ax_hm.add_patch(rect)

        ax_hm.set_title(f"Final Weights:\n{label}", fontsize=10, fontweight='bold')
        ax_hm.set_xticks([0, 1, 2])
        ax_hm.set_xticklabels(['Straight', 'Right', 'Left'], fontsize=9)
        ax_hm.set_yticks(range(len(FEATURE_NAMES)))
        ax_hm.set_yticklabels(FEATURE_NAMES, fontsize=9)

    titles =["Training Reward (Smoothed)", "Eval Reward", "Eval Food Eaten", "Eval Episode Length", "Crystallized Synapses (MIS)"]
    for ax, title in zip(line_axes, titles):
        ax.set_title(title, fontweight='bold')
        if switch_ep:
            ax.axvline(x=switch_ep, color='black', linestyle='--', alpha=0.5, label='Age Transfer')
        ax.legend(fontsize=9)

    fig.suptitle("Ageing Transfer Benchmark: Prior Knowledge Integration", fontsize=16, fontweight='bold', y=0.95)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved dynamic plot to {out_path}")
    plt.close()


def main(SEED):

    results =[
        train_baseline_agent(age=10, label="Middle (From Scratch)", seed=SEED),
        train_baseline_agent(age=20, label="Old (From Scratch)", seed=SEED),
        train_transfer_agent(age_start=10, age_transfer=20, label="Transfer: Middle -> Old", seed=SEED)
    ]

    plot_dynamic_results(results, switch_ep=SWITCH_EPISODE, out_path=f"outputs/plots/transfer_benchmark_{SEED}.png")

    best_agent = results[2]["agent"]
    test_env = SnakeEnv(max_steps=1000)
    test_env.seed(SEED)
    
    print("\nStarting Pygame visualization for the Transfer Agent...")
    best_agent.set_weights(results[2]["best_weights"])
    # watch_agent(best_agent, test_env, cell_size=100)

if __name__ == "__main__":
    for i in range(42, 62):
        main(i)
