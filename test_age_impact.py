import numpy as np
import matplotlib.pyplot as plt
from env import SnakeEnv
from graph import AgeingBipartiteGraph
from benchmark import run_episode, evaluate, N_EPISODES

def run_age_experiment(ages, seed=42):
    results = []
    
    for age in ages:
        np.random.seed(seed)
        env = SnakeEnv()
        # SnakeEnv doesn't seem to have a seed() method in our codebase directly natively used inside init,
        # but let's check its definition if possible. Let's just set np.random.seed for now.
        
        # Use tuned parameters from benchmark.py
        agent = AgeingBipartiteGraph(
            8, 3, age=age, 
        )
        
        print(f"Training agent with age {age:.2f}...")
        for ep in range(1, N_EPISODES + 1):
            run_episode(agent, env, train=True)
            
        print(f"Evaluating agent with age {age:.2f}...")
        # Evaluate to get final stable metrics
        er, el, ef = evaluate(agent)
        
        diags = agent.diagnostics()
        
        results.append({
            "age": age,
            "eval_reward": er,
            "eval_length": el,
            "eval_food": ef,
            "crystallized": diags["crystallized"],
            "reward_baseline": diags["reward_baseline"],
            "temperature": diags["temperature"],
            "lr_scale": agent.da_scale  # age_factor
        })
        
    return results

def plot_age_results(results, fp="age_impact.png"):
    ages = [r["age"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # 1. Performance (Reward)
    ax = axes[0]
    eval_rewards = [r["eval_reward"] for r in results]
    ax.plot(ages, eval_rewards, marker='o', color='tab:blue')
    ax.set_title("Final Evaluation Reward vs. Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    
    # 2. Food & Length
    ax = axes[1]
    eval_foods = [r["eval_food"] for r in results]
    eval_lengths = [r["eval_length"] for r in results]
    
    ax.plot(ages, eval_foods, marker='s', color='tab:green', label="Food Eaten")
    ax_twin = ax.twinx()
    ax_twin.plot(ages, eval_lengths, marker='^', color='tab:red', linestyle='--', label="Length")
    
    ax.set_title("Survival Stats vs. Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Mean Food Eaten")
    ax_twin.set_ylabel("Mean Length", color='tab:red')
    
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    
    # 3. Crystallized and Learning Rate Scale
    ax = axes[2]
    cryst = [r["crystallized"] for r in results]
    lr_scale = [r["lr_scale"] for r in results]
    
    ax.plot(ages, cryst, marker='D', color='purple')
    ax.set_title("Plasticity vs. Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Crystallized Synapses")
    
    ax_twin2 = ax.twinx()
    ax_twin2.plot(ages, lr_scale, color='orange', linestyle=':')
    ax_twin2.set_ylabel("Effective LR Scale", color='orange')
    
    # 4. Temperature and Baseline
    ax = axes[3]
    temps = [r["temperature"] for r in results]
    baselines = [r["reward_baseline"] for r in results]
    
    ax.plot(ages, temps, marker='v', color='tab:cyan', label="Temperature")
    ax.set_title("Actor Stats vs. Age")
    ax.set_xlabel("Age")
    ax.set_ylabel("Temperature Softmax")
    
    ax_twin3 = ax.twinx()
    ax_twin3.plot(ages, baselines, marker='p', color='black', alpha=0.6, label="Reward Baseline")
    ax_twin3.set_ylabel("Advantage Baseline")
    
    plt.tight_layout()
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f"Saved plot to {fp}")

if __name__ == "__main__":
    ages_to_test = np.linspace(0, 60, 16)
    results = run_age_experiment(ages_to_test, seed=42)
    plot_age_results(results)
