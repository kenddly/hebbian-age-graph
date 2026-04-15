from benchmarks.benchmark import run_episode, evaluate
from environments.env import SnakeEnv
from models.ageing_graph import AgeingGraph
import matplotlib.pyplot as plt

def benchmark_ageing(graph_class, env, num_episodes=1000):
    agent = graph_class(8, 3, age=0.0, ageing_threshold=200.0)
    results = {
        "age": [],
        "eval_rewards": [],
        "eval_lengths": [],
        "eval_foods": []
    }

    for episode in range(num_episodes):
        run_episode(agent, env)

        if episode % 50 == 0:
            r, l, f = evaluate(agent)
            results["age"].append(agent.age)
            results["eval_rewards"].append(r)
            results["eval_lengths"].append(l)
            results["eval_foods"].append(f)

    return results

def plot_results(results):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(results["age"], results["eval_rewards"], label="Reward")
    plt.xlabel("Age")
    plt.ylabel("Reward")
    plt.title("Reward vs Age")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(results["age"], results["eval_lengths"], label="Length", color='orange')
    plt.xlabel("Age")
    plt.ylabel("Length")
    plt.title("Length vs Age")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(results["age"], results["eval_foods"], label="Food Eaten", color='green')
    plt.xlabel("Age")
    plt.ylabel("Food Eaten")
    plt.title("Food Eaten vs Age")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = SnakeEnv()
    env.seed(42)

    ageing_results = benchmark_ageing(graph_class=AgeingGraph, env=env)
    plot_results(ageing_results)
    print(ageing_results)

