import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_results(results, feature_names, out_path="snake_benchmark.png"):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 4, figure=fig)

    # Row 0: eval reward (spans 2 cols), food (spans 2 cols)
    ax_eval   = fig.add_subplot(gs[0, 0:2])
    ax_food   = fig.add_subplot(gs[0, 2:4])
    # Row 1: episode length (spans 2 cols), weight heatmaps (1 col each)
    ax_length = fig.add_subplot(gs[1, 0:2])
    ax_w0     = fig.add_subplot(gs[1, 2])
    ax_w1     = fig.add_subplot(gs[1, 3])

    axes = [ax_eval, ax_food, ax_length, ax_w0, ax_w1]
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)

    ages = [r["age"] for r in results]
    cmap = plt.cm.viridis
    colors = {r["label"]: cmap(a / max(ages)) for r, a in zip(results, ages)}

    # ── eval reward
    for r in results:
        ax_eval.plot(r["eval_x"], r["eval_rewards"], marker='o',
                     label=r["label"], color=colors[r["label"]])
    ax_eval.set_title("Evaluation reward")
    ax_eval.set_xlabel("Episode")
    ax_eval.legend()

    # ── food
    for r in results:
        ax_food.plot(r["eval_x"], r["eval_foods"],
                     label=r["label"], color=colors[r["label"]])
    ax_food.set_title("Food eaten")
    ax_food.set_xlabel("Episode")
    ax_food.legend()

    # ── episode length
    for r in results:
        ax_length.plot(r["eval_x"], r["eval_lengths"],
                       label=r["label"], color=colors[r["label"]])
    ax_length.set_title("Episode length")
    ax_length.set_xlabel("Episode")
    ax_length.legend()

    # ── weight heatmaps
    for ax, r in zip([ax_w0, ax_w1], [results[0], results[-1]]):
        im = ax.imshow(r["agent"].get_weights(), aspect="auto")
        ax.set_title(f"{r['label']} weights")
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")
