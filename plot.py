import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def smooth(x, w):
    if len(x) < w:
        return np.array(x)
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_results(results, feature_names, smooth_w=30, out_path="snake_benchmark.png"):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)

    # color map by age (NOT label string anymore)
    ages = [r["age"] for r in results]
    cmap = plt.cm.viridis
    colors = {r["label"]: cmap(a / max(ages)) for r, a in zip(results, ages)}

    # ── training reward
    ax = axes[0]
    for r in results:
        ax.plot(smooth(r["ep_rewards"], smooth_w), label=r["label"], color=colors[r["label"]])
    ax.set_title("Training reward (smoothed)")
    ax.legend()

    # ── eval reward
    ax = axes[1]
    for r in results:
        ax.plot(r["eval_x"], r["eval_rewards"], marker='o', label=r["label"], color=colors[r["label"]])
    ax.set_title("Evaluation reward")
    ax.legend()

    # ── food
    ax = axes[2]
    for r in results:
        ax.plot(r["eval_x"], r["eval_foods"], label=r["label"], color=colors[r["label"]])
    ax.set_title("Food eaten")
    ax.legend()

    # ── length
    ax = axes[3]
    for r in results:
        ax.plot(r["eval_x"], r["eval_lengths"], label=r["label"], color=colors[r["label"]])
    ax.set_title("Episode length")
    ax.legend()

    # ── weight heatmaps (first + last agent generically)
    for i, r in enumerate([results[0], results[-1]]):
        ax = axes[4 + i]
        im = ax.imshow(r["agent"].weights, aspect="auto")

        ax.set_title(f"{r['label']} weights")
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=8)

        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"Saved → {out_path}")
