import seaborn as sns
import numpy as np
import itertools
import matplotlib.pyplot as plt
from pathlib import Path

def plot_cluster(ds_values, extreme_indices, bins, f_values, v_values, save_path, clusters):
    plt.clf()
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(Path(save_path).stem, fontsize=20)

    # deviation scores histogram
    ax1 = fig.add_subplot(121)
    sns.distplot(ds_values, bins='auto', label="density", hist=True, kde=False, norm_hist=True, ax=ax1)
    ax1.set_ylim([0, None])

    extreme_max_indices, extreme_min_indices = extreme_indices
    kwargs = {
        "alpha": 0.5,
        "linewidth": 0.8,
    }
    for idx in extreme_max_indices:
        ax1.axvline(bins[idx], color="red", linestyle="-", label="relmax", **kwargs)
    for idx in extreme_min_indices:
        ax1.axvline(bins[idx], color="green", linestyle="--", label="relmin", **kwargs)

    ax1.set_xlim([-0.9, 1.2])
    ax1.set_xlabel('deviation score')
    ax1.set_ylabel('pdf')
    by_label1 = dict(zip(*reversed(ax1.get_legend_handles_labels())))
    by_label2 = {}
    ax1.legend(
        list(by_label1.values()) + list(by_label2.values()),
        list(by_label1.keys()) + list(by_label2.keys()),
        loc="upper left",
    )

    # cluster scatter
    ax2 = fig.add_subplot(122)
    ax2.set_ylabel("predict_error")
    ax2.set_ylim((-100, 100))
    ax2.set_xlabel("real_measure")
    ax2.set_xlim((0, 1000))
    kwargs = dict(marker='.', alpha=0.5)

    for idx, cluster in enumerate(clusters):
        x = v_values[cluster]
        y = f_values[cluster] - x
        ax2.scatter(x=x, y=y, label=f"cluster_{idx}", **kwargs)
    remains = np.ones(v_values.shape, bool)
    remains[np.unique(list(itertools.chain.from_iterable(clusters)))] = False
    x = v_values[remains]
    y = f_values[remains] - x
    ax2.scatter(x=x, y=y, label=f"remain", **kwargs)
    ax2.legend()

    fig.savefig(save_path, bbox_inches="tight")
