"""Plotting functionality."""

__all__ = ('plot_confusion_matrix',)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure

from .stats import cohens_kappa


def plot_confusion_matrix(
    categories,
    confusion_matrix,
    ax=None,
    description=None,
    xlabel='Predicted labels',
    ylabel='True labels',
    heatmap_cmap='Purples',
    label_fontsize=14,
    title_fontsize=14,
    annot_fontsize=12,
    include_pcts: bool = True,
) -> Figure | None:
    """Plot a confusion matrix on the specified axis.

    Args:
        confusion_matrix (np.array): N x N confusion matrix
        ax (matplotlib.Axes, optional): Axis object for plotting, will create one if None specified.
        description (str, optional): Optional string describing classifier. Defaults to ''.
        heatmap_cmap (str, optional): Confusion matrix color map. Defaults to 'Greys'.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    else:
        fig = None
    group_counts = ['{0:0.0f}'.format(value) for value in confusion_matrix.flatten()]
    if include_pcts:
        group_percentages = np.asarray(
            [['{0:.2%}'.format(i / np.sum(row) if np.sum(row) != 0 else 0) for i in row] for row in confusion_matrix]
        ).flatten()
        box_labels = [f'{v2}\n({v3})'.strip() for v2, v3 in zip(group_counts, group_percentages)]
    else:
        box_labels = group_counts
    box_labels = np.reshape(box_labels, (len(confusion_matrix), len(confusion_matrix)))
    sns.set_theme()
    sns.heatmap(
        confusion_matrix,
        fmt='',
        annot=box_labels,
        cmap=heatmap_cmap,
        ax=ax,
        xticklabels=categories,
        yticklabels=categories,
        cbar=False,
        annot_kws={'size': annot_fontsize},
    )
    accuracy = np.trace(confusion_matrix) / float(np.sum(confusion_matrix))
    kappa = cohens_kappa(confusion_matrix, n_classes=len(confusion_matrix))
    plt.sca(ax)
    # Add precision/recall ticks
    ax2x = ax.twinx()
    ax2y = ax.twiny()
    ax2x.set_zorder(ax.get_zorder() - 1)
    ax2y.set_zorder(ax.get_zorder() - 1)
    precisions, recalls = compute_pr(confusion_matrix)
    ax2y.set_xticks(0.5 + np.arange(confusion_matrix.shape[0]))
    ax2x.set_yticks(0.5 + np.arange(confusion_matrix.shape[0]))  # ax.get_yticks())
    ax2y.set_xticklabels(precisions, fontsize=annot_fontsize)
    ax2x.set_yticklabels(reversed(recalls), fontsize=annot_fontsize)
    for l in ax.get_xticklabels() + ax.get_yticklabels():
        l.set_fontsize(annot_fontsize)
    ax2x.set_ylim(0, confusion_matrix.shape[0])
    ax2y.set_xlim(0, confusion_matrix.shape[0])
    ax2y.set_xlabel('Precision', fontsize=annot_fontsize)
    ax2x.set_ylabel('Recall', fontsize=annot_fontsize)
    ## Acc
    acc_str = f'Acc={100 * accuracy:0.1f}. $\kappa$={kappa:.3f}'
    if description is None:
        title_str = acc_str
    else:
        title_str = f'{description} | {acc_str}'
    ax.set_title(title_str, fontsize=title_fontsize)
    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    return fig


def compute_pr(cmat):
    precisions, recalls = [], []
    for i, _ in enumerate(cmat):
        if cmat[:, i].sum() == 0:
            p, r = 0, 0
        else:
            p = 100 * cmat[i, i] / cmat[:, i].sum()
            r = 100 * cmat[i, i] / cmat[i].sum()
        precisions.append(f'{p:.1f}%')
        recalls.append(f'{r:.1f}%')
    return precisions, recalls
