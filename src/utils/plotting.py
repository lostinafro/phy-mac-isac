# Plotting: matplotlib helpers, color utilities, node visual style
import os
import colorsys
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import rc

# The following are defined for graphic purpose only
node_color = {'BS': '#DC2516',  'UE': '#36F507', 'RIS': '#0F4EEA'}
node_mark = {'BS': 'o', 'UE': 'x', 'RIS': '^'}


def shade_of(base_color: str, rng: np.random.Generator = np.random.default_rng(),
             hue_jitter: float = 0.08, sat_jitter: float = 0.15, val_jitter: float = 0.15):
    """Return an RGB tuple that is a random variation around `base_color`,
    by perturbing its hue/saturation/value slightly and clamping to [0, 1]."""
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    h = (h + rng.uniform(-hue_jitter, hue_jitter)) % 1.0
    s = np.clip(s + rng.uniform(-sat_jitter, sat_jitter), 0.0, 1.0)
    l = np.clip(l + rng.uniform(-val_jitter, val_jitter), 0.1, 0.9)  # avoid near-black/near-white

    return colorsys.hls_to_rgb(h, l, s)


def printplot(fig: plt.Figure = None,
              ax: plt.Axes or np.array = None,
              render: bool = False,
              filename: str = '',
              dirname: str = '',
              title: str or list = None,
              labels: list = None,
              grid: bool = True,
              orientation: str = 'vertical',
              tight_layout: bool = True):
    # Common print options with LaTeX type definitions
    rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)

    if fig is None:
        fig = plt.gcf()
        ax = plt.gca()
    if isinstance(ax, plt.Axes):    # Single axis plot
        if grid:
            ax.grid(axis='both', color='#E9E9E9', linestyle='--', linewidth = 0.8)
        try:
            ax.set_xlabel(labels[0])
            ax.set_ylabel(labels[1])
            ax.set_ylabel(labels[2])
        except (TypeError, IndexError):
            pass
        if ax.get_legend_handles_labels()[1]:   # Is there is at least a label, print the legend
            ax.legend()
        if not render:
            ax.set_title(title)
            fig.show()
            plt.close(fig=fig)
        else:
            filename = os.path.join(dirname, filename)
            ax.set_title(title)
            fig.savefig(filename + '.jpg', dpi=300)
            ax.set_title('')
            plt.close(fig=fig)
            return
    else:   # Multi-axes plot
        if ax[0].get_legend_handles_labels()[1]:  # Is there is at least a label, print the legend
            ax[0].legend()
        if grid:
            for a in ax:
                a.grid(axis='both', color='#E9E9E9', linestyle='--', linewidth = 0.8)
        try:
            if orientation == 'vertical':
                ax[-1].set_xlabel(labels[0])
                for i, a in enumerate(ax):
                    a.set_ylabel(labels[i+1])
            elif orientation == 'horizontal':
                ax[0].set_ylabel(labels[-1])
                for i, a in enumerate(ax):
                    a.set_xlabel(labels[i])
            else:   # No particular orientation
                i = 0
                for a in ax:
                    a.set_xlabel(labels[i])
                    a.set_ylabel(labels[i + 1])
                    i += 2
        except (TypeError, IndexError):
            pass
        # Keep the title for more than one figure
        try:
            for i, a in enumerate(ax):
                a.set_title(title[i])
        except IndexError:
            ax[0].set_title(title)
        except TypeError:
            pass
        if not render:
            if tight_layout:
                fig.tight_layout()
            fig.show()
            plt.close(fig=fig)
        else:
            filename = os.path.join(dirname, filename)
            if tight_layout:
                fig.tight_layout()
            fig.savefig(filename + '.jpg', dpi=300, bbox_inches='tight')