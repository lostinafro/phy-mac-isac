import os
import re
import gc
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from resource_allocation import available_methods
import parameters as p
from simulate_vs_ue import num_ue_vec
from simulate_vs_regions import num_regions_vec

def latex_look(pl):
    # Set LaTeX as the default text interpreter
    pl.rcParams['text.usetex'] = True
    pl.rcParams['font.family'] = 'serif'
    pl.rcParams['font.serif'] = ['Computer Modern Roman']  # LaTeX default font

# Convention for `stats_list` entries:
#   "mean"                -> np.nanmean
#   "std"                 -> np.nanstd
#   "<p>-percentile"      -> np.nanpercentile(data, p), p in [0, 100]  e.g. "95-percentile"
#   "<q>-quantile"        -> np.nanquantile(data, q),   q in [0, 1]    e.g. "0.95-quantile"
_STAT_PATTERN = re.compile(r"^([\d.]+)-(percentile|quantile)$")


def _compute_stat(data: np.ndarray, stat: str) -> float:
    """Compute a single named summary statistic over a flattened sample array."""
    if stat == "mean":
        return np.nanmean(data)
    if stat == "std":
        return np.nanstd(data)
    m = _STAT_PATTERN.match(stat)
    if m:
        value, kind = float(m.group(1)), m.group(2)
        if kind == "percentile":
            cdf = stats.ecdf(data).cdf
            return np.interp(value, cdf.probabilities, cdf.quantiles)
        else:
            return np.nanquantile(data, value)
    raise ValueError(f"Unrecognized stat '{stat}'. Use 'mean', 'std', '<p>-percentile', or '<q>-quantile'.")


def _load_metric_flat(datapath: str, ra_label: str, num_episodes: int, field: str,
                      preprocess=None) -> np.ndarray:
    """Load one field across all episodes for one (num_test, method) combination, flatten
    it into a 1D sample array, then let the caller discard it. No accumulation across calls
    -- this is the piece that keeps memory bounded to one combination at a time."""
    values = []
    for ep in range(num_episodes):
        filepath = os.path.join(datapath, f"{ra_label}_ep{ep:04d}.npz")
        try:
            with np.load(filepath) as data:
                if field not in data:
                    print(f"Field '{field}' not found in {filepath}. Available keys: {list(data.keys())}")
                    continue
                arr = data[field][:, :-1]   # drop the initialization frame, per original convention
        except FileNotFoundError:
            print(f"File '{filepath}' not found.")
            continue

        if preprocess is not None:
            arr = preprocess(arr)
        values.append(arr.flatten())

    if not values:
        return np.array([])
    flat = np.concatenate(values)
    return flat[~np.isnan(flat)]


# Per-metric preprocessing, matching the special-casing in the original script.
_FIELD_PREPROCESSORS = {
    "risk_predicted": lambda arr: arr.sum(axis=-1),
    "misclassification": lambda arr: arr.mean(axis=0),
}


def compute_statistics(num_vec: list | np.ndarray, datapath: str,
                       method_labels: list[str], metric_labels: list[str],
                       stats_list: list[str], num_episodes: int,
                       suffix: str = "", n_quantile_points: int = 500) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute summary statistics for each metric, without ever holding more than one
    (num_test, method) combination's raw samples in memory at a time.

    :param num_vec: list of sweep values (e.g. num_ue_vec or num_regions_vec).
    :param datapath: str, base directory containing the per-combination subfolders.
    :param method_labels: list[str], resource-allocation method names.
    :param metric_labels: list[str], metric field names to summarize (aka "results_labels").
    :param stats_list: list[str], stat names per the convention above.
    :param num_episodes: int, number of episodes saved per combination.
    :param suffix: str, folder-naming suffix (e.g. "_vs_ue_U").

    :returns: dict[str, np.ndarray], one entry per metric, each of shape
        (len(num_vec), len(method_labels), len(stats_list)).
    """
    results = {field: np.full((len(num_vec), len(method_labels), len(stats_list)), np.nan)
               for field in metric_labels}
    cdfs = {field: np.empty((len(num_vec), len(method_labels)), dtype=object) for field in metric_labels}
    prob_grid = np.linspace(0., 1., n_quantile_points)

    for n_idx, n in enumerate(num_vec):
        for m_idx, method in enumerate(method_labels):
            data_folder = os.path.join(datapath, method + suffix + f"{n:03d}")
            for field in metric_labels:
                try:
                    flat = _load_metric_flat(data_folder, method, num_episodes, field,
                                             preprocess=_FIELD_PREPROCESSORS.get(field))
                    if flat.size == 0:
                        continue
                    for s_idx, stat in enumerate(stats_list):
                        results[field][n_idx, m_idx, s_idx] = _compute_stat(flat, stat)

                    ecdf_res = stats.ecdf(flat)
                    down_quantiles = np.interp(prob_grid, ecdf_res.cdf.probabilities, ecdf_res.cdf.quantiles)
                    cdfs[field][n_idx, m_idx] = {"quantiles": down_quantiles, "probabilities": prob_grid}
                except Exception as e:
                    print(f"Error computing stats for {method}, n={n}, field={field}: {e}")
                finally:
                    # Explicitly drop the (potentially large) flat sample array before the
                    # next field/method/n iteration -- this is what actually bounds memory.
                    flat = None
            gc.collect()

    return results, cdfs


def plot_cdfs(num_tests: list | np.ndarray,
              cdfs: type[stats.ecdf],
              metrics: list[str],
              methods: list[str],
              method_colors: dict[str, str],
              test_label: str,
              method_markers: dict[str, str] | None = None,
              **plot_shared_kwargs):
    """Plot eCDFs in a A3 size multifigures."""

    if method_markers is None:
        method_markers = dict(zip(methods, [''] * len(methods)))

    # Generate figure
    fig, ax = plt.subplots(nrows=len(metrics), ncols=len(num_tests), figsize=(42 * cm, 29.7 * cm))

    # Cycle through axes
    for n, value in enumerate(num_tests):
        # Each column uses the same parameter
        ax[0, n].set_title(test_label + f": {value}" )
        # Cycle through different variables
        for i, var_label in enumerate(metrics):
            # Cycle through different methods
            for phase, method in enumerate(methods):
                try:
                    cdf = cdfs[var_label][n, phase]
                    if cdf is not None:
                        # Always having max 10 markers
                        step = max(1, len(cdf["quantiles"]) // 10)
                        mark_every = (phase * step // len(methods), step)
                        ax[i, n].plot(cdf["quantiles"], cdf["probabilities"], label=method,
                                      color=method_colors[method], marker=method_markers[method],
                                      markevery=mark_every, **plot_shared_kwargs)
                except Exception as e:
                    print(f"Error plotting CDF for {method} and {var_label}: {e}")
            # If there is at least a label, print the legend
            if ax[i, n].get_legend_handles_labels()[1]:
               ax[i, n].legend()
            ax[i, n].grid(True)
            ax[i, n].set_xlabel(var_label)
            ax[i, n].set_ylabel("CDF")
    fig.tight_layout()
    return fig, ax


## GENERAL PARAMETERS
latex_look(plt)
cm = 1 / 2.54


if __name__ == "__main__":
    # Define colors and markers for each RA method
    ra_labels = list(available_methods.keys())
    colors = dict(zip(available_methods.keys(), ["#ff0000", "#ffd700", "#ffb14e", "#fa8775", "#cd34b5", "#0000ff"]))
    markers = dict(zip(available_methods.keys(), ["o", "s", "D", "p", "X", "v"]))

    # Setting the results to read
    results_metrics = ["queue_max", "queue",
                      "se_achievable", "se_set", "sinr",
                      "efim_oracle", "efim_predicted",
                      "misclassification", "risk_hat"]
    # Setting the results for cdf computation
    metrics = ["queue", "se_achievable", "se_set", "misclassification", "risk_hat"]

    # Other parameters
    data_path = os.path.join("/home/fabios/Nextcloud/data")

    stats_list = ["mean", "95-percentile"]

    # Shared plot args
    shared_plot_kwargs = dict(linestyle="dashed", linewidth=1, markersize=3)

    suffixes = ["_vs_ue_U", "_vs_regions_R"]
    for trial, name in zip((num_ue_vec, num_regions_vec), suffixes):
        var = name[-1]
        print(f"-- Name {name} - Variable {var} --")
        stat, cdfs = compute_statistics(trial, data_path, ra_labels, metrics, stats_list,
                                        num_episodes=p.NUM_EP, suffix=name)

        # print_avg_perc becomes a trivial lookup now -- no recomputation needed
        for metric in metrics:
            print(metric.upper())
            # Saving
            df = pd.DataFrame()
            # cycling through methods
            for m_idx, method in enumerate(ra_labels):
                row = stat[metric][:, m_idx, :]  # (len(trial), len(stats_list))
                print(f"\t{method}: " + " ".join(f"{stats_list[s]}={row[:, s]}" for s in range(len(stats_list))))

                df[f"{method}_avg"] = row[:, 0]
                df[f"{method}_95"] = row[:, 1]
            # Saving

            df.insert(0, var, np.arange(len(trial)))
            df.to_csv(os.path.join(data_path, f"__{metric}_vs_{var}.csv"), index=False)


        # Plot CDFs
        fig, ax = plot_cdfs(trial, cdfs, metrics, ra_labels, colors, "UEs", method_markers=markers,
                                  **shared_plot_kwargs)
        filename = os.path.join(data_path, f"__cdf_vs_{var}")
        fig.savefig(filename + '.jpg', dpi=300, bbox_inches='tight')
