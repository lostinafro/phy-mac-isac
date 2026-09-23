import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from resource_allocation import available_methods
import scipy.stats as stats
import parameters as p
from simulate_vs_ue import num_ue_vec
from simulate_vs_regions import num_regions_vec

def latex_look(pl):
    # Set LaTeX as the default text interpreter
    pl.rcParams['text.usetex'] = True
    pl.rcParams['font.family'] = 'serif'
    pl.rcParams['font.serif'] = ['Computer Modern Roman']  # LaTeX default font

def load_episodes(datapath: str, ra_label: str, cycle: range, fields: list[str]) -> dict:
    """Reconstruct a per-ue summary pandas from an already-saved .npz."""
    results = {}
    for field in fields:
        results_field = []
        for ep in cycle:
            filepath = os.path.join(datapath, f"{ra_label}_ep{ep:04d}.npz")
            # Reading data
            try:
                with np.load(filepath) as data:
                    # Print warning if the field is not found
                    if field not in data:
                        print(f"Field '{field}' not found in {filepath}. Available keys: {list(data.keys())}")
                    results_field.append(data[field])
            except FileNotFoundError:
                print(f"File '{filepath}' not found.")
        results[field] = np.stack(results_field, axis=0)
    return results

def output_results(num_vec: list | np.ndarray, datapath: str,
                   method_labels: list[str], results_labels: list[str], cdf_labels: list[str],
                   suffix: str = "",
                   ):
    # Prepare the lists
    overall_dict = []
    method_cdfs = []

    for n in num_vec:
        n_dict = {}
        n_method_cdfs = {}
        for method in method_labels:
            # Data folder to read the data from
            data_folder = os.path.join(datapath, method + suffix + f"{n:03d}")
            # Check for algorithm
            try:
                n_dict[method] = load_episodes(data_folder, method, range(p.NUM_EP), results_labels)
                field_cdfs = {}
                for field_label in cdf_labels:
                    if field_label in ["risk_predicted"]:
                        sample = n_dict[method][field_label][:, :-1].sum(axis=-1).flatten()
                        sample = sample[~np.isnan(sample)]
                        field_cdfs[field_label] = stats.ecdf(sample)
                    elif field_label in ["misclassification"]:
                        sample = n_dict[method][field_label][:, :-1].mean(axis=0).flatten()
                        sample = sample[~np.isnan(sample)]
                        field_cdfs[field_label] = stats.ecdf(sample)
                    else:
                        sample = n_dict[method][field_label][:, :-1].flatten()
                        sample = sample[~np.isnan(sample)]
                        field_cdfs[field_label] = stats.ecdf(sample)
                n_method_cdfs[method] = field_cdfs
            except Exception as e:
                print(f"Error loading episodes for {method} with {n} UEs: {e}")
        overall_dict.append(n_dict)
        method_cdfs.append(n_method_cdfs)
    return overall_dict, method_cdfs


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
                    cdf = cdfs[n][method][var_label].cdf
                    # Always having max 10 markers
                    step = max(1, len(cdf.quantiles) // 10)
                    mark_every = (phase * step // len(methods), step)
                    cdf.plot(ax=ax[i, n], label=f"{method}",
                             color=method_colors[method],
                             marker=method_markers[method],
                             markevery=mark_every,
                             **plot_shared_kwargs)
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


def print_avg_perc(tests_values: list | np.ndarray,
                   res_dict: dict,
                   res_cdfs: dict,
                   perc: float | list | np.ndarray):

    if isinstance(perc, float):
        perc = [perc]

    for metric in metrics:
        print(f"{metric}".upper())
        for method in ra_labels:
            try:
                # Take average
                avg = [f"{np.nanmean(res_dict[u][method][metric][:, :-1]):.3f}" for u, _ in enumerate(tests_values)]
                print(f"\tAvg. for {method}: {' '.join(avg)}")

                # Test perc
                for p in perc:
                    idx = [np.where(res_cdfs[u][method][metric].cdf.probabilities > p)[0][0] for u, _ in enumerate(tests_values)]
                    prob = [f"{res_cdfs[u][method][metric].cdf.quantiles[idx[u]]:.3f}" for u, _ in enumerate(tests_values)]
                    print(f"\t{p * 100:.1f}% for {method}: {' '.join(prob)}")
            except Exception as e:
                print(f"Error computing statistics for {method} and {metric}: {e}")

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

    ## TEST VS_UE
    # Generate dictionaries
    ue_dict, ue_cdfs = output_results(num_ue_vec, data_path, ra_labels, results_metrics, metrics, suffix="_vs_ue_U")
    reg_dict, reg_cdfs = output_results(num_regions_vec, data_path, ra_labels, results_metrics, metrics, suffix="_vs_regions_R")

    # Save metrics
    # Generate mean and 95% confidence intervals
    # for metric in metrics:
    #     df = pd.DataFrame()
    #     for name in ra_labels:
    #         avg = []
    #         qua = []
    #         lo = []
    #         hi = []
    #         for u, ue in enumerate(num_ue_vec):
    #             data = ue_dict[u][name][metric][:,:-1].mean(0)
    #             avg.append(data.mean())
    #             qua.append(np.interp(0.95, ue_cdfs[u][name][metric].cdf.probabilities, ue_cdfs[u][name][metric].cdf.quantiles))
    #             lo.append(np.percentile(data, 2.5))
    #             hi.append(np.percentile(data, 97.5))
    #         df[f"{name}_avg"] = avg
    #         df[f"{name}_95"] = qua
    #         df[f"{name}_lo"] = lo
    #         df[f"{name}_hi"] = hi
    #     df.insert(0, "U", np.arange(len(num_ue_vec)))
    #     df.to_csv(os.path.join(data_path, f"{metric}_vs_ue.csv"), index=False)

    for metric in metrics:
        df = pd.DataFrame()
        for name in ra_labels:
            avg = []
            qua = []
            lo = []
            hi = []
            for r, reg in enumerate(num_regions_vec):
                data = reg_dict[r][name][metric][:,:-1].mean(0)
                avg.append(data.mean())
                qua.append(np.interp(0.95, reg_cdfs[r][name][metric].cdf.probabilities, reg_cdfs[r][name][metric].cdf.quantiles))
                lo.append(np.percentile(data, 2.5))
                hi.append(np.percentile(data, 97.5))
            df[f"{name}_avg"] = avg
            df[f"{name}_95"] = qua
            df[f"{name}_lo"] = lo
            df[f"{name}_hi"] = hi
        df.insert(0, "R", np.arange(len(num_regions_vec)))
        df.to_csv(os.path.join(data_path, f"{metric}_vs_regions.csv"), index=False)

    # Plot CDFs
    shared_args = dict(linestyle="dashed", linewidth=1, markersize=3)
    fig_ue, ax_ue = plot_cdfs(num_ue_vec, ue_cdfs, metrics, ra_labels, colors, "UEs", method_markers=markers,
                              **shared_args)
    filename = os.path.join(data_path, "cdf_vs_ue")
    fig_ue.savefig(filename + '.jpg', dpi=300, bbox_inches='tight')

    fig_reg, ax_reg = plot_cdfs(num_regions_vec, reg_cdfs, metrics, ra_labels, colors, "R", method_markers=markers,
                                **shared_args)
    filename = os.path.join(data_path, "cdf_vs_reg")
    fig_reg.savefig(filename + '.jpg', dpi=300, bbox_inches='tight')

    print_avg_perc(num_regions_vec, reg_dict, reg_cdfs, perc=[0.95])
