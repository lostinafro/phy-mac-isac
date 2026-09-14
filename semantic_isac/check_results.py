from logging import exception
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import os
from resource_allocation import available_methods
import scipy.stats as stats
import pandas as pd
import utils as ut
import parameters as p

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
            data_folder = os.path.join(datapath, method + suffix + f"_U{n:03d}")
            # Check for algorithm
            try:
                n_dict[method] = load_episodes(data_folder, method, range(100), results_labels)
                field_cdfs = {}
                for field_label in cdf_labels:
                    if field_label in ["risk_predicted"]:
                        sample = n_dict[method][field_label][:, :-1].sum(axis=-1).flatten()
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


if __name__ == "__main__":
    # Setting the results to read
    results_labels = ["avg_datasize", "area_coordinates", "ue_slot_map", "large_scale", "large_scale_hat", "queue", "delta_queue",
                      "se_achievable", "se_set", "sinr", "efim_oracle", "efim_predicted",
                      "misclassification", "risk_hat"]
    # Setting the results for cdf computation
    cdf_labels = ["queue", "se_achievable", "se_set", "misclassification", "risk_hat"]
    colors =  dict(zip(available_methods.keys(), [ "#ff0000", "#ffd700", "#ffb14e", "#fa8775", "#cd34b5", "#0000ff" ]))


    # Other parameters
    num_ue_vec = np.array([12, 18, 27, 36, 54])
    ra_labels = list(available_methods.keys())
    data_path = os.path.join(p.DATA_FOLDER, "2026-09-14", "se_test")

    big_dict, ra_cdfs = output_results(num_ue_vec, data_path, ra_labels, results_labels, cdf_labels, suffix="_vs_ue")

    # Plot CDFs
    latex_look(plt)
    cm = 1 / 2.54
    fig, ax = plt.subplots(nrows=len(cdf_labels), ncols=len(num_ue_vec), figsize=(42*cm, 29.7*cm))
    for u, ue in enumerate(num_ue_vec):
        ax[0, u].set_title(f"{ue} UEs")
        for i, field_label in enumerate(cdf_labels):
            for method in ra_labels:
                try:
                    curve = ra_cdfs[u][method][field_label].cdf.plot(ax=ax[i, u])[0]
                    curve.set_label(f"{method}")
                    curve.set_color(colors[method])
                    curve.set_linestyle("dashed")
                except Exception as e:
                    print(f"Error computing CDF for {method} and {field_label}: {e}")
            if ax[i, u].get_legend_handles_labels()[1]:  # If there is at least a label, print the legend
               ax[i, u].legend()
            ax[i, u].grid(True)
            ax[i, u].set_xlabel(field_label)
            ax[i, u].set_ylabel("CDF")
    fig.tight_layout()
    # fig.show()
    # Save figure
    filename = os.path.join(data_path, "cdf_vs_ue")
    fig.savefig(filename + '.jpg', dpi=300, bbox_inches='tight')

    for field_label in cdf_labels:
        print(f"{field_label}".upper())
        for method in ra_labels:
            avg = [float(np.nanmean(big_dict[u][method][field_label][:, :-1])) for u, _ in enumerate(num_ue_vec)]
            prob_95 = [float(ra_cdfs[u][method][field_label].cdf.quantiles[np.where(ra_cdfs[u][method][field_label].cdf.probabilities > 0.95)[0][0]]) for u, _ in enumerate(num_ue_vec)]
            print(f"\tAvg. for {method}: {avg}")
            print(f"\t95% for {method}: {prob_95}")


    a = 42

