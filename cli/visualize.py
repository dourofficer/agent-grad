"""
python -m cli.visualize --file outputs/gpt-oss-20b_sweep-by-length.tsv --metric step_acc --by_length
python -m cli.visualize --file outputs/gpt-oss-20b_sweep-by-length.tsv --metric step_acc --by_length --save outputs/visualization/step_acc_by_length.png
python -m cli.visualize --file outputs/gpt-oss-20b_sweep.tsv --metric step_acc
python -m cli.visualize --file outputs/gpt-oss-20b_sweep.tsv --metric agent_acc --save outputs/visualization/agent_acc.png

python -m cli.visualize \
    --file outputs/gpt-oss-20b_sweep-by-length.tsv \
    --metric step_acc \
    --by_length \
    --save outputs/visualization/step_acc_by_length.png
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def sort_length_labels(labels):
    """Sort trajectory length bin labels by leading integer."""
    return sorted(labels, key=lambda x: int(x.split('-')[0].replace('+', '')))


def plot_by_length(df, metric, save_path):
    """
    One figure per k value.
    x-axis: trajectory_length (excluding 'all'), lines: (model, strategy, subset).
    """
    df = df[df["trajectory_length"] != "all"].copy()
    k_values = sorted(df["k"].unique())

    figs = []
    for k in k_values:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = df[df["k"] == k]

        all_lengths = sort_length_labels(sub["trajectory_length"].unique().tolist())
        length_order = {l: i for i, l in enumerate(all_lengths)}

        configs = sub[["model", "strategy", "subset"]].drop_duplicates().values.tolist()
        for model, strategy, subset in configs:
            mask   = (sub["model"] == model) & (sub["strategy"] == strategy) & (sub["subset"] == subset)
            group  = sub[mask].copy()
            group  = group.sort_values("trajectory_length", key=lambda s: s.map(length_order))
            label  = f"{strategy}/{subset}" if model == sub["model"].iloc[0] else f"{model}/{strategy}/{subset}"
            ax.plot(group["trajectory_length"], group[metric], marker='o', label=label)

        ax.set_xlabel("Trajectory Length")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by Trajectory Length  (k={k})")
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.5)
        fig.tight_layout()
        figs.append((k, fig))

    return figs


def plot_by_k(df, metric, save_path):
    """
    Single figure.
    x-axis: k, lines: (model, strategy, subset).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    k_values = sorted(df["k"].unique())

    configs = df[["model", "strategy", "subset"]].drop_duplicates().values.tolist()
    for model, strategy, subset in configs:
        mask  = (df["model"] == model) & (df["strategy"] == strategy) & (df["subset"] == subset)
        group = df[mask].sort_values("k")
        label = f"{strategy}/{subset}" if len(df["model"].unique()) == 1 else f"{model}/{strategy}/{subset}"
        ax.plot(group["k"], group[metric], marker='o', label=label)

    ax.set_xticks(k_values)
    ax.set_xlabel("k")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} vs k")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.tight_layout()

    return [(None, fig)]


def save_figures(figs, save_path, by_length):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if len(figs) == 1:
        figs[0][1].savefig(save_path, dpi=150)
        print(f"Saved to {save_path}")
    else:
        for k, fig in figs:
            stem = save_path.stem
            out  = save_path.with_name(f"{stem}_k{k}{save_path.suffix}")
            fig.savefig(out, dpi=150)
            print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file',      required=True, help='TSV file from cli.evaluate --sweep')
    parser.add_argument('--metric',    default='step_acc', choices=['step_acc', 'agent_acc'])
    parser.add_argument('--by_length', action='store_true', help='Plot along trajectory length axis')
    parser.add_argument('--save',      help='Output image path (default: outputs/visualization/<metric>[_by_length].png)')
    args = parser.parse_args()

    if args.save is None:
        suffix    = "_by_length" if args.by_length else ""
        args.save = f"outputs/visualization/{args.metric}{suffix}.png"

    df = pd.read_csv(args.file, sep='\t')

    figs = plot_by_length(df, args.metric, args.save) if args.by_length \
        else plot_by_k(df, args.metric, args.save)

    save_figures(figs, args.save, args.by_length)


if __name__ == '__main__':
    main()