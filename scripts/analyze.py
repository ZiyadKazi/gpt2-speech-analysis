"""
Script: analyze.py
===================
Purpose: Analyze the relationship between semantic surprisal and speech
         timing variables (word duration and pre-gap silence).

         Tests two hypotheses:
         1. Duration hypothesis:  higher surprisal → longer word duration
         2. Pre-gap hypothesis:   higher surprisal → longer silence before word

         Runs analysis for each speaker separately and combined.
         Produces scatter plots with regression lines and correlation statistics.

Usage:
    python scripts/analyze.py --richness <path1> [<path2> ...]


Input:  outputs/embeddings/<name>_layer<n>_richness.csv (one or more)
Output: outputs/results/correlation_results.csv
        outputs/figures/duration_vs_surprisal.png
        outputs/figures/pregap_vs_surprisal.png
        outputs/figures/combined_analysis.png
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze surprisal vs speech timing variables"
    )
    parser.add_argument(
        "--richness",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to richness CSV file(s) (space separated)"
    )
    parser.add_argument(
        "--output_dir_results",
        type=str,
        default="outputs/results",
        help="Directory to save correlation results (default: outputs/results)"
    )
    parser.add_argument(
        "--output_dir_figures",
        type=str,
        default="outputs/figures",
        help="Directory to save figures (default: outputs/figures)"
    )
    return parser.parse_args()


# Load and label data
def load_data(richness_paths):
    """
    Load one or more richness CSV files and add a speaker label to each.
    Combines all into one DataFrame.

    Args:
        richness_paths: list of Path objects

    Returns:
        df_all: combined DataFrame with 'speaker' column added
        dfs: dict of {speaker_name: DataFrame} for per-speaker analysis
    """
    dfs = {}
    all_frames = []

    for path in richness_paths:
        # Extract speaker name from filename
        # e.g. "brenebrown_5min_layer36_richness.csv" → "brenebrown"
        speaker = path.stem.split("_")[0]
        df = pd.read_csv(path)
        df["speaker"] = speaker
        dfs[speaker] = df
        all_frames.append(df)
        print(f"Loaded {len(df)} words for speaker: {speaker}")

    df_all = pd.concat(all_frames, ignore_index=True)
    print(f"Total words: {len(df_all)}")
    return df_all, dfs


# Filter outliers
def filter_outliers(df, columns, n_std=3):
    """
    Remove extreme outliers from specified columns.
    Keeps rows where all specified columns are within n_std
    standard deviations of their mean.

    Args:
        df: DataFrame
        columns: list of column names to check
        n_std: number of standard deviations (default: 3)

    Returns:
        df_filtered: DataFrame with outliers removed
        n_removed: number of rows removed
    """
    mask = pd.Series([True] * len(df), index=df.index)

    for col in columns:
        mean = df[col].mean()
        std  = df[col].std()
        mask = mask & (df[col] - mean).abs() <= n_std * std

    df_filtered = df[mask]
    n_removed   = len(df) - len(df_filtered)
    return df_filtered, n_removed


# Correlation analysis
def run_correlation(df, x_col, y_col, label):
    """
    Run Pearson correlation between two columns.
    Returns a dictionary of results.

    Args:
        df: DataFrame
        x_col: name of x variable column (surprisal)
        y_col: name of y variable column (duration or pre_gap)
        label: string label for this analysis

    Returns:
        result: dict with r, p, n, and interpretation
    """
    x = df[x_col].values
    y = df[y_col].values

    r, p = stats.pearsonr(x, y)
    n    = len(x)

    # Interpret significance
    if p < 0.001:
        sig = "***"
    elif p < 0.01:
        sig = "**"
    elif p < 0.05:
        sig = "*"
    else:
        sig = "ns"

    # Interpret direction
    if r > 0:
        direction = "positive"
    else:
        direction = "negative"

    result = {
        "label":     label,
        "x":         x_col,
        "y":         y_col,
        "r":         round(r, 4),
        "p":         round(p, 6),
        "n":         n,
        "sig":       sig,
        "direction": direction
    }

    print(f"  {label}: r={r:.4f}, p={p:.4f} {sig}, n={n}")
    return result

def run_partial_correlation(df, x_col, y_col, control_col, label):
    """
    Run partial correlation between x and y controlling for control_col.
    This removes the effect of control_col from both x and y,
    then correlates the residuals.

    Concretely:
        1. Regress x on control_col → get residuals of x
        2. Regress y on control_col → get residuals of y
        3. Correlate the two residual vectors

    This answers: after removing the effect of word_length,
    does surprisal still predict duration?

    Args:
        df: DataFrame
        x_col: predictor (surprisal)
        y_col: outcome (duration or pre_gap)
        control_col: confound to control for (word_length)
        label: string label for this analysis

    Returns:
        result: dict with r, p, n
    """
    x       = df[x_col].values.astype(float)
    y       = df[y_col].values.astype(float)
    control = df[control_col].values.astype(float)

    # Residuals of x after removing control
    _, _, _, _, _ = stats.linregress(control, x)
    slope_x, intercept_x, _, _, _ = stats.linregress(control, x)
    x_resid = x - (slope_x * control + intercept_x)

    # Residuals of y after removing control
    slope_y, intercept_y, _, _, _ = stats.linregress(control, y)
    y_resid = y - (slope_y * control + intercept_y)

    # Correlate residuals
    r, p = stats.pearsonr(x_resid, y_resid)
    n    = len(x)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    direction = "positive" if r > 0 else "negative"

    result = {
        "label":     label,
        "x":         x_col,
        "y":         y_col,
        "control":   control_col,
        "r":         round(r, 4),
        "p":         round(p, 6),
        "n":         n,
        "sig":       sig,
        "direction": direction
    }

    print(f"  {label}: r={r:.4f}, p={p:.4f} {sig}, n={n}")
    return result


# Plotting
def plot_surprisal_vs_variable(df_all, dfs, y_col, y_label, output_path):
    """
    Create a figure with one subplot per speaker plus a combined subplot.
    Each subplot shows a scatter plot of surprisal vs y_col with
    a regression line.

    Args:
        df_all: combined DataFrame
        dfs: dict of {speaker: DataFrame}
        y_col: name of y variable column
        y_label: human readable label for y axis
        output_path: where to save the figure
    """
    n_speakers = len(dfs)
    fig, axes  = plt.subplots(1, n_speakers + 1,
                               figsize=(6 * (n_speakers + 1), 5))

    # Color palette — one color per speaker
    colors = sns.color_palette("husl", n_speakers)

    # Per-speaker subplots
    for idx, (speaker, df) in enumerate(dfs.items()):
        ax    = axes[idx]
        color = colors[idx]

        # Scatter plot — use small alpha since many points overlap
        ax.scatter(df["richness"], df[y_col],
                   alpha=0.3, s=8, color=color, label=speaker)

        # Regression line
        x   = df["richness"].values
        y   = df[y_col].values
        m, b, r, p, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, m * x_line + b, color=color, linewidth=2)

        # Labels
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        ax.set_title(f"{speaker}\nr={r:.3f}, p={p:.4f} {sig}",
                     fontsize=11)
        ax.set_xlabel("Surprisal", fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        sns.despine(ax=ax)

    # Combined subplot
    ax = axes[-1]
    for idx, (speaker, df) in enumerate(dfs.items()):
        ax.scatter(df["richness"], df[y_col],
                   alpha=0.2, s=6, color=colors[idx], label=speaker)

    # Combined regression line
    x   = df_all["richness"].values
    y   = df_all[y_col].values
    m, b, r, p, _ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b, color="black", linewidth=2.5)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.set_title(f"Combined (both speakers)\nr={r:.3f}, p={p:.4f} {sig}",
                 fontsize=11)
    ax.set_xlabel("Surprisal", fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    plt.suptitle(f"Surprisal vs {y_label}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


# Main
def main():
    args = parse_args()

    richness_paths   = [Path(p) for p in args.richness]
    output_dir_res   = Path(args.output_dir_results)
    output_dir_fig   = Path(args.output_dir_figures)

    output_dir_res.mkdir(parents=True, exist_ok=True)
    output_dir_fig.mkdir(parents=True, exist_ok=True)

    # Checks
    for p in richness_paths:
        assert p.exists(), f"Richness file not found: {p}"

    # Step 1: Load data
    print("\nStep 1: Loading data...")
    df_all, dfs = load_data(richness_paths)

    # Step 2: Filter outliers
    print("\nStep 2: Filtering outliers (3 std devs)...")
    df_all, n_removed = filter_outliers(
        df_all, ["richness", "duration", "pre_gap"]
    )
    print(f"Removed {n_removed} outlier words")
    print(f"Words remaining: {len(df_all)}")

    # Filter per-speaker dfs too
    for speaker in dfs:
        dfs[speaker], _ = filter_outliers(
            dfs[speaker], ["richness", "duration", "pre_gap"]
        )

    # Step 3: Correlation analysis
    print("\nStep 3: Running correlation analysis...")
    results = []

    # Per speaker
    for speaker, df in dfs.items():
        print(f"\n  Speaker: {speaker}")
        results.append(run_correlation(
            df, "richness", "duration",
            f"{speaker}_surprisal_vs_duration"
        ))
        results.append(run_correlation(
            df, "richness", "pre_gap",
            f"{speaker}_surprisal_vs_pregap"
        ))

    # Combined
    print(f"\n  Combined:")
    results.append(run_correlation(
        df_all, "richness", "duration",
        "combined_surprisal_vs_duration"
    ))
    results.append(run_correlation(
        df_all, "richness", "pre_gap",
        "combined_surprisal_vs_pregap"
    ))

    # Save results
    results_df   = pd.DataFrame(results)
    results_path = output_dir_res / "correlation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    # ── Step 3b: Partial correlations controlling for word length ──────────────
    print("\nStep 3b: Partial correlations controlling for word length...")
    partial_results = []

    for speaker, df in dfs.items():
        print(f"\n  Speaker: {speaker}")
        partial_results.append(run_partial_correlation(
            df, "richness", "duration", "word_length",
            f"{speaker}_surprisal_vs_duration_controlling_wordlength"
        ))
        partial_results.append(run_partial_correlation(
            df, "richness", "pre_gap", "word_length",
            f"{speaker}_surprisal_vs_pregap_controlling_wordlength"
        ))

    print(f"\n  Combined:")
    partial_results.append(run_partial_correlation(
        df_all, "richness", "duration", "word_length",
        "combined_surprisal_vs_duration_controlling_wordlength"
    ))
    partial_results.append(run_partial_correlation(
        df_all, "richness", "pre_gap", "word_length",
        "combined_surprisal_vs_pregap_controlling_wordlength"
    ))

    # Save partial results
    partial_df   = pd.DataFrame(partial_results)
    partial_path = output_dir_res / "partial_correlation_results.csv"
    partial_df.to_csv(partial_path, index=False)
    print(f"\nSaved partial results: {partial_path}")

    # Step 4: Plots
    print("\nStep 4: Creating plots...")

    plot_surprisal_vs_variable(
        df_all, dfs,
        y_col="duration",
        y_label="Word Duration (seconds)",
        output_path=output_dir_fig / "duration_vs_surprisal.png"
    )

    plot_surprisal_vs_variable(
        df_all, dfs,
        y_col="pre_gap",
        y_label="Pre-word Silence (seconds)",
        output_path=output_dir_fig / "pregap_vs_surprisal.png"
    )

    # Step 5: Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for _, row in results_df.iterrows():
        print(f"\n{row['label']}")
        print(f"  r = {row['r']:+.4f}  ({row['direction']})")
        print(f"  p = {row['p']:.6f} {row['sig']}")
        print(f"  n = {row['n']} words")

    print("\n" + "="*60)
    print("PARTIAL CORRELATION RESULTS (controlling for word length)")
    print("="*60)
    for _, row in partial_df.iterrows():
        print(f"\n{row['label']}")
        print(f"  r = {row['r']:+.4f}  ({row['direction']})")
        print(f"  p = {row['p']:.6f} {row['sig']}")
        print(f"  n = {row['n']} words")
        
    print("\nDone!")


if __name__ == "__main__":
    main()