from load_results import load_all_results
from summary_stats import summarize_mean_std
from wilcoxon_test import wilcoxon_per_experiment, significant_ratio

import pandas as pd
from pathlib import Path

RESULTS_DIR = "outputs/synthetic/results"
OUT_DIR = "analysis/synthetic/results"

METRICS = ["kl", "hd", "jsd", "acc", "ace", "spa", "mse"]

def main():
    # --- 出力ディレクトリ作成 ---
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 結果読み込み ---
    df = load_all_results(RESULTS_DIR)

    # --- Mean / Std ---
    summary = summarize_mean_std(df, METRICS)
    summary.to_csv(out_dir / "summary_mean_std.csv", index=False)

    # --- Wilcoxon ---
    all_ratios = []

    for metric in METRICS:
        w = wilcoxon_per_experiment(df, metric)
        w.to_csv(out_dir / f"wilcoxon_{metric}.csv", index=False)

        ratio = significant_ratio(w)
        ratio["metric"] = metric
        all_ratios.append(ratio)

    ratio_df = pd.concat(all_ratios, ignore_index=True)
    ratio_df.to_csv(out_dir / "significant_ratio.csv", index=False)

if __name__ == "__main__":
    main()
