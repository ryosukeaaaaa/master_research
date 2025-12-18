import pandas as pd
from scipy.stats import wilcoxon

# 指標の向き
HIGHER_IS_BETTER = {
    "acc": True,
    "spa": True,
    "ace": False,
    "mse": False,
    "kl": False,
    "hd": False,
    "jsd": False,
}

def wilcoxon_per_experiment(
    df,
    metric,
    proposed_method="proposed",
    alpha=0.05,
):
    """
    experiment ごとに Wilcoxon を実行
    """
    records = []

    for exp, df_exp in df.groupby("experiment"):
        df_p = df_exp[df_exp["method"] == proposed_method]

        for method in df_exp["method"].unique():
            if method == proposed_method:
                continue

            df_m = df_exp[df_exp["method"] == method]

            # seed で揃える（対応あり）
            merged = df_p.merge(
                df_m,
                on="seed",
                suffixes=("_p", "_m")
            )

            if merged.empty:
                continue

            x = merged[f"{metric}_p"]
            y = merged[f"{metric}_m"]

            # 小さい方が良い指標は符号反転
            if not HIGHER_IS_BETTER[metric]:
                x = -x
                y = -y

            stat, p_value = wilcoxon(x, y, alternative="greater")

            records.append({
                "experiment": exp,
                "metric": metric,
                "baseline": method,
                "p_value": p_value,
                "significant": p_value < alpha,
            })

    return pd.DataFrame(records)


def significant_ratio(df_wilcoxon):
    """
    baseline × metric ごとに有意だった実験割合
    """
    return (
        df_wilcoxon
        .groupby(["metric", "baseline"])["significant"]
        .mean()
        .reset_index(name="significant_ratio")
    )
