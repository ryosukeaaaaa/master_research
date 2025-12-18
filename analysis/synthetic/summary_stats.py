def summarize_mean_std(df, metrics):
    """
    全実験・全 seed をまとめた method ごとの Mean / Std
    """
    summary = (
        df
        .groupby("method")[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    return summary
