import pandas as pd
from pathlib import Path

def load_all_results(results_dir):
    """
    results_dir: outputs/synthetic/results
    """
    results_dir = Path(results_dir)
    dfs = []

    for csv_path in results_dir.glob("*.csv"):
        df = pd.read_csv(csv_path)
        df["experiment"] = csv_path.stem  # 実験ID
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No result CSV files found.")

    return pd.concat(dfs, ignore_index=True)
