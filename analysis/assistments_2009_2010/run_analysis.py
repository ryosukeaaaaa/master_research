"""
ASSISTments 2009-2010 実験結果の分析スクリプト

実験設定:
- K=5 と K=10 の2つのデータセット
- 各データセットで seed 0-99 の100パターン実行
- 各指標について mean, std を計算
- proposed が各指標で勝利した回数をカウント
- Wilcoxon符号順位検定で有意性を判定
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

# 指標の向き（大きい方が良いか、小さい方が良いか）
HIGHER_IS_BETTER = {
    "acc": True,
    "spa": True,
    "ace": False,
    "mse": False,
    "kl": False,
    "hd": False,
    "jsd": False,
}


def load_results(results_dir: str) -> dict:
    """
    結果CSVを読み込む（ファイルごとに分けて返す）
    
    Args:
        results_dir: 結果ディレクトリのパス
    
    Returns:
        dict: ファイル名をキー、DataFrameを値とする辞書
    """
    results_path = Path(results_dir)
    csv_files = list(results_path.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"結果CSVが見つかりません: {results_path}")
    
    # 各CSVファイルを読み込んでディクショナリに格納
    results_dict = {}
    for csv_file in csv_files:
        dataset_name = csv_file.stem  # ファイル名（拡張子なし）
        df = pd.read_csv(csv_file)
        results_dict[dataset_name] = df
    
    return results_dict


def compute_statistics(df: pd.DataFrame, metrics: list, dataset_name: str) -> pd.DataFrame:
    """
    メソッドごとに各指標の平均と標準偏差を計算
    
    Args:
        df: 実験結果のDataFrame
        metrics: 評価指標のリスト
        dataset_name: データセット名
    
    Returns:
        DataFrame: 統計量
    """
    records = []
    
    for method in df["method"].unique():
        df_method = df[df["method"] == method]
        
        record = {
            "dataset": dataset_name,
            "method": method,
            "n_samples": len(df_method),
        }
        
        for metric in metrics:
            if metric in df_method.columns:
                record[f"{metric}_mean"] = df_method[metric].mean()
                record[f"{metric}_std"] = df_method[metric].std()
        
        records.append(record)
    
    return pd.DataFrame(records)


def compute_win_rates(df: pd.DataFrame, metrics: list, dataset_name: str, proposed_method: str = "proposed") -> pd.DataFrame:
    """
    proposed が各ベースラインに対して勝利した回数と勝率を計算
    
    Args:
        df: 実験結果のDataFrame
        metrics: 評価指標のリスト
        dataset_name: データセット名
        proposed_method: 提案手法の名前
    
    Returns:
        DataFrame: 勝率
    """
    records = []
    
    df_proposed = df[df["method"] == proposed_method]
    
    if df_proposed.empty:
        return pd.DataFrame()
    
    for method in df["method"].unique():
        if method == proposed_method:
            continue
        
        df_baseline = df[df["method"] == method]
        
        # seedで結合
        merged = df_proposed.merge(
            df_baseline,
            on="seed",
            suffixes=("_p", "_b")
        )
        
        if merged.empty:
            continue
        
        record = {
            "dataset": dataset_name,
            "baseline": method,
            "n_trials": len(merged),
        }
        
        for metric in metrics:
            if f"{metric}_p" not in merged.columns or f"{metric}_b" not in merged.columns:
                continue
            
            x_p = merged[f"{metric}_p"]
            x_b = merged[f"{metric}_b"]
            
            # 勝利条件
            if HIGHER_IS_BETTER[metric]:
                wins = (x_p > x_b).sum()
            else:
                wins = (x_p < x_b).sum()
            
            record[f"{metric}_wins"] = wins
            record[f"{metric}_win_rate"] = wins / len(merged)
        
        records.append(record)
    
    return pd.DataFrame(records)


def wilcoxon_test(df: pd.DataFrame, metrics: list, dataset_name: str, proposed_method: str = "proposed", alpha: float = 0.05) -> pd.DataFrame:
    """
    Wilcoxon符号順位検定で有意性を判定
    
    Args:
        df: 実験結果のDataFrame
        metrics: 評価指標のリスト
        dataset_name: データセット名
        proposed_method: 提案手法の名前
        alpha: 有意水準
    
    Returns:
        DataFrame: 検定結果
    """
    records = []
    
    df_proposed = df[df["method"] == proposed_method]
    
    if df_proposed.empty:
        return pd.DataFrame()
    
    for method in df["method"].unique():
        if method == proposed_method:
            continue
        
        df_baseline = df[df["method"] == method]
        
        # seedで結合
        merged = df_proposed.merge(
            df_baseline,
            on="seed",
            suffixes=("_p", "_b")
        )
        
        if merged.empty:
            continue
        
        for metric in metrics:
            if f"{metric}_p" not in merged.columns or f"{metric}_b" not in merged.columns:
                continue
            
            x = merged[f"{metric}_p"]
            y = merged[f"{metric}_b"]
            
            # 小さい方が良い指標は符号反転
            if not HIGHER_IS_BETTER[metric]:
                x = -x
                y = -y
            
            # Wilcoxon符号順位検定（片側検定: proposed > baseline）
            try:
                stat, p_value = wilcoxon(x, y, alternative="greater")
                significant = p_value < alpha
            except Exception as e:
                stat = np.nan
                p_value = np.nan
                significant = False
            
            records.append({
                "dataset": dataset_name,
                "metric": metric,
                "baseline": method,
                "statistic": stat,
                "p_value": p_value,
                "significant": significant,
                "alpha": alpha,
            })
    
    return pd.DataFrame(records)


def main():
    """メイン処理"""
    # 結果ディレクトリ
    results_dir = "outputs/assistments_2009_2010/results"
    output_dir = Path("analysis/assistments_2009_2010")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 評価指標
    metrics = ["acc", "spa", "ace", "mse", "kl", "hd", "jsd"]
    
    print("="*80)
    print("ASSISTments 2009-2010 実験結果の分析")
    print("="*80)
    
    # 結果読み込み
    print("\n[1] 結果の読み込み")
    results_dict = load_results(results_dir)
    print(f"  データセット数: {len(results_dict)}")
    for dataset_name, df in results_dict.items():
        print(f"  - {dataset_name}: {len(df)} records")
    
    # 各データセットごとに分析
    all_stats = []
    all_wins = []
    all_wilcoxon = []
    
    for dataset_name, df in results_dict.items():
        print("\n" + "="*80)
        print(f"データセット: {dataset_name}")
        print("="*80)
        
        print(f"\n  手法: {df['method'].unique()}")
        print(f"  seed範囲: {df['seed'].min()} - {df['seed'].max()}")
        print(f"  総レコード数: {len(df)}")
        
        # 統計量の計算
        print(f"\n[2-{dataset_name}] 統計量の計算（mean, std）")
        df_stats = compute_statistics(df, metrics, dataset_name)
        all_stats.append(df_stats)
        print("\n" + df_stats.to_string(index=False))
        
        # 勝率の計算
        print(f"\n[3-{dataset_name}] 勝率の計算")
        df_wins = compute_win_rates(df, metrics, dataset_name)
        all_wins.append(df_wins)
        print("\n" + df_wins.to_string(index=False))
        
        # Wilcoxon検定
        print(f"\n[4-{dataset_name}] Wilcoxon符号順位検定")
        df_wilcoxon = wilcoxon_test(df, metrics, dataset_name)
        all_wilcoxon.append(df_wilcoxon)
        if not df_wilcoxon.empty:
            print("\n" + df_wilcoxon.to_string(index=False))
        else:
            print("  検定結果なし（proposedメソッドまたはベースラインが見つかりません）")
        
        # 有意な結果のサマリー
        print(f"\n[5-{dataset_name}] 有意な結果のサマリー")
        if not df_wilcoxon.empty:
            df_significant = df_wilcoxon[df_wilcoxon["significant"]]
            if not df_significant.empty:
                print(f"  有意な結果: {len(df_significant)} / {len(df_wilcoxon)}")
                print("\n" + df_significant.to_string(index=False))
            else:
                print("  有意な結果はありませんでした")
        else:
            print("  検定結果なし")
    
    # 全データセットの結果を統合して保存
    print("\n" + "="*80)
    print("結果の保存")
    print("="*80)
    
    df_all_stats = pd.concat(all_stats, ignore_index=True)
    stats_path = output_dir / "statistics.csv"
    df_all_stats.to_csv(stats_path, index=False)
    print(f"  統計量: {stats_path}")
    
    df_all_wins = pd.concat(all_wins, ignore_index=True)
    wins_path = output_dir / "win_rates.csv"
    df_all_wins.to_csv(wins_path, index=False)
    print(f"  勝率: {wins_path}")
    
    df_all_wilcoxon = pd.concat(all_wilcoxon, ignore_index=True)
    wilcoxon_path = output_dir / "wilcoxon_results.csv"
    df_all_wilcoxon.to_csv(wilcoxon_path, index=False)
    print(f"  Wilcoxon検定: {wilcoxon_path}")
    
    print("\n" + "="*80)
    print("分析完了")
    print("="*80)


if __name__ == "__main__":
    main()
