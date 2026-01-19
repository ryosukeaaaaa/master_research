# analysis/synthetic/parameter_analysis.py

import pandas as pd
import numpy as np
import re
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns

from wilcoxon_test import HIGHER_IS_BETTER


def parse_experiment_params(exp_name):
    """実験名からパラメータを抽出"""
    params = {}
    
    patterns = {
        'n_students': r'n_students-(\d+)',
        'n_skills': r'n_skills-(\d+)',
        'smoothing': r'smoothing-([\d.]+)',
        'current_alpha_beta': r'current_alpha_beta-\[([\d., ]+)\]',
        'future_alpha_beta': r'future_alpha_beta-\[([\d., ]+)\]',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, exp_name)
        if match:
            if key in ['n_students', 'n_skills']:
                params[key] = int(match.group(1))
            elif key == 'smoothing':
                params[key] = float(match.group(1))
            else:
                # alpha, betaの形式で保存
                values = match.group(1).replace(' ', '')
                params[key] = values
    
    return params


def compute_cliff_delta(x, y):
    """
    Cliff's Delta を計算（効果量）
    |delta| < 0.147: negligible
    |delta| < 0.33: small
    |delta| < 0.474: medium
    |delta| >= 0.474: large
    """
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    
    greater = sum((xi > yi) for xi in x for yi in y)
    less = sum((xi < yi) for xi in x for yi in y)
    
    return (greater - less) / (n1 * n2)


def analyze_single_parameter(df, metric, parameter, proposed_method="proposed", alpha=0.05, verbose=True):
    """
    単一パラメータの影響を詳細に分析
    
    重要な指標:
    - win_rate: 勝率（実務的な意味）
    - mean_diff: 平均性能差（実質的な改善量）
    - effect_size: Cliff's Delta（効果の大きさ）
    - p_value: 統計的有意性
    """
    # パラメータを抽出
    if parameter not in df.columns:
        if verbose:
            print(f"    パラメータ抽出中...")
        params_df = df['experiment'].apply(parse_experiment_params).apply(pd.Series)
        df = pd.concat([df, params_df], axis=1)
    
    results = []
    param_values = sorted([v for v in df[parameter].unique() if pd.notna(v)])
    
    if verbose:
        print(f"    パラメータ値: {param_values[:10]}{'...' if len(param_values) > 10 else ''}")
        print(f"    処理中...", end='', flush=True)
    
    processed = 0
    baselines = [m for m in df['method'].unique() if m != proposed_method]
    total = len(param_values) * len(baselines)
    
    for param_value in param_values:
        df_param = df[df[parameter] == param_value]
        
        for baseline in baselines:
            processed += 1
            if verbose and processed % 10 == 0:
                print(f" [{processed}/{total}]", end='', flush=True)
            
            df_p = df_param[df_param['method'] == proposed_method]
            df_b = df_param[df_param['method'] == baseline]
            
            merged = df_p.merge(
                df_b,
                on=['experiment', 'seed'],
                suffixes=('_p', '_b')
            )
            
            if len(merged) < 3:  # 最低限のサンプル数
                continue
            
            x = merged[f"{metric}_p"].values
            y = merged[f"{metric}_b"].values
            
            # 勝率と性能差
            if HIGHER_IS_BETTER[metric]:
                wins = (x > y).sum()
                ties = (x == y).sum()
                diff = x - y
                effect_size = compute_cliff_delta(x, y)
            else:
                wins = (x < y).sum()
                ties = (x == y).sum()
                diff = y - x
                effect_size = compute_cliff_delta(-x, -y)
            
            win_rate = wins / len(merged)
            
            # Wilcoxon検定
            try:
                if HIGHER_IS_BETTER[metric]:
                    stat, p_value = wilcoxon(x, y, alternative='greater')
                else:
                    stat, p_value = wilcoxon(x, y, alternative='less')
            except:
                p_value = np.nan
            
            results.append({
                parameter: param_value,
                'baseline': baseline,
                'metric': metric,
                'n_experiments': merged['experiment'].nunique(),
                'n_samples': len(merged),
                'win_rate': win_rate,
                'tie_rate': ties / len(merged),
                'mean_diff': diff.mean(),
                'median_diff': np.median(diff),
                'std_diff': diff.std(),
                'effect_size': effect_size,
                'p_value': p_value,
                'significant': p_value < alpha if not np.isnan(p_value) else False,
            })
    
    if verbose:
        print(f" 完了!")
    
    return pd.DataFrame(results)


def interpret_effect_size(delta):
    """効果量の解釈"""
    if pd.isna(delta):
        return "unknown"
    abs_delta = abs(delta)
    if abs_delta < 0.147:
        return "negligible"
    elif abs_delta < 0.33:
        return "small"
    elif abs_delta < 0.474:
        return "medium"
    else:
        return "large"


def comprehensive_single_parameter_analysis(df, metrics, output_dir):
    """
    各パラメータについて単変量分析を実行
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 全パラメータを分析対象に
    parameters = ['n_students', 'n_skills', 'smoothing', 
                  'current_alpha_beta', 'future_alpha_beta']
    
    all_results = {}
    total_params = len(parameters)
    
    for param_idx, param in enumerate(parameters, 1):
        print(f"\n{'='*60}")
        print(f"[{param_idx}/{total_params}] 分析中: {param}")
        print('='*60)
        
        param_results = []
        
        for metric_idx, metric in enumerate(metrics, 1):
            print(f"  [{metric_idx}/{len(metrics)}] メトリクス: {metric}")
            
            result = analyze_single_parameter(df, metric, param, verbose=True)
            
            if not result.empty:
                result['effect_interpretation'] = result['effect_size'].apply(interpret_effect_size)
                param_results.append(result)
                print(f"    ✓ {len(result)} 件の結果を取得")
            else:
                print(f"    ⚠ 結果なし")
        
        if param_results:
            combined = pd.concat(param_results, ignore_index=True)
            combined = combined.sort_values([param, 'metric', 'baseline'])
            
            # 詳細結果を保存
            output_file = output_dir / f"parameter_{param}.csv"
            combined.to_csv(output_file, index=False)
            all_results[param] = combined
            
            print(f"\n  ✓ 保存完了: {output_file.name} ({len(combined)} rows)")
            
            # サマリーを表示
            print(f"\n  主要な発見 ({param}):")
            
            # 効果が大きい条件を抽出
            strong_effects = combined[
                (combined['effect_size'].abs() > 0.33) &  # medium以上
                (combined['significant'] == True) &
                (combined['win_rate'] > 0.6)
            ].sort_values('effect_size', ascending=False)
            
            if not strong_effects.empty:
                print(f"    効果が顕著な条件 ({len(strong_effects)}件):")
                for _, row in strong_effects.head(5).iterrows():
                    print(f"      • {param}={row[param]}, {row['metric']} vs {row['baseline']}: "
                          f"勝率={row['win_rate']:.2f}, 効果量={row['effect_size']:.3f} ({row['effect_interpretation']}), "
                          f"p={row['p_value']:.4f}")
            else:
                print("    → 顕著な効果は見られませんでした")
    
    return all_results


def create_summary_table(all_results, output_dir):
    """
    論文用のサマリーテーブルを作成
    """
    print("  集計中...", end='', flush=True)
    
    output_dir = Path(output_dir)
    summary_rows = []
    
    for param, df in all_results.items():
        # パラメータ × メトリクスで集約
        for metric in df['metric'].unique():
            df_metric = df[df['metric'] == metric]
            
            # 全ベースラインの平均
            avg_win_rate = df_metric['win_rate'].mean()
            avg_effect = df_metric['effect_size'].mean()
            sig_ratio = df_metric['significant'].mean()
            
            summary_rows.append({
                'parameter': param,
                'metric': metric,
                'avg_win_rate': avg_win_rate,
                'avg_effect_size': avg_effect,
                'significant_ratio': sig_ratio,
                'effect_interpretation': interpret_effect_size(avg_effect)
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(['parameter', 'metric'])
    summary_df.to_csv(output_dir / "overall_summary.csv", index=False)
    
    print(" 完了!")
    
    return summary_df


def create_visualizations(all_results, output_dir):
    """
    論文用の図を作成
    """
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    
    # 数値パラメータのみ可視化（alpha_betaは文字列なのでスキップ）
    numeric_params = ['n_students', 'n_skills', 'smoothing']
    
    total_figs = sum(
        len(all_results[p]['metric'].unique()) 
        for p in numeric_params if p in all_results
    )
    fig_count = 0
    
    for param in numeric_params:
        if param not in all_results:
            continue
            
        df = all_results[param]
        metrics = df['metric'].unique()
        
        for metric in metrics:
            fig_count += 1
            print(f"    [{fig_count}/{total_figs}] {metric} × {param}...", end='', flush=True)
            
            df_metric = df[df['metric'] == metric]
            
            # ベースラインごとに色分け
            plt.figure(figsize=(10, 6))
            
            for baseline in df_metric['baseline'].unique():
                df_baseline = df_metric[df_metric['baseline'] == baseline]
                
                plt.plot(df_baseline[param], df_baseline['effect_size'], 
                        marker='o', label=baseline, linewidth=2, markersize=8)
            
            plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            plt.xlabel(param, fontsize=12)
            plt.ylabel(f'Effect Size (Cliff\'s Delta)', fontsize=12)
            plt.title(f'{metric.upper()}: Effect Size vs {param}', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(fig_dir / f"effect_{metric}_{param}.png", dpi=300)
            plt.close()
            
            print(" ✓")


if __name__ == "__main__":
    from load_results import load_all_results
    
    RESULTS_DIR = "outputs/synthetic/modify_alpha_beta/results"
    OUT_DIR = "analysis/synthetic/modify_alpha_beta/parameter_analysis"
    # RESULTS_DIR = "outputs/synthetic/results"
    # OUT_DIR = "analysis/synthetic/parameter_analysis"
    METRICS = ["kl", "hd", "jsd", "acc", "ace", "spa", "mse"]
    
    print("="*60)
    print("パラメータ有効性分析（全パラメータ）")
    print("="*60)
    
    print("\n[1/4] データ読み込み中...")
    df = load_all_results(RESULTS_DIR)
    print(f"✓ 読み込み完了: {len(df):,} rows, {df['experiment'].nunique()} experiments")
    print(f"  手法: {df['method'].unique().tolist()}")
    
    # メイン分析
    print("\n[2/4] パラメータ分析（n_students, n_skills, smoothing, alpha_beta）")
    results = comprehensive_single_parameter_analysis(df, METRICS, OUT_DIR)
    
    # サマリー作成
    print("\n[3/4] サマリーテーブル作成")
    summary = create_summary_table(results, OUT_DIR)
    print(f"  ✓ 保存完了: overall_summary.csv")
    
    # 可視化
    print("\n[4/4] 可視化（数値パラメータのみ）")
    create_visualizations(results, OUT_DIR)
    print(f"  ✓ 図を保存完了: {OUT_DIR}/figures/")
    
    print("\n" + "="*60)
    print("✓ 分析完了!")
    print(f"結果ディレクトリ: {OUT_DIR}")
    print("="*60)
    
    # 最終サマリー表示
    print("\n【全体サマリー】")
    print(summary.to_string(index=False))