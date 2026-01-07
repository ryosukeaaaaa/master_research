import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


from src.utils.config_synthetic import load_yaml, generate_grid
from src.utils.seed import set_seed, derive_seed
from src.utils.device import get_device
from src.methods.registry import build_method
from src.datasets.generate_syntheticdata import dependency_matrix, generate_data
from src.datasets.factory import build_dataset, build_test_dataset
from src.evaluation.evaluator import evaluate_testdata, evaluate_distribution

base_cfg = load_yaml("configs/synthetic/base.yaml")
grid_cfg = load_yaml("configs/synthetic/grid.yaml")

# グリッドパラメータの組み合わせ数をカウント
grid_count = 0
for _ in generate_grid(grid_cfg):
    grid_count += 1

# 各パラメータの数を表示
print("=== パラメータ数 ===")
for key, values in grid_cfg.items():
    print(f"{key}: {len(values)}通り")

print(f"\nグリッド組み合わせ数: {grid_count}通り")
print(f"シード数: {len(base_cfg['seeds'])}個")
print(f"手法数: {len(base_cfg['methods'])}個")
print(f"\n総実験数: {grid_count} × {len(base_cfg['seeds'])} × {len(base_cfg['methods'])} = {grid_count * len(base_cfg['seeds']) * len(base_cfg['methods'])}回")

"""
例：
{'n_students': 50, 'n_skills': 5, 'smoothing': 0.1, 'current_alpha_beta': [2.5, 2.5], 'future_alpha_beta': [2.5, 2.5]}
"""

def main():
    base_cfg = load_yaml("configs/synthetic/base.yaml")
    grid_cfg = load_yaml("configs/synthetic/grid.yaml")
    train_cfg = load_yaml("configs/synthetic/train.yaml")

    device = get_device()

    for exp_cfg in generate_grid(grid_cfg):
        print("===")
        print("Experiment Configuration:")
        print(exp_cfg)
        print("===")
        
        # === 既存結果の読み込み ===
        results_dir = Path(base_cfg["results_folder"])
        results_dir.mkdir(parents=True, exist_ok=True)
        exp_name = "_".join([f"{k}-{v}" for k, v in exp_cfg.items()])
        csv_path = results_dir / f"{exp_name}.csv"
        
        if csv_path.exists():
            existing_df = pd.read_csv(csv_path)
            print(f"Loaded existing results from {csv_path}")
        else:
            existing_df = pd.DataFrame()
            print(f"No existing results found, starting fresh")
        
        records = []

        for seed in base_cfg["seeds"]:
            A = dependency_matrix(exp_cfg["n_skills"])
            current_dataset, future_dataset = generate_data(A, seed=seed, cfg=exp_cfg)
            _, test_dataset = generate_data(A, seed=seed + 10000, cfg=exp_cfg, n_test=100)

            for method_cfg in base_cfg["methods"]:
                # 既存結果にこのseed×methodの組み合わせがあるかチェック
                if not existing_df.empty:
                    exists = ((existing_df["seed"] == seed) & 
                             (existing_df["method"] == method_cfg["id"])).any()
                    if exists:
                        print(f"  Skipping seed={seed}, method={method_cfg['id']} (already exists)")
                        continue
                
                print(f"  Running seed={seed}, method={method_cfg['id']}")
                method_seed = derive_seed(seed, method_cfg["id"])  # methodごとに分ける
                set_seed(method_seed)

                # モデル定義
                method = build_method(
                    method_cfg=method_cfg,
                    device=device,
                    seed=method_seed,
                    exp_cfg=exp_cfg,
                    training_cfg=train_cfg,
                )

                # モデル学習
                train_data = build_dataset(
                    method_cfg=method_cfg,
                    current_dataset=current_dataset,
                    future_dataset=future_dataset,
                    device=device,
                )
                method.fit(train_data)

                # テストデータ予測
                test_data = build_test_dataset(test_dataset=test_dataset, device=device)
                out = method.predict(test_data)

                # 評価
                dist_metrics = base_cfg["evaluation"]["distribution_metrics"]
                test_metrics = base_cfg["evaluation"]["test_metrics"]

                scores_distribution = evaluate_distribution(
                    A=A,
                    model=method,
                    metrics=dist_metrics,
                    smoothing=exp_cfg["smoothing"],
                )
                scores_testdata = evaluate_testdata(
                    pred=out,
                    input=test_data["X_test"].cpu().numpy(),
                    target=test_data["y_test"].cpu().numpy(),
                    metrics=test_metrics,
                )
                record = {
                    "seed": seed,
                    "method": method_cfg["id"],
                }
                for k, v in scores_distribution.items():
                    record[k] = v
                for k, v in scores_testdata.items():
                    record[k] = v
                records.append(record)

        # === 保存 ===
        if records:
            new_df = pd.DataFrame(records)
            # 既存データと新規データを結合
            if not existing_df.empty:
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined_df = new_df
            
            combined_df.to_csv(csv_path, index=False)
            print(f"Saved {len(records)} new results to {csv_path}")
        else:
            print(f"No new results to save for this experiment configuration")

if __name__ == "__main__":
    main()