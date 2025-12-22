import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


from src.utils.config_synthetic import load_yaml, generate_grid
from src.utils.seed import set_seed, derive_seed
from src.utils.device import get_device
from src.methods.registry import build_method
from src.datasets.generate_assistments_2009_2010_data import generate_data
from src.datasets.factory import build_dataset, build_test_dataset
from src.evaluation.evaluator import evaluate_testdata, evaluate_distribution


def main():
    base_cfg = load_yaml("configs/Assistments_2009_2010/base.yaml")
    train_cfg = load_yaml("configs/Assistments_2009_2010/train.yaml")

    device = get_device()

    for dataset_cfg in base_cfg["dataset"]:
        records = []  # データセットごとに初期化
        pre_current_dataset, pre_future_dataset = generate_data(data_dir=dataset_cfg["id"], n_skills=dataset_cfg["n_skills"])
        split_ratio = base_cfg["split_ratio"]
        print("あああああ")
        print(pre_current_dataset.shape, pre_future_dataset.shape)
        for seed in base_cfg["seeds"]:        
            # seedを使ってシャッフル
            n_samples = len(pre_current_dataset)
            rng = np.random.RandomState(seed)
            indices = rng.permutation(n_samples)
            
            # シャッフルされたデータ
            shuffled_current = pre_current_dataset[indices]
            shuffled_future = pre_future_dataset[indices]
            
            # train/test分割
            split_idx = int(n_samples * split_ratio)
            current_dataset = shuffled_current[:split_idx]
            future_dataset = shuffled_future[:split_idx]
            test_future = shuffled_future[split_idx:]
            
            test_data = {
                "X_test": torch.from_numpy(test_future[:, 0, :]).to(device),
                "y_test": torch.from_numpy(test_future[:, 1, :]).to(device),
            }
            for method_cfg in base_cfg["methods"]:
                method_seed = derive_seed(seed, method_cfg["id"])  # methodごとに分ける
                set_seed(method_seed)

                # exp_cfgを作成（proposedメソッドの場合のみ必要）
                if method_cfg["id"] in ["proposed", "full_information"]:
                    exp_cfg = {
                        "n_students": len(current_dataset),
                        "n_skills": dataset_cfg["n_skills"],
                        "dataset_name": dataset_cfg["name"],
                    }
                else:
                    exp_cfg = None

                # モデル定義
                method = build_method(
                    method_cfg=method_cfg,
                    device=device,
                    seed=method_seed,
                    exp_cfg=exp_cfg,
                    training_cfg=train_cfg if exp_cfg is not None else None,
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
                out = method.predict(test_data)

                # 評価
                test_metrics = base_cfg["evaluation"]["test_metrics"]

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
                for k, v in scores_testdata.items():
                    record[k] = v
                records.append(record)

        # === 保存 ===
        results_dir = Path(base_cfg["results_folder"])
        results_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(records)
        exp_name = dataset_cfg["name"]
        csv_path = results_dir / f"{exp_name}_{train_cfg['recipe']['regularization']['type']}_{train_cfg['recipe']['regularization']['lambda']}.csv"

        df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    main()