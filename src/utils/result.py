import os
import csv
import numpy as np
from pathlib import Path

def generate_header(model_dict):
    header = ["Iteration"]
    metrics = ["accuracy", "ce", "softacc", "mse", "KL", "HD", "JSD"]
    for metric in metrics:
        for model in model_dict.keys():
            header.append(f"{metric}_{model}")
    return header

def ensure_csv_header(csv_path, header):
    """CSVファイルにヘッダーがあることを保証する。なければ作成。"""
    # ディレクトリが存在しない場合は作成
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # ファイルが既に存在すればスキップ
    if csv_path.exists():
        return
    
    # 無ければ新規にヘッダーを書き出す
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)

def append_row(csv_path, row):
    """CSVファイルに行を追加する。"""
    csv_path = Path(csv_path)
    # ディレクトリが存在しない場合は作成（念のため）
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)


def merge_metrics(dict_a, dict_b):
    """
    {model: {metric: val}} 形式をマージ。
    同名キーは dict_b で上書き（通常は重複なし想定）。
    """
    merged = {}
    for k in set(dict_a.keys()) | set(dict_b.keys()):
        merged[k] = {}
        if k in dict_a:
            merged[k].update(dict_a[k])
        if k in dict_b:
            merged[k].update(dict_b[k])
    return merged

def make_row(iteration, header, model_metrics):
    """ヘッダー順に値を並べ、無いところは NaN で埋める。"""
    row = []
    for col in header:
        if col == "Iteration":
            row.append(iteration)
        else:
            metric, model_name = col.split("_", 1)
            row.append(model_metrics.get(model_name, {}).get(metric, np.nan))
    return row
