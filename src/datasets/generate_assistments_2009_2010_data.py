# ASSISTments 2009-2010データセットの加工・生成ファイル
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def generate_data(
    data_dir: str,
    n_skills: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    ASSISTments 2009-2010データセットから研究用データを生成する。
    
    前提: 一度習得したスキルは忘れない
    処理:
        1. 前半・後半のスキル状態を読み込み
        2. 後半データに対して前半データとの包含的論理和を取る（新・後半データ）
        3. current_dataset: [(初期状態=0ベクトル, 前半データ)]
        4. future_dataset: [(前半データ, 新・後半データ)]
    
    Args:
        data_dir (str): データディレクトリパス
            例: 'data/processed/assistments_2009_2010/min150_k10'
        n_skills (int): スキル数（K）
    
    Returns:
        tuple: (current_dataset, future_dataset)
            - current_dataset: (n_students, 2, n_skills)の配列
                [:, 0, :] = 初期状態（ゼロベクトル）
                [:, 1, :] = 前半のスキル状態
            - future_dataset: (n_students, 2, n_skills)の配列
                [:, 0, :] = 前半のスキル状態
                [:, 1, :] = 新・後半のスキル状態（前半との論理和）
    """
    
    data_path = Path(data_dir)
    
    # スキル状態データを読み込み
    skill_states_first_path = data_path / 'skill_states_first.csv'
    skill_states_second_path = data_path / 'skill_states_second.csv'
    
    if not skill_states_first_path.exists():
        raise FileNotFoundError(f"前半データが見つかりません: {skill_states_first_path}")
    if not skill_states_second_path.exists():
        raise FileNotFoundError(f"後半データが見つかりません: {skill_states_second_path}")
    
    # CSVからスキル状態を読み込み
    df_first = pd.read_csv(skill_states_first_path)
    df_second = pd.read_csv(skill_states_second_path)
    
    # user_id列を除いてスキル状態のみ抽出
    skill_cols = [col for col in df_first.columns if col.startswith('skill_')]
    
    if len(skill_cols) != n_skills:
        raise ValueError(
            f"指定されたスキル数({n_skills})とデータのスキル数({len(skill_cols)})が一致しません"
        )
    
    # numpy配列に変換
    first_states = df_first[skill_cols].values.astype(np.float32)  # (n_students, n_skills)
    second_states = df_second[skill_cols].values.astype(np.float32)  # (n_students, n_skills)
    
    n_students = len(first_states)
    
    # ユーザーIDの一致を確認
    if not np.array_equal(df_first['user_id'].values, df_second['user_id'].values):
        raise ValueError("前半と後半でユーザーIDが一致しません")
    
    # 新・後半データ = 前半データ OR 後半データ（一度習得したスキルは忘れない）
    new_second_states = np.maximum(first_states, second_states).astype(np.float32)
    
    # 初期状態（ゼロベクトル）を生成
    initial_states = np.zeros((n_students, n_skills), dtype=np.float32)
    
    # current_dataset: [(初期状態, 前半データ)]
    current_dataset = np.stack([initial_states, first_states], axis=1)  # (n_students, 2, n_skills)
    
    # future_dataset: [(前半データ, 新・後半データ)]
    future_dataset = np.stack([first_states, new_second_states], axis=1)  # (n_students, 2, n_skills)
    
    # 統計情報を表示
    print(f"データセット生成完了:")
    print(f"  学生数: {n_students}")
    print(f"  スキル数: {n_skills}")
    print(f"  前半平均習得率: {first_states.mean():.2%}")
    print(f"  後半平均習得率（元）: {second_states.mean():.2%}")
    print(f"  後半平均習得率（新）: {new_second_states.mean():.2%}")
    
    # スキルの習得・喪失の分析
    print(f"\nスキル状態の変化:")
    
    # 習得: 前半で0、後半で1
    acquired = np.sum((first_states == 0) & (second_states == 1))
    acquired_rate = acquired / (n_students * n_skills)
    print(f"  習得: {acquired}件 ({acquired_rate:.2%})")
    
    # 喪失: 前半で1、後半で0
    lost = np.sum((first_states == 1) & (second_states == 0))
    lost_rate = lost / (n_students * n_skills)
    if lost > 0:
        print(f"  ⚠️  喪失: {lost}件 ({lost_rate:.2%}) → 論理和で修正済み")
    else:
        print(f"  喪失: {lost}件 (なし)")
    
    # 維持: 前半で1、後半でも1
    maintained = np.sum((first_states == 1) & (second_states == 1))
    maintained_rate = maintained / (n_students * n_skills)
    print(f"  維持: {maintained}件 ({maintained_rate:.2%})")
    
    # 未習得のまま: 前半で0、後半でも0
    unlearned = np.sum((first_states == 0) & (second_states == 0))
    unlearned_rate = unlearned / (n_students * n_skills)
    print(f"  未習得: {unlearned}件 ({unlearned_rate:.2%})")
    
    # 習得スキル数の統計
    first_mastered = first_states.sum(axis=1)
    second_mastered_original = second_states.sum(axis=1)
    second_mastered = new_second_states.sum(axis=1)
    
    print(f"\n学生ごとの習得スキル数:")
    print(f"  前半平均: {first_mastered.mean():.2f} / {n_skills}")
    print(f"  後半平均（元）: {second_mastered_original.mean():.2f} / {n_skills}")
    print(f"  後半平均（新）: {second_mastered.mean():.2f} / {n_skills}")
    print(f"  平均純増: {(second_mastered - first_mastered).mean():.2f}")
    print(f"  平均純増（元）: {(second_mastered_original - first_mastered).mean():.2f}")
    
    return current_dataset, future_dataset


def load_skill_mapping(data_dir: str) -> dict:
    """
    スキルマッピング情報を読み込む。
    
    Args:
        data_dir (str): データディレクトリパス
    
    Returns:
        dict: スキルマッピング情報
    """
    data_path = Path(data_dir)
    skill_mapping_path = data_path / 'skill_mapping.json'
    
    if not skill_mapping_path.exists():
        raise FileNotFoundError(f"スキルマッピングが見つかりません: {skill_mapping_path}")
    
    import json
    with open(skill_mapping_path, 'r') as f:
        mapping = json.load(f)
    
    return mapping


def get_data_info(data_dir: str) -> dict:
    """
    データセット情報を読み込む。
    
    Args:
        data_dir (str): データディレクトリパス
    
    Returns:
        dict: データセット情報
    """
    data_path = Path(data_dir)
    info_path = data_path.parent / 'data_info.json'
    
    if not info_path.exists():
        raise FileNotFoundError(f"データ情報が見つかりません: {info_path}")
    
    import json
    with open(info_path, 'r') as f:
        info = json.load(f)
    
    return info


if __name__ == '__main__':
    # 使用例
    data_dir_k10 = 'outputs/assistments_2009_2010/dina_estimation/min150_k10'
    data_dir_k5 = 'outputs/assistments_2009_2010/dina_estimation/min150_k5'
    
    print("="*80)
    print("K=10 データセット生成")
    print("="*80)
    current_k10, future_k10 = generate_data(data_dir_k10, n_skills=10)
    print(f"current_dataset shape: {current_k10.shape}")
    print(f"future_dataset shape: {future_k10.shape}")
    
    print("\n" + "="*80)
    print("K=5 データセット生成")
    print("="*80)
    current_k5, future_k5 = generate_data(data_dir_k5, n_skills=5)
    print(f"current_dataset shape: {current_k5.shape}")
    print(f"future_dataset shape: {future_k5.shape}")
