# 人工データの生成ファイル
import numpy as np
from src.utils.seed import set_seed


# ---ランダムな依存関係行列（下三角行列）を生成する関数---

def dependency_matrix(n):
    """
    nスキルに対する依存関係行列（下三角行列）をランダム生成。
    各行はスキルを表し、列の値が1の場合、そのスキルが習得要件となる。
    下三角行列のため、後のスキルが先のスキルに依存する構造を表現。
    Args:
        n (int): スキル数
    Returns:
        np.ndarray: (n, n)の下三角行列
    """
    matrix = np.tril(np.random.randint(0, 2, (n, n)), k=-1).astype(float)
    return matrix


# ---次の各スキルの習得確率を計算する関数---

def skill_acquisition_probabilities(A, X, smoothing=0.3):
    """
    現在の習得状態から次に習得するスキルの確率を計算。
    ※ Xに全習得した状態は入れない。
    
    Args:
        A (np.ndarray): スキルの依存関係行列 (n, n)
        X (np.ndarray): 現在の習得状態ベクトル (n,) - 0:未習得, 1:習得済み
        smoothing (float): ランダムに習得する確率の重み（0-1）
        
    Returns:
        np.ndarray: 各スキルの習得確率 (n,)
    """
    n = len(X)
    probabilities = np.zeros(n)
    
    # 未習得スキル（X[i] == 0）について、習得要件を満たしているかチェック
    for i in range(n):
        if X[i] == 0:
            probabilities[i] = DINA(A[i, :], X)  # 要件を満たしていれば1
    
    # 習得関係に基づく確率
    # 正規化（必ず習得可能スキルはあるのでゼロ除算は起きない）
    probabilities = (1 - smoothing) * probabilities / np.sum(probabilities)
    
    # 未習得スキルからランダムに習得する確率
    unlearned_mask = (X == 0)
    unlearned_count = np.sum(unlearned_mask)
    probabilities[unlearned_mask] += smoothing / unlearned_count
    
    return probabilities


def DINA(skill_requirements, current_state):
    # skill_requirements が 1 の場所で、current_state も 1 である必要がある
    # つまり、(必要条件) <= (現在の状態) が常に成り立つ
    # skill_req=1, state=0 のときだけ False になる
    return int(np.all(current_state >= skill_requirements))


# ---データ生成---

def _simulate_steps(A, current_state, steps, cfg, record_history=False):
    """指定ステップ数だけスキル習得をシミュレーションする"""
    n = len(current_state)
    state = current_state.copy()
    history = []
    
    for _ in range(steps):
        probs = skill_acquisition_probabilities(A, state, cfg["smoothing"])
        next_skill = np.random.choice(n, p=probs)
        
        prev_state = state.copy()
        state[next_skill] = 1
        
        if record_history:
            history.append((prev_state, state.copy()))
            
    return state, history


def generate_data(A, seed, cfg, n_test=None):
    """
    A: 依存関係行列
    seed: 乱数シード
    cfg: データ生成設定
    n_test: テストデータ数（Noneの場合、cfgのn_studentsを使用
    学生の知識状態データを生成する。
    Returns:
        tuple: (dataset, dataset_all, test_dataset)
            - dataset: [(初期状態(0), 現在のスキル状態)]
            - dataset_all: [(初期状態(0), １ステップ目), ..., (第一回の1ステップ前, 現在のスキル状態)]
            - test_dataset: [(現在のスキル状態, 将来のスキル状態)]
    """
    set_seed(seed)

    n_skills = cfg["n_skills"]
    
    n_students = cfg["n_students"] if n_test is None else n_test

    # 各学生のステップ数をまとめて生成（ベクトル化）
    current_probs = np.random.beta(cfg["current_alpha_beta"][0], cfg["current_alpha_beta"][1], n_students)
    current_counts = np.random.binomial(n_skills, current_probs)

    future_probs = np.random.beta(cfg["future_alpha_beta"][0], cfg["future_alpha_beta"][1], n_students)
    future_counts = np.random.binomial(n_skills - current_counts, future_probs)

    current_dataset, dataset_all, future_dataset = [], [], []

    for i in range(n_students):
        initial_state = np.zeros(n_skills, dtype=int)
        
        # フェーズ1: 初期状態 -> 中間状態
        intermediate_state, history = _simulate_steps(
            A, initial_state, current_counts[i], cfg, record_history=True
        )
        
        current_dataset.append((initial_state, intermediate_state))
        # dataset_all.extend(history)
        
        # フェーズ2: 中間状態 -> 最終状態
        final_state, _ = _simulate_steps(
            A, intermediate_state, future_counts[i], cfg, record_history=False
        )
        
        future_dataset.append((intermediate_state, final_state))

    return (
        np.array(current_dataset, dtype=np.float32),
        # np.array(dataset_all, dtype=np.float32),
        np.array(future_dataset, dtype=np.float32),
    )

