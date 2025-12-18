# 評価指標を定義するファイル
import numpy as np

import itertools
from collections import defaultdict

"""
入力:
    p: np.ndarray, shape=(N, D) または (D,)  ・・・比較元の分布（複数サンプルも可）
    q: np.ndarray, shape=(N, D) または (D,)  ・・・比較先の分布
    epsilon: float ・・・ゼロ割防止のための微小値

出力:
    float: サンプルごとの平均値
"""

def kl_divergence(p, q, epsilon=1e-10):
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return float(np.sum(p * np.log(p / q)))

def hellinger_distance(p, q):
    return float(
        np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2)
    )

def js_divergence(p, q, epsilon=1e-10):
    m = 0.5 * (p + q)
    return float(
        0.5 * kl_divergence(p, m, epsilon)
        + 0.5 * kl_divergence(q, m, epsilon)
    )



"""
evaluate_testdata
"""

def new_skill_accuracy(pred, input, target, rng=None):
    """
    New-Skill Accuracy (個別学習者レベル)

    pred:   np.ndarray, shape=(n_students, n_skills)
            各スキルが次に習得される確率（またはスコア）
    input:  np.ndarray, shape=(n_students, n_skills)
            現在のスキル状態 s^(current) (0/1)
    target: np.ndarray, shape=(n_students, n_skills)
            将来のスキル状態 s^(future) (0/1)
    rng:    np.random.Generator (同値処理用)
    """
    if rng is None:
        rng = np.random.default_rng()

    n_students, n_skills = pred.shape
    total_accuracy = 0.0

    for i in range(n_students):
        s_cur = input[i]
        s_fut = target[i]
        p = pred[i]

        # A = 新規習得スキル集合
        acquired = np.where(s_fut - s_cur == 1)[0]
        k = len(acquired)

        # k = 0 の場合は定義通り 1
        if k == 0:
            total_accuracy += 1.0
            continue

        # 未習得スキル集合
        candidates = np.where(s_cur == 0)[0]

        # 念のため（理論上は起きない）
        if len(candidates) == 0:
            total_accuracy += 1.0
            continue

        scores = p[candidates]

        # 上位 k 個を選択（同値はランダム）
        if k >= len(candidates):
            chosen = candidates
        else:
            shuffled = rng.permutation(len(candidates))
            topk_idx = shuffled[np.argsort(scores[shuffled])[-k:]]
            chosen = candidates[topk_idx]

        hits = np.intersect1d(chosen, acquired).size
        total_accuracy += hits / k

    return total_accuracy / n_students


def acquisition_cross_entropy(pred, input, target, epsilon=1e-10):
    """
    Acquisition Cross-Entropy (ACE)

    pred:   np.ndarray, shape=(n_students, n_skills)
            各スキルが新規習得される確率 p_j
    input:  np.ndarray, shape=(n_students, n_skills)
            現在のスキル状態 s^(current) (0/1)
    target: np.ndarray, shape=(n_students, n_skills)
            将来のスキル状態 s^(future) (0/1)
    epsilon: log(0) 回避用の微小値
    """
    n_students = pred.shape[0]
    total_ace = 0.0

    for i in range(n_students):
        s_cur = input[i]
        s_fut = target[i]
        p = pred[i]

        # A = 新規習得スキル集合
        acquired = np.where(s_fut - s_cur == 1)[0]

        # A = ∅ の場合
        if acquired.size == 0:
            continue  # ACE_i = 0

        probs = np.clip(p[acquired].astype(float), epsilon, 1.0)
        ace_i = -np.mean(np.log(probs))
        total_ace += ace_i

    return total_ace / n_students


def skill_wise_probabilistic_accuracy(pred, input, target):
    """
    Skill-wise Probabilistic Accuracy (SPA)

    pred:   np.ndarray, shape=(n_students, n_skills)
            各スキルが新規習得される確率 p_j
    input:  np.ndarray, shape=(n_students, n_skills)
            現在のスキル状態 s^(current) (0/1)
    target: np.ndarray, shape=(n_students, n_skills)
            将来のスキル状態 s^(future) (0/1)
    """
    n_students = pred.shape[0]
    total_spa = 0.0

    for i in range(n_students):
        s_cur = input[i]
        s_fut = target[i]
        p = pred[i]

        delta = s_fut - s_cur
        acquired = set(np.where(delta == 1)[0])   # A
        not_yet = np.where(s_cur == 0)[0]          # U

        # 全スキル習得済みなら自明に正しい
        if not_yet.size == 0:
            total_spa += 1.0
            continue

        vals = []
        for j in not_yet:
            pj = float(p[j])
            vals.append(pj if j in acquired else 1.0 - pj)

        total_spa += float(np.mean(vals))

    return total_spa / n_students


def skill_state_mse(pred, target):
    """
    Skill-State Mean Squared Error (MSE)

    pred:   np.ndarray, shape=(n_students, n_skills)
            将来状態の予測ベクトル \hat{s}^{(future)}
    target: np.ndarray, shape=(n_students, n_skills)
            真の将来状態 s^{(future)} (0/1)
    """
    # 学習者 × スキル 全要素平均
    return float(np.mean((pred - target) ** 2))
