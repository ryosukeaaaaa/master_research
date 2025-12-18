from src.evaluation.metrics import new_skill_accuracy, acquisition_cross_entropy, skill_wise_probabilistic_accuracy, skill_state_mse
from collections import defaultdict
import itertools
import numpy as np
from src.datasets.generate_syntheticdata import skill_acquisition_probabilities
from src.evaluation.metrics import kl_divergence, hellinger_distance, js_divergence

def evaluate_testdata(pred, input, target, metrics):
    scores = {}

    if "acc" in metrics:
        scores["acc"] = new_skill_accuracy(pred, input, target)

    if "ace" in metrics:
        scores["ace"] = acquisition_cross_entropy(pred, input, target)

    if "spa" in metrics:
        scores["spa"] = skill_wise_probabilistic_accuracy(pred, input, target)

    if "mse" in metrics:
        scores["mse"] = skill_state_mse(pred, target)

    return scores


def evaluate_distribution(A, model, metrics, smoothing=1e-6):
    """
    遷移分布レベルの評価（状態到達確率で重み付け）

    A: np.ndarray, shape=(n_skills, n_skills)
       依存関係行列（論文の A）
    model: method object
       状態 s を入力として遷移分布 q(.|s) を返す
    metrics: list[str]
       ["kl", "hd", "jsd"] など
    smoothing: float
       真の遷移分布計算時のスムージング
    """
    n_skills = A.shape[0]

    # 各状態の存在確率 Pr(s)
    node_prob = defaultdict(float)
    init_state = tuple(0 for _ in range(n_skills))
    node_prob[init_state] = 1.0

    # 評価結果
    results = {m: 0.0 for m in metrics}

    # 状態列挙（習得数昇順）
    states = sorted(
        itertools.product([0, 1], repeat=n_skills),
        key=sum
    )

    for state in states:
        if state == (1,) * n_skills:
            continue

        weight = node_prob[state]
        if weight == 0.0:
            continue

        state_arr = np.array(state)

        # 真の 1-step 遷移分布 p(.|s)
        true_probs = skill_acquisition_probabilities(
            A,
            state_arr,
            smoothing
        )

        # 次状態の存在確率更新
        for i in range(n_skills):
            if state[i] == 0:
                next_state = list(state)
                next_state[i] = 1
                node_prob[tuple(next_state)] += weight * true_probs[i]

        # モデルの予測遷移分布 q(.|s)
        pred_probs = model.predict_proba(state_arr)

        # 正規化チェック
        s = np.sum(pred_probs)
        if not np.isclose(s, 1.0):
            if s > 0:
                pred_probs = pred_probs / s

        # 指標計算（存在確率で重み付け）
        if "kl" in metrics:
            results["kl"] += weight * kl_divergence(true_probs, pred_probs)

        if "hd" in metrics:
            results["hd"] += weight * hellinger_distance(true_probs, pred_probs)

        if "jsd" in metrics:
            results["jsd"] += weight * js_divergence(true_probs, pred_probs)

    return results