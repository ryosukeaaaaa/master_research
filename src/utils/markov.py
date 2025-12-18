import numpy as np
from collections import defaultdict
from typing import Dict, Tuple


def build_node_count_dict(d_dataset: np.ndarray) -> Dict[Tuple[int, ...], int]:
    """
    d_dataset: shape (N, n_skills), 0/1
    戻り値: {(0/1, ..., 0/1): count}
    """
    d_dataset = np.asarray(d_dataset)
    if not np.issubdtype(d_dataset.dtype, np.integer):
        d_dataset = np.rint(d_dataset).astype(int)

    node_count = defaultdict(int)

    for row in d_dataset:
        key = tuple(int(x) for x in row)
        node_count[key] += 1

    return dict(node_count)


# ---次ノードをもとにした予測---
def SimpleMarkov_Prob(
    state: np.ndarray,
    node_count: Dict[Tuple[int, ...], int],
) -> np.ndarray:
    """
    Args:
        state: 現在スキル状態（0/1, float 可）
        node_count: build_node_count_dict の出力

    Returns:
        Node_pred: shape (n_skills,) の確率分布
    """
    state = np.asarray(state)
    if not np.issubdtype(state.dtype, np.integer):
        # redistribute 後の float を最近傍の 0/1 に丸める
        state = np.rint(state).astype(int)

    n_skills = state.size
    nonacquired = np.flatnonzero(state == 0)

    Node_pred = np.zeros(n_skills, dtype=float)

    for j in nonacquired:
        nxt = state.copy()
        nxt[j] = 1
        Node_pred[j] = node_count.get(tuple(nxt.tolist()), 0)

    total = Node_pred.sum()
    if total > 0:
        Node_pred /= total
    elif nonacquired.size > 0:
        # 次ノードに誰もいない場合は未習得スキルで一様分配
        Node_pred[nonacquired] = 1.0 / nonacquired.size

    return Node_pred