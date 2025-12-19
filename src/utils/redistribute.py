# ルールベース手法の予測分布をスキル状態に変換
from __future__ import annotations
import numpy as np 

def redistribute(state: np.ndarray, p: np.ndarray, total: int) -> np.ndarray:
    """
    予測分布が不変であるモデルに対して、予測状態に変換する関数。
    分布 p の相対比率に従って、state が 0 の要素に合計 `total` を割り振る。
    各要素は上限 1.0 を超えない（cap）。重みが実質ゼロしか残らない場合は均等分配にフォールバックする。

    Args:
        state:  現在状態 (0/1) のベクトル。1 の位置は分配対象外。
        p:      分配の重み（各成分 >= 0）
        total:  追加で割り振る総量（非負の整数を想定）

    Returns:
        predicted_state: state を float に拡張して分配後のベクトル（各成分 ≤ 1.0）
    """
    predicted_state = state.astype(float).copy()
    remaining = float(total)
    if remaining <= 0.0:
        return predicted_state

    EPS = 1e-9           # ゼロ判定（数値誤差対策）
    SMALL_WEIGHT = 1e-12 # 極小重みはゼロとみなす
    MAX_ITERS = 10_000

    iters = 0
    while remaining > EPS and iters < MAX_ITERS:
        iters += 1

        # まだ増やせる位置（未習得かつ < 1）
        mask = (predicted_state < 1.0) & (state == 0)
        if not np.any(mask):
            break  # 分配先がない

        # 比率分配のための重み
        weights = (p * mask).astype(float)
        # 極小重みは 0 とみなす（s がダラダラと EPS を僅かに超えるのを防ぐ）
        weights[weights < SMALL_WEIGHT] = 0.0
        s = weights.sum()

        if s <= EPS:
            # --- 均等分配（フォールバック） ---
            idxs = np.where(mask)[0]
            caps = 1.0 - predicted_state[idxs]
            cap_sum = caps.sum()
            if cap_sum <= EPS:
                break  # 実質もう入らない

            portion = min(remaining, cap_sum)
            per = portion / len(idxs)
            inc = np.minimum(per, caps)

            gain = float(inc.sum())
            if gain <= EPS:
                break

            predicted_state[idxs] += inc
            remaining -= gain
        else:
            # --- 比率分配 ---
            weights /= s
            caps = 1.0 - predicted_state
            raw_allocation = weights * remaining
            allocation = np.minimum(raw_allocation, caps)  # cap=1.0 を考慮

            gain = float(allocation.sum())
            if gain <= EPS:
                # 進捗がないなら均等分配にフォールバック
                idxs = np.where(mask)[0]
                caps_i = 1.0 - predicted_state[idxs]
                cap_sum = caps_i.sum()
                if cap_sum <= EPS:
                    break
                portion = min(remaining, cap_sum)
                per = portion / len(idxs)
                inc = np.minimum(per, caps_i)

                gain_fb = float(inc.sum())
                if gain_fb <= EPS:
                    break

                predicted_state[idxs] += inc
                remaining -= gain_fb
            else:
                predicted_state += allocation
                remaining -= gain

    # ごく微小な負残量は 0 扱い（数値誤差）
    if remaining < 0.0 and abs(remaining) <= 1e-7:
        remaining = 0.0

    # 数値誤差で 1.0+ε などになっていないか最終クリップ
    np.clip(predicted_state, 0.0, 1.0, out=predicted_state)

    if sum(predicted_state - state) > total + EPS:
        raise RuntimeError("Redistribution exceeded total allocation.")
    return predicted_state