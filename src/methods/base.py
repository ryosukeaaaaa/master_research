from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseMethod(ABC):
    """NN/ルール/Oracle を統一して回すためのインターフェース"""

    def fit(self, train_data: Dict[str, Any]) -> None:
        """学習が必要な手法だけ実装すればOK（不要なら何もしない）"""
        return

    @abstractmethod
    def predict(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        必須。戻り値は辞書で統一:
          - pred: (N,) 予測ラベル
          - proba: (N, C) 予測確率（あれば）
          - extra: 任意
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        return

    def load(self, path: str) -> None:
        return
