# src/methods/popularity.py
from typing import Any, Dict
import numpy as np
import torch
from typing import Union

from .base import BaseMethod
from src.utils.redistribute import redistribute  # 既存関数を想定

class PopularityMethod(BaseMethod):
    def __init__(
        self,
        *,
        method_name: str,
        device: torch.device,
        seed: int,
    ):
        self.method_name = method_name
        self.device = device
        self.seed = seed

        # PopulationBase では model 不要
        self.population = None  # c_j を格納

    def fit(self, train_data: Dict[str, Any]) -> None:
        """
        train_data["X_train"]: (n_students, n_skills)
        """
        X = train_data["X_train"]

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        # c_j = 各スキルの習得者数
        self.population = X.sum(axis=0).astype(float)

    @torch.no_grad()
    def predict_proba(
        self,
        state: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """
        state: (n_skills,) current state (0/1)

        Returns:
            np.ndarray, shape=(n_skills,)
            新規習得確率分布
        """
        # torch -> numpy
        state = (
            state.detach().cpu().numpy()
            if isinstance(state, torch.Tensor)
            else state
        )

        mask = (state == 0)
        proba = np.zeros_like(state, dtype=float)

        if not mask.any():
            return proba  # 全習得済み

        weights = self.population * mask
        s = weights.sum()

        if s > 0:
            proba = weights / s
        else:
            # フォールバック：未習得で一様
            proba[mask] = 1.0 / mask.sum()

        return proba

    @torch.no_grad()
    def predict(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        test_data:
        - X_test: (n_students, n_skills) current state
        - y_test: (n_students, n_skills) next state (GT)
        """
        X = test_data["X_test"]
        y = test_data["y_test"]

        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        preds = []

        for i in range(X_np.shape[0]):
            state = X_np[i]

            # 1-step 遷移分布（1 次元）
            p = self.predict_proba(state)  # (n_skills,)

            # 次状態で増えたスキル数
            delta = y_np[i] - state
            total = int(np.round(delta.sum()))
            total = max(total, 0)  # 安全策

            pred_state = redistribute(
                state=state,
                p=p,
                total=total,
            )
            preds.append(pred_state)

        preds = np.stack(preds, axis=0)

        return preds  # (n_students, n_skills)
