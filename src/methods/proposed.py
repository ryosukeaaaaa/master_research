# src/methods/proposed_bcsoftmax.py
from __future__ import annotations
from typing import Any, Dict
import torch
import numpy as np

from .base import BaseMethod
from src.models.proposed import ProposedModel
from src.training.bcsoftmax_trainer import fit_bcsoftmax_model, forward_predict, forward_predict_proba

# src/methods/proposed.py

class ProposedBCSoftmaxMethod(BaseMethod):
    def __init__(
        self,
        *,
        method_name: str, # methodの名前
        n_skills: int,      # スキル数
        device: torch.device, # デバイス
        seed: int,         # 乱数シード
        exp_cfg: dict,      # 実験設定
        training_cfg: dict,   # 学習設定
    ):
        self.method_name = method_name
        self.device = device
        self.seed = seed
        self.exp_cfg = exp_cfg
        self.training_cfg = training_cfg["training_cfg"]
        self.recipe = training_cfg["recipe"]

        self.model = ProposedModel(n_skills=n_skills).to(device)

    def fit(self, train_data: Dict[str, Any]) -> None:
        self.model = fit_bcsoftmax_model(
            model=self.model,
            method_name=self.method_name,
            seed=self.seed,
            train_data=train_data,
            device=self.device,
            exp_cfg=self.exp_cfg,
            training_cfg=self.training_cfg,
            recipe=self.recipe,
        )

    @torch.no_grad()
    def predict(self, test_data: Dict[str, Any]) -> Dict[str, Any]:
        X = test_data["X_test"].to(self.device)
        y = test_data["y_test"].to(self.device)
        y_pred = forward_predict(self.model, X, y, self.device)  # (n_students, n_skills)

        return y_pred.detach().cpu().numpy()  # 状態の予測

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray) -> Dict[str, Any]:
        X = torch.tensor(X, dtype=torch.float32, device=self.device) 
        y_proba = forward_predict_proba(self.model, X, self.device)  # (n_skills, )

        return y_proba.detach().cpu().numpy() # 各スキルの習得確率
