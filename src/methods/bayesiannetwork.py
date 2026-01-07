import numpy as np
import torch
from typing import Dict, Any, Union

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import HillClimbSearch, BayesianEstimator, BIC
from pgmpy.inference import VariableElimination

from .base import BaseMethod
from src.utils.redistribute import redistribute


class BayesianNetworkMethod(BaseMethod):
    def __init__(
        self,
        *,
        method_name: str,
        device: torch.device,
        seed: int,
        max_parents: int = 3,
    ):
        self.method_name = method_name
        self.device = device
        self.seed = seed
        self.max_parents = max_parents

        self.model: DiscreteBayesianNetwork | None = None
        self.infer: VariableElimination | None = None
        self.skill_names = None

    # ------------------------------------------------------------------
    # FIT
    # ------------------------------------------------------------------
    def fit(self, train_data: Dict[str, Any]) -> None:
        """
        train_data["current_data"]: (n_students, n_skills)
        """
        import logging
        logging.getLogger("pgmpy").setLevel(logging.WARNING)
        
        X = train_data["current_data"]
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        n_students, n_skills = X.shape
        self.skill_names = [f"s{i}" for i in range(n_skills)]

        import pandas as pd
        df = pd.DataFrame(X, columns=self.skill_names).astype(int)

        # --- 構造学習（HC + BIC） ---
        hc = HillClimbSearch(df)
        best_model = hc.estimate(
            scoring_method=BIC(df),
            max_indegree=self.max_parents,
        )

        # --- 離散BN ---
        self.model = DiscreteBayesianNetwork()
        self.model.add_nodes_from(self.skill_names)
        self.model.add_edges_from(best_model.edges())

        # --- CPT 推定 ---
        self.model.fit(
            df,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=1.0,
        )

        # ★ CPD 補完（孤立ノード対策）
        from pgmpy.factors.discrete import TabularCPD

        for node in self.model.nodes():
            if self.model.get_cpds(node) is None:
                p = df[node].mean()
                p = float(np.clip(p, 1e-6, 1 - 1e-6))
                cpd = TabularCPD(
                    variable=node,
                    variable_card=2,
                    values=[[1 - p], [p]],
                )
                self.model.add_cpds(cpd)

        # --- 推論エンジン ---
        self.infer = VariableElimination(self.model)


    # ------------------------------------------------------------------
    # PREDICT PROBA
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_proba(
        self,
        state: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        state: (n_skills,) current state (0/1)

        Returns:
            np.ndarray, shape=(n_skills,)
            「次に1つ習得する」確率分布
        """
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()

        n_skills = state.shape[0]
        proba = np.zeros(n_skills, dtype=float)

        # 全習得済み
        if (state == 1).all():
            return proba

        # 各未習得スキルについて P(s_i=1 | s_-i)
        for i in range(n_skills):
            if state[i] == 1:
                continue

            # ★ i 番目のスキルは evidence から除外
            evidence = {
                self.skill_names[j]: int(state[j])
                for j in range(n_skills)
                if j != i
            }

            q = self.infer.query(
                variables=[self.skill_names[i]],
                evidence=evidence,
                show_progress=False,
            )

            # binary variable → index 1 が「習得」
            proba[i] = float(q.values[1])

        # 未習得スキル上で正規化
        s = proba.sum()
        if s > 0:
            proba /= s
        else:
            mask = (state == 0)
            proba[mask] = 1.0 / mask.sum()

        return proba

    # ------------------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------------------
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

            # 1-step 習得分布
            p = self.predict_proba(state)

            # delta = 習得スキル数
            delta = y_np[i] - state
            total = int(np.round(delta.sum()))
            total = max(total, 0)

            pred_state = redistribute(
                state=state,
                p=p,
                total=total,
            )
            preds.append(pred_state)

        preds = np.stack(preds, axis=0)

        return preds



# # skill snapshots : DataFrame with columns = skill names (0/1)
# data = snapshots_df  

# # --- 1) 構造学習（HC/PCなど選択可） ---
# hc = HillClimbSearch(data, scoring_method=BicScore(data))
# best_model = hc.estimate()

# model = BayesianModel(best_model.edges())
# model.fit(data, estimator=MaximumLikelihoodEstimator)

# # --- 2) P(s|current,k)の生成 ---
# def predict_future_prob(current_state, k):
#     candidates = []
#     skills = list(data.columns)
#     for comb in itertools.combinations([s for s in skills if current_state[s]==0], k):
#         future = current_state.copy()
#         for s in comb: future[s]=1
#         prob = model.predict_probability(future).values[0]
#         candidates.append((future,prob))
#     Z=sum(p for _,p in candidates)
#     return [(f,p/Z) for f,p in candidates]  # normalized
