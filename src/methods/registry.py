from __future__ import annotations
from typing import Any, Dict, Optional
import torch

from src.methods.proposed import ProposedBCSoftmaxMethod
from src.methods.popularity import PopularityMethod
from src.methods.simple_markov import SimpleMarkovMethod
from src.methods.random import RandomMethod


def build_method(
    method_cfg: dict,
    device: torch.device,
    seed: int,
    exp_cfg: Optional[Dict[str, Any]] = None,
    training_cfg: Optional[Dict[str, Any]] = None,
):
    """各手法に必要なパラメータだけを渡す"""
    t = method_cfg["id"]
    method_name = method_cfg["id"]

    if t == "proposed" or t == "full_information":
        if exp_cfg is None or training_cfg is None:
            raise ValueError(f"Method '{t}' requires exp_cfg and training_cfg")
        
        return ProposedBCSoftmaxMethod(
            method_name=method_name,
            n_skills=exp_cfg["n_skills"],
            device=device,
            seed=seed,
            exp_cfg=exp_cfg,
            training_cfg=training_cfg
        )
    
    elif t == "popularity":
        return PopularityMethod(
            method_name=method_name,
            device=device,
            seed=seed,
        )
    
    elif t == "simple_markov":
        return SimpleMarkovMethod(
            method_name=method_name,
            device=device,
            seed=seed,
        )
    
    elif t == "random":
        return RandomMethod(
            method_name=method_name,
            device=device,
            seed=seed,
        )

    raise ValueError(f"Unknown method type: {t}")