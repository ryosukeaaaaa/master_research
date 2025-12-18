# src/training/builders.py
import torch
import torch.nn as nn

def build_loss(loss_name: str):
    name = loss_name.lower()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unknown loss: {loss_name}")

def build_optimizer(model, recipe: dict):
    opt = recipe["optimizer"].lower()
    params = recipe.get("optimizer_params", {})

    if opt == "adam":
        return torch.optim.Adam(model.parameters(), **params)
    if opt == "sgd":
        return torch.optim.SGD(model.parameters(), **params)
    raise ValueError(f"Unknown optimizer: {recipe['optimizer']}")
