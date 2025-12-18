# src/training/bcsoftmax_trainer.py
import torch
import torch.nn.functional as F

from src.training.builders import build_loss, build_optimizer
from src.models.model_manager import ModelManager, ModelConfig

def compute_loss(outputs, targets, model, criterion, reg_cfg: dict | None):
    base_loss = criterion(outputs, targets)

    if not reg_cfg or reg_cfg.get("type") in (None, "None", "none"):
        return base_loss

    reg_type = reg_cfg.get("type")
    lam = float(reg_cfg.get("lambda", 0.0))

    if reg_type == "L1":
        reg_loss = lam * sum(p.abs().sum() for p in model.parameters())
    elif reg_type == "L2":
        reg_loss = lam * sum(p.pow(2).sum() for p in model.parameters())
    else:
        reg_loss = 0.0

    return base_loss + reg_loss

def model_predict(model, Xi, target, device):
    n_skills = len(Xi)
    constraints = torch.ones(n_skills, dtype=torch.float32, device=device) - Xi
    current_state = Xi.clone().to(device)

    target_sum = int(torch.clamp((target - Xi).sum(), min=0).item())

    for _ in range(target_sum):
        prob = constraints if constraints.sum() <= 1 else model(current_state, constraints)
        current_state = torch.clamp(current_state + prob, max=1.0)
        constraints = F.relu(constraints - prob)

    return current_state

def forward_predict(model, X, y, device):
    outputs = [model_predict(model, X[i], y[i], device) for i in range(len(y))]
    return torch.stack(outputs)

def forward_predict_proba(model, Xi, device):
    n_skills = len(Xi)
    constraints = torch.ones(n_skills, dtype=torch.float32, device=device) - Xi
    current_state = Xi.clone().to(device)
    prob = constraints if constraints.sum() <= 1 else model(current_state, constraints)
    return prob


def train_one_epoch(model, X_train, y_train, criterion, optimizer, device, reg_cfg):
    model.train()
    outputs = forward_predict(model, X_train, y_train, device)
    loss = compute_loss(outputs, y_train, model, criterion, reg_cfg)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def val_one_epoch(model, X_val, y_val, criterion, device, reg_cfg):
    model.eval()
    outputs = forward_predict(model, X_val, y_val, device)
    loss = compute_loss(outputs, y_val, model, criterion, reg_cfg)
    return float(loss.item())

def fit_bcsoftmax_model(
    *,
    model,
    method_name: str,
    seed: int,
    train_data: dict,          # {"X_train","y_train","X_val","y_val"} torch.Tensor
    device,
    exp_cfg: dict,
    training_cfg: dict,        # epochs, early_stopping, save_folder...
    recipe: dict               # optimizer/loss/reg...
):
    # 保存設定
    mm = ModelManager(ModelConfig(
        save_folder=training_cfg["save_folder"],
        save_best_only=bool(training_cfg.get("save_best_only", True)),
        checkpoint_freq=int(training_cfg.get("checkpoint_freq", 100)),
    ))

    save_path = mm.get_save_path(exp_cfg, method_name, seed)

    # 既存モデルがあればスキップ（あなたの挙動を維持）
    if mm.load_model_if_exists(save_path, model):
        return model

    # recipe から loss/optimizer を構築
    criterion = build_loss(recipe["loss"])
    optimizer = build_optimizer(model, recipe)
    reg_cfg = recipe.get("regularization", None)

    epochs = int(recipe["epochs"])
    early_stopping = int(recipe["early_stopping"])

    X_train = train_data["X_train"].to(device)
    y_train = train_data["y_train"].to(device)
    X_val   = train_data["X_val"].to(device)
    y_val   = train_data["y_val"].to(device)

    best = float("inf")
    patience = 0

    for epoch in range(epochs):
        tr = train_one_epoch(model, X_train, y_train, criterion, optimizer, device, reg_cfg)
        va = val_one_epoch(model, X_val, y_val, criterion, device, reg_cfg)

        if va < best:
            best = va
            patience = 0
            mm.save_best_model(model, save_path, epoch, va)
        else:
            patience += 1

        if (epoch + 1) % int(training_cfg["checkpoint_freq"]) == 0:
            print(f"[{method_name}] epoch={epoch+1} train={tr:.4f} val={va:.4f}")

        if patience >= early_stopping:
            print(f"[{method_name}] early stop at epoch={epoch+1}, best_val={best:.4f}")
            break

    return model
