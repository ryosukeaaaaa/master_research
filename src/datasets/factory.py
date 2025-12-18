from src.datasets.split import split_balanced_data
import numpy as np
import torch

def build_dataset(method_cfg, current_dataset, future_dataset, device):
    if method_cfg["dataset_id"] == "current_data" and method_cfg["id"] == "proposed":
        X_train, y_train, X_val, y_val = split_balanced_data(current_dataset, method_cfg["split_ratio"], device=device)
        train_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            }
    elif method_cfg["dataset_id"] == "full_information_data" and method_cfg["id"] == "full_information":
        full_information_dataset = np.concatenate([current_dataset, future_dataset], axis=0)
        X_train, y_train, X_val, y_val = split_balanced_data(full_information_dataset, method_cfg["split_ratio"], device=device)
        train_data = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            }
    elif method_cfg["dataset_id"] == "current_data":
        X_train = torch.tensor(current_dataset[:, 0, :], device=device)
        y_train = torch.tensor(current_dataset[:, 1, :], device=device)
        train_data = {
            "X_train": X_train,
            "y_train": y_train,
            }
    else:
        raise ValueError(f"Unknown dataset_id: {method_cfg['dataset_id']}")
    return train_data

def build_test_dataset(test_dataset, device):
    """
    test_dataset: shape=(n_students, 2, n_skills)
        [:, 0, :] が X (現在の状態)
        [:, 1, :] が y (将来の状態)
    """
    X_np = test_dataset[:, 0, :].astype(np.float32)
    y_np = test_dataset[:, 1, :].astype(np.float32)
    
    test_data = {
        "X_test": torch.from_numpy(X_np).to(device),
        "y_test": torch.from_numpy(y_np).to(device),
    }
    return test_data