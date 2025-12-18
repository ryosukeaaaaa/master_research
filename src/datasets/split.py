from collections import defaultdict
import numpy as np
import torch

def split_balanced_data(dataset, ratio=0.7, seed: int | None = None,
                        device: str | torch.device = "cpu",
                        x_dtype=torch.float32, y_dtype=torch.float32):
    """
    dataset: numpy [[x, y], ...] で、x,y は同形状（stack可能）を想定 (学生数, 遷移前・遷移後(2), スキル数)
    返り値: X_train, y_train, X_val, y_val を torch.Tensor で返す (学生数, スキル数)
    """
    # 空のデータセット処理（numpy配列やリスト両方に対応）
    if len(dataset) == 0:
        empty_tensor = torch.empty((0, 0), dtype=x_dtype, device=device)
        return empty_tensor, empty_tensor.clone(), empty_tensor.clone(), empty_tensor.clone()
    
    if isinstance(dataset, np.ndarray):
        dataset = dataset.copy()

    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(dataset)
    else:
        np.random.shuffle(dataset)

    data_per_correct_count = defaultdict(list)
    for x, y in dataset:
        correct_count = int(np.sum(y))
        data_per_correct_count[correct_count].append((x, y))

    X_train, y_train, X_val, y_val = [], [], [], []
    for _, data in data_per_correct_count.items():
        split_index = int(len(data) * ratio)
        train_data = data[:split_index]
        val_data = data[split_index:]

        X_train.extend([x for x, _ in train_data])
        y_train.extend([y for _, y in train_data])
        X_val.extend([x for x, _ in val_data])
        y_val.extend([y for _, y in val_data])

    # 空のリストのハンドリング付きでテンソル変換
    def to_tensor(data_list, dtype):
        if len(data_list) == 0:
            sample_shape = dataset[0][0].shape
            return torch.empty((0,) + sample_shape, dtype=dtype, device=device)
        return torch.as_tensor(np.stack(data_list), dtype=dtype, device=device)

    return (to_tensor(X_train, x_dtype), to_tensor(y_train, y_dtype),
            to_tensor(X_val, x_dtype), to_tensor(y_val, y_dtype))