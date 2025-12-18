import numpy as np
import random
import torch

def set_seed(seed):
    random.seed(seed)  # Pythonのrandomモジュールのシードを固定
    np.random.seed(seed)  # NumPyのシードを固定
    torch.manual_seed(seed)  # PyTorchのCPUシードを固定
    torch.cuda.manual_seed(seed)  # PyTorchのGPUシードを固定
    torch.cuda.manual_seed_all(seed)  # マルチGPU環境でのシード固定
    torch.backends.cudnn.deterministic = True  # 再現性のためにCuDNNを固定
    torch.backends.cudnn.benchmark = False  # 再現性のためにCuDNNベンチマークをOFF

import hashlib

def derive_seed(seed_base: int, method_id: str) -> int:
    s = f"{seed_base}:{method_id}".encode()
    return int(hashlib.md5(s).hexdigest()[:8], 16)  # 0〜2^32-1
