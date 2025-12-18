import torch

def get_device():
    """利用可能なデバイスを取得する"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     # MPS (Metal Performance Shaders) for Mac
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")
    return device