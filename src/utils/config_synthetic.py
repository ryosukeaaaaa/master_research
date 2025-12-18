import yaml

# パラメータ読み込み
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# grid展開
import itertools
def generate_grid(grid_cfg: dict):
    keys = list(grid_cfg.keys())
    values = list(grid_cfg.values())

    for v in itertools.product(*values):
        yield dict(zip(keys, v))

# # 結合
# from copy import deepcopy
# def build_experiment_configs(base_cfg, grid_cfg):
#     for grid in generate_grid(grid_cfg):
#         cfg = deepcopy(base_cfg)
#         cfg["synthetic"] = grid
#         yield cfg
