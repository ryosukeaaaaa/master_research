# モデルの保存・管理ファイル

from dataclasses import dataclass
from pathlib import Path
import torch
import json

@dataclass
class ModelConfig:
    """モデル設定クラス"""
    save_folder: str = "./simu_model"
    save_best_only: bool = True
    checkpoint_freq: int = 100

class ModelManager:
    """モデル管理クラス"""
    
    def __init__(self, cfg: ModelConfig):
        self.config = cfg
        Path(cfg.save_folder).mkdir(parents=True, exist_ok=True)
    
    # src/models/model_manager.py の修正
    def get_save_path(self, exp_cfg, method_name, seed, recipe=None):
        """保存パスを生成"""
        config_str = f"{exp_cfg['n_students']}_{exp_cfg['n_skills']}"
        
        # オプションパラメータを追加（存在する場合のみ）
        if 'smoothing' in exp_cfg:
            config_str += f"_{exp_cfg['smoothing']}"
        if 'current_alpha_beta' in exp_cfg:
            config_str += f"_{exp_cfg['current_alpha_beta'][0]}_{exp_cfg['current_alpha_beta'][1]}"
        if 'future_alpha_beta' in exp_cfg:
            config_str += f"_{exp_cfg['future_alpha_beta'][0]}_{exp_cfg['future_alpha_beta'][1]}"
        if recipe and 'regularization' in recipe:
            reg = recipe['regularization']
            config_str += f"_{reg['type']}_{reg['lambda']}"
        config_str += f"_{method_name}_{seed}"
        return Path(self.config.save_folder) / f"{config_str}.pt"
    
    def save_model(self, model, save_path, epoch, loss):
        """モデルを保存（.ptファイル）"""
        torch.save({
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'loss': loss
        }, save_path)
    
    def save_best_model(self, model, save_path, epoch, val_loss):
        """
        ベストモデルを保存（.ptファイル + best.pth + メタ情報）
        
        Args:
            model: 保存するモデル
            save_path: メインの保存パス (.pt)
            epoch: エポック数
            val_loss: 検証損失
        """
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # メインの .pt ファイルに保存
        self.save_model(model, save_path, epoch, val_loss)
        
        # best.pth にも保存（state_dict のみ）
        torch.save(model.state_dict(), save_dir / "best.pth")
        
        # メタ情報を保存
        with open(save_dir / "best_meta.json", "w") as f:
            json.dump({"epoch": epoch, "val_loss": float(val_loss)}, f)
    
    def load_model_if_exists(self, save_path, model):
        """
        既存モデルが存在すればロードする
        
        Args:
            save_path: モデルファイルのパス (.pt)
            model: ロード先のモデル
            
        Returns:
            bool: ロードに成功した場合True
        """
        save_path = Path(save_path)
        if not save_path.exists():
            return False
        
        print(f"[Skip] Found existing model file: {save_path}")
        
        try:
            checkpoint = torch.load(save_path, map_location="cpu")
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # チェックポイント形式
                model.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                # 直接state_dict
                model.load_state_dict(checkpoint)
            else:
                # その他の形式
                print(f"[Warning] Unexpected checkpoint format, attempting direct load...")
                model.load_state_dict(checkpoint)
            
            return True
            
        except Exception as e:
            print(f"[Warning] Failed to load existing model: {e}")
            print(f"[Info] Will proceed with training...")
            return False