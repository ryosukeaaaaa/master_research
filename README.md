project/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/                # 実験設定（YAML/JSON）
│   ├── default.yaml
│   ├── exp001.yaml
│   └── exp002.yaml
│
├── src/                    # コード本体（import前提）
│   ├── __init__.py
│   │
│   ├── models/             # モデル定義
│   │   ├── base.py
│   │   ├── proposed.py
│   │   └── baselines.py
│   │
│   ├── datasets/           # データ処理系
│   │   ├── synthetic.py    # 人工データ生成
│   │   ├── preprocess.py   # 実データ前処理
│   │   └── split.py        # train/val/test 分割
│   │
│   ├── training/           # 学習関連
│   │   ├── trainer.py
│   │   ├── loss.py
│   │   └── optimizer.py
│   │
│   ├── evaluation/         # 評価指標・評価処理
│   │   ├── metrics.py
│   │   └── evaluate.py
│   │
│   ├── experiments/        # 実験ロジック
│   │   ├── run_experiment.py
│   │   └── ablation.py
│   │
│   └── utils/              # 汎用関数
│       ├── seed.py
│       ├── logger.py
│       └── io.py
│
├── data/                   # データ保存（git管理外が多い）
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── outputs/                # 実験結果
│   ├── logs/
│   ├── figures/
│   └── metrics/
│
├── checkpoints/            # 学習済みモデル
│   └── exp001/
│
└── scripts/                # 実行用スクリプト
    ├── train.py
    ├── evaluate.py
    └── run_all.sh
