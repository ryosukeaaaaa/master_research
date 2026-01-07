# 修士論文用リポジトリ
## 人工データ
1. 実行
```
python3 -m scripts.synthetic.run_experiment
```
2. 結果分析
```
python3 analysis/synthetic/parameter_analysis.py
```
3. 結果表示
数値結果：synthetic_results_analysis.ipynb
パラメータ分析：parameter_analysis.ipynb
パラメータ全体分析：comprehensive_metric_analysis.ipynb

## 実データデータ
### 対象スキルのみの時系列を前半と全体に分割してデータセットを作成


### 対象スキルのみの時系列を前後半に分割してデータセットを作成
#### Greedy algorithmを用いて、カバレッジが多いスキルを選定
1. スキルセット選定
run_skillset_selection.py
2. データ抽出と時間分割
ファイル: extract_selected_data.py
3. DINA モデルによるスキル状態推定
ファイル: estimate_skill_states.py

#### 特定のスキルのみを対象にデータセット作成
1. create_manual_skillset.py (手動スキル選択)
```
python3 scripts/assistments_2009_2010/create_manual_skillset.py \
  --skills 70  77 280                                        
```
出力: config_s70_s77_s280.json、valid_users_s70_s77_s280.csv

2. extract_selected_data.py (このコード - データ抽出)
   ```
   python3 scripts/assistments_2009_2010/extract_selected_data.py \ 
     --config_name config_s48_s77_s79_s276_s280 \
     --output_dir data/processed/assistments_2009_2010/selected_data/s48_s77_s79_s276_s280
   ```
出力: filtered_data.csv、first_half_data.csv / second_half_data.csv、response_matrix_*.csv

3. estimate_skill_states.py (DINA推定)
   ```
   python3 scripts/assistments_2009_2010/estimate_skill_states.py \ 
     --data_dir data/processed/assistments_2009_2010/selected_data/s48_s77_s79_s276_s280 \
     --K 5
   ```
出力: skill_states.csv

### 全体的な懸念点
データログは問題提示順であり、学習順とは限らない。
