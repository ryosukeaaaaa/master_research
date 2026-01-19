# 修士論文用リポジトリ
## 人工データ
1. 実行
```
python3 -m scripts.synthetic.run_experiment
```
2. 結果分析
```
python3 analysis/synthetic/run_analysis.py
```
3. パラメータ分析
```
python3 analysis/synthetic/parameter_analysis.py
```
4. 結果表示
数値結果：synthetic_results_analysis.ipynb
パラメータ分析：parameter_analysis.ipynb
パラメータ全体分析：comprehensive_metric_analysis.ipynb

## 実データデータ
### 対象スキルのみの時系列を前半と全体に分割してデータセットを作成
1. 実行
```
python3 -m scripts.assistments_2009_2010.run_experiment
```
2. 結果分析
```
python3 analysis/assistments_2009_2010/run_analysis.py
```
```
python analysis/assistments_2009_2010/run_analysis.py \
    --input-dir outputs/assistments_2009_2010/first_all_dina_estimation/results_L1_0.0001 \
    --output-dir analysis/assistments_2009_2010/s47_s49_s50_s58_s67_s70_s74_s77_s79_s86_s277_s278_s279_s280_s309_L1_0.001 \
    --input-files s47_s49_s50_s58_s67_s70_s74_s77_s79_s86_s277_s278_s279_s280_s309_L1_0.001.csv
```
3. 結果表示
```
assistments_2009_2010_results_analysis.ipynb
```

### 対象スキルのみの時系列を前後半に分割してデータセットを作成
#### Greedy algorithmを用いて、カバレッジが多いスキルを選定
```
python3 scripts/assistments_2009_2010/first_all_skillstate_estimation.py --skills 47 49 50 58 67 70 74 77 79 86 277 278 279 280 309
```
でスキル状態を推定
first_all_skillstate_estimate.ipynbで習得人数などを確認
skill_possibility.ipynbで加えて共起しやすいスキルを探す
eda_rawdata.ipynbで各スキルの名称や学習者の人数を確認
demo.ipynbで習得人数の割合やデモによるモデルの精度を確認

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



1/9
方針
人数上位１５スキルで実験
同じユーザーで５、１０スキルで実験　勝っていれば嬉しい
これによって同じCDMで検証できる
上手くいかなかったらCDMの適用方法（データの切り分け方）を変えてみる。2回チャンスがある

もし上手くいかなかったら、やっていないスキルが多少あってもCDM適用してみる

LLMにクラスタリングさせて検証