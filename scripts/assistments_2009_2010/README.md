
ASSISTments 2009-2010データセットから前後半のスキルセットを作成するまでの流れを説明します。

データ処理パイプライン
1. スキルセット選定
ファイル: run_skillset_selection.py

入力:

skill_builder_data.csv - ASSISTments生データ
処理:

Strategy B (Co-occurrence Maximization)を使用
require_temporal_coverage=Trueで前後半両方でカバレッジを保証
Greedy algorithmで指定数（K）のスキルを選定
選定されたスキルを持つ有効ユーザーを特定
出力 (skillset_selection):

config_min{MIN_USERS}_k{K}.json - 選定されたスキルID・名前・設定
valid_users_min{MIN_USERS}_k{K}.csv - 有効ユーザーIDリスト
実行例:

2. データ抽出と時間分割
ファイル: extract_selected_data.py

入力:

skill_builder_data.csv - 生データ
outputs/assistments_2009_2010/skillset_selection/config_min{MIN_USERS}_k{K}.json - スキル設定
outputs/assistments_2009_2010/skillset_selection/valid_users_min{MIN_USERS}_k{K}.csv - ユーザーリスト
処理:

選定されたスキルと有効ユーザーでデータをフィルタリング
各ユーザーの問題を時系列順（order_id）でソート
各ユーザーのデータを前半・後半に分割（中間点で分割）
前後半両方で全Kスキルに取り組んだユーザーのみを保持
各スキルの正解率を集約（aggregated_matrix）
出力 (data/processed/assistments_2009_2010/min{MIN_USERS}_k{K}/):

data_info.json - データセット情報（K, ユーザー数など）
filtered_data.csv - フィルタ済み全データ
first_half_data.csv - 前半期間の問題データ
second_half_data.csv - 後半期間の問題データ
aggregated_matrix_first.csv - 前半の正解率集約データ
aggregated_matrix_second.csv - 後半の正解率集約データ
user_statistics.csv - ユーザーごとの問題数統計
実行例:

3. DINA モデルによるスキル状態推定
ファイル: estimate_skill_states.py

入力:

data/processed/assistments_2009_2010/min{MIN_USERS}_k{K}/aggregated_matrix_first.csv
data/processed/assistments_2009_2010/min{MIN_USERS}_k{K}/aggregated_matrix_second.csv
data/processed/assistments_2009_2010/min{MIN_USERS}_k{K}/data_info.json
処理:

集約された正解率から仮想問題を生成（各スキル20問）
Q行列を作成（各スキルに20問を割り当て）
前半データでDINAモデルを推定（EMアルゴリズム）
後半データでDINAモデルを推定（独立に推定）
スキル状態の遷移を分析
出力 (outputs/assistments_2009_2010/dina_estimation/min{MIN_USERS}_k{K}/):

skill_states_first.csv - 前半のスキル習得状態（binary）
skill_states_second.csv - 後半のスキル習得状態（binary）
skill_transitions.csv - スキル状態遷移の詳細
estimation_summary.json - 習得率・遷移の集計結果
skill_mapping.json - スキルID・名前のマッピング
実行例:

データフォーマット詳細
aggregated_matrix_first.csv / aggregated_matrix_second.csv
skill_{ID}: そのスキルの正解率（0.0-1.0）
skill_{ID}_count: そのスキルで解いた問題数
skill_states_first.csv / skill_states_second.csv
0: スキル未習得
1: スキル習得済み
重要な注意点
時間分割: 各ユーザーの問題を時系列順に並べて中間点で分割
カバレッジ保証: 前後半両方で全Kスキルに取り組んだユーザーのみを使用
仮想問題生成: 実際の問題正誤ではなく、正解率から確率的に問題応答を生成
独立推定: 前半と後半のDINAモデルは独立に推定（同じQ行列を使用）