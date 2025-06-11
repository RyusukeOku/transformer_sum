# transformer_sum

このリポジトリは、Transformerモデルを用いて足し算を学習させるためのプロジェクトです。
様々なデータ形式や学習戦略を試し、モデルがどのように算術タスクを習得するかを実験・分析することを目的としています。

## 機能

* **多様なデータ生成**:
    * 標準的な足し算問題 (例: `123+456=`)
    * 交換法則を考慮した問題 (例: `A+B=` と `B+A=`)
    * 数を分解した形式の問題 (例: `(10+2)+(30+4)=`)
    * カリキュラム学習のための難易度別の問題セット
* **複数の学習アプローチ**:
    * 基本的な学習・評価パイプライン
    * 簡単な問題から徐々に難しくするカリキュラム学習
    * 数字と「桁位置」を組み合わせた特殊なトークン化手法
    * 学習率ウォームアップやラベルスムージングなどの高度な学習テクニック
* **詳細な実験管理**:
    * 学習ログ（損失、正解率）、学習済みモデル、予測結果の分析レポートを自動で保存します。
    * 複数のシード値で実験を繰り返し、結果の平均を評価する機能も含まれています。

## ファイル構成と役割

このリポジトリの主要なファイルとディレクトリの役割は以下の通りです。

### 主要スクリプト

* `transformer.py`: 中核となるSeq2Seq Transformerモデル（`Seq2SeqTransformer`）を定義しています。
* **データ生成スクリプト (`generate_*.py`)**:
    * `generate_data.py`: 3桁の数字同士の足し算データセットを生成します。
    * `generate_data_reversed.py`: 交換法則を考慮し、「A+B=」と「B+A=」の両方のペアを生成します。
    * `generate_data_splited.py`: 数字を「(10の位+1の位)」の形式に分解したデータセットを生成します。
    * `generate_data_reverse_splited.py`: 上記の「交換法則」と「分解形式」を組み合わせたデータセットを生成します。
    * `generate_data_curriculum.py`: カリキュラム学習用に、1桁の簡単な問題から2桁の複雑な問題まで、難易度別に複数のデータセットを生成します。
* **学習・評価スクリプト (`data_load_and_train_*.py`)**:
    * `data_load_and_train_and_eval.py`: 最も基本的な学習・評価スクリプトです。
    * `data_load_and_train_curriculum.py`: `generate_data_curriculum.py`で生成したデータセットを使い、カリキュラム学習を実行します。
    * `data_load_and_train_disit_pos.py`: 数字を「`7_pos0`（1の位の7）」のように、桁情報を持つトークンとして扱う手法で学習を行います。複数回の実験を自動で実行する機能も持ちます。
    * `data_load_and_train_warmup.py`: `disit_pos`と同様に桁位置エンコーディングを使いつつ、学習率ウォームアップなどのテクニックを導入した学習スクリプトです。

### ディレクトリ構造

スクリプトを実行すると、以下のディレクトリが自動的に作成され、成果物が保存されます。

* `data/`: `generate_*.py` によって生成された学習・テスト用のデータセット（`.csv`形式）が保存されます。
* `models/`: 学習済みのモデルの重み（`.pth`形式）が保存されます。
* `log/`: エポックごとの損失や正解率などの学習過程を記録したログファイル（`.csv`形式）が保存されます。
* `analysis/`: モデルがテストデータに対してどのような予測を行ったかを記録した、詳細な分析ファイル（`.csv`形式）が保存されます。

## 実行方法

### 1. 準備

まず、必要なライブラリをインストールします。

```bash
pip install torch numpy pandas
```

### 2. データセットの生成

次に、学習に使用するデータセットを生成します。目的に応じて、いずれかの`generate_*.py`スクリプトを実行してください。

**例：基本的な3桁の足し算データを生成する場合**
```bash
python generate_data.py
```
これにより、`data/`ディレクトリに`addition_train_dataset_3d.csv`と`addition_test_dataset_3d.csv`が作成されます。

**例：カリキュラム学習用のデータを生成する場合**
```bash
python generate_data_curriculum.py
```
これにより、`data/`ディレクトリに`curriculum_stage1_*.csv`から`curriculum_stage6_*.csv`まで、複数のファイルが作成されます。

### 3. モデルの学習と評価

生成したデータセットに対応する学習スクリプトを実行します。

**例：基本的な学習を行いたい場合**
（`generate_data.py`で生成したデータを使用）
```bash
python data_load_and_train_and_eval.py
```
学習が完了すると、`log/`にログが、ルートディレクトリにモデルファイル（`seq2seq_transformer_addition_best.pth`）が保存されます。

**例：カリキュラム学習を行いたい場合**
（`generate_data_curriculum.py`で生成したデータを使用）
```bash
python data_load_and_train_curriculum.py
```
学習が完了すると、`log/`、`models/`、`analysis/`にそれぞれの成果物が保存されます。

**例：桁位置エンコーディングを用いた学習を行いたい場合**
（`generate_data.py`などを実行して`data/addition_train_dataset_2d.csv`などを用意）
```bash
python data_load_and_train_disit_pos.py
```
設定された回数（デフォルトは10回）の実験が連続して実行され、成果物が`log/logs_2d`、`models/model_2d`、`analysis/analysis_2d`に実験ごとのファイルとして保存されます。
