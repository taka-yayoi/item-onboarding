# Databricks notebook source
# MAGIC %md
# MAGIC # モデルウェイトのダウンロード
# MAGIC
# MAGIC 私たちは、オープンソースのLLAMAモデルを使用します。これらは、A100 GPUに快適に収まり、優れたパフォーマンスを持ち、簡単に仕事をこなすことができます。
# MAGIC
# MAGIC プロジェクトで活用する2つのLLAMAモデルは次のとおりです：
# MAGIC
# MAGIC - [LLAMA 3.2 11B Vision Model](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
# MAGIC - [LLAMA 3.1 8B Instruct Model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
# MAGIC
# MAGIC ビジョンモデルはアイテムの画像から情報を抽出するために使用され、インストラクトモデルはテキストベースのクエリに使用されます。
# MAGIC
# MAGIC ここでモデルウェイトをダウンロードする理由は、そうしないと、このワークフローを実行するたびにウェイトをダウンロードする必要があるためです。ダウンロードには数時間かかるわけではありませんが、毎回5〜10分を節約できれば、長期的には大きな節約になります。既存の場所からモデルウェイトをロードする方が効率的です。
# MAGIC
# MAGIC モデルをダウンロードするためにHuggingFaceを使用します。LLAMAモデルはウェブサイトでの簡単な登録が必要です。登録が完了したら、[トークンを生成](https://huggingface.co/settings/tokens)し、それをワークフローの残りの部分で使用します。
# MAGIC
# MAGIC huggingfaceパッケージはすでにランタイムにインストールされているため、再インストールする必要はありません。
# MAGIC
# MAGIC データ準備ノートブックと同様に、ここではシングルノードコンピュートを使用できます。この時点ではクラスターやGPUは必要ありません。`4 CPU`と`32 GB RAMメモリ`を持ち、`15.4 ML LTS`ランタイムを実行しているマシンで十分です。ただし、**MLランタイムは必要です**。

# COMMAND ----------

# MAGIC %md
# MAGIC ### コンテナの作成
# MAGIC
# MAGIC データ準備段階で行ったように、モデルウェイトを保存するためのボリュームロケーションを作成します。この場合、ロケーションをmodelsと呼びます。

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use this catalog by default
# MAGIC USE CATALOG takaakiyayoi_catalog;
# MAGIC
# MAGIC -- Use this schema by default
# MAGIC USE SCHEMA item_onboarding;
# MAGIC
# MAGIC -- Create a volume if it doesnt exist
# MAGIC CREATE VOLUME IF NOT EXISTS models;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 画像モデルのダウンロード
# MAGIC
# MAGIC 次に、シェルスクリプトを作成して画像モデルのウェイトをダウンロードします。ダウンロードの進行状況を追跡したい場合は、--quietフラグを削除できます。

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # HFトークンをエクスポート
# MAGIC export HF_TOKEN="HFトークン"
# MAGIC
# MAGIC # ダウンロードコマンドを実行
# MAGIC huggingface-cli \
# MAGIC   download \
# MAGIC   "meta-llama/Llama-3.2-11B-Vision-Instruct" \
# MAGIC   --local-dir "/Volumes/takaakiyayoi_catalog/item_onboarding/models/llama-32-11b-vision-instruct" \
# MAGIC   --exclude "original/*" \ # このフォルダ内の統合された重みは必要ありません
# MAGIC   --quiet # モデルのダウンロード進行状況を追跡したい場合はこれを削除

# COMMAND ----------

# MAGIC %md
# MAGIC ### テキストモデルのダウンロード
# MAGIC
# MAGIC 画像モデルと同様に、テキストモデルのウェイトもダウンロードします。

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # HFトークンをエクスポート
# MAGIC export HF_TOKEN="HFトークン"
# MAGIC
# MAGIC # ダウンロードコマンドを実行
# MAGIC huggingface-cli \
# MAGIC   download \
# MAGIC   "meta-llama/Meta-Llama-3.1-8B-Instruct" \
# MAGIC   --local-dir "/Volumes/takaakiyayoi_catalog/item_onboarding/models/llama-31-8b-instruct" \
# MAGIC   --exclude "original/*" \ # このフォルダ内の統合された重みは必要ありません
# MAGIC   --quiet # モデルのダウンロード進行状況を追跡したい場合はこれを削除

# COMMAND ----------

# MAGIC %md
# MAGIC この時点でモデルのウェイトは準備完了です。両方のダウンロードには、インターネット接続によっては最大30分かかる場合があります。
