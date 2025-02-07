# Databricks notebook source
# MAGIC %md
# MAGIC # 結果
# MAGIC
# MAGIC 推論中にParquetを生成したので、結果をDeltaテーブルとして保存し、その後、結果を監視するためのシンプルなインターフェースを構築します。
# MAGIC
# MAGIC ここで推奨されるコンピュートは、最新のランタイムを備えたシングルノードのシンプルなマシンです。私たちは、`4 CPUコア`、`32 GBのRAMメモリ`、および`Runtime 15.4 LTS`を備えたマシンを使用しました。このノートブックではクラスターやGPUは必要ありません。

# COMMAND ----------

# MAGIC %md
# MAGIC ### 画像分析の保存
# MAGIC
# MAGIC 画像分析から始めましょう。ボリュームディレクトリに中間データフレームをParquetとして保存しています。Parquetファイルを選択し、Unity CatalogにDeltaとして保存できます。

# COMMAND ----------

# parquetファイルを読み込む
from pyspark.sql import functions as SF

image_analysis_df = spark.read.parquet(
    "/Volumes/takaakiyayoi_catalog/item_onboarding/interim_data/image_analysis"
)

image_analysis_df = image_analysis_df.drop("image")

pattern = r"assistant<\|end_header_id\|>\s*([\s\S]*?)<\|eot_id\|>"
image_analysis_df = (
    image_analysis_df
    .withColumn("gen_description", SF.regexp_extract("description", pattern, 1))
    .withColumn("gen_color", SF.regexp_extract("color", pattern, 1))
)

(
    image_analysis_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.image_analysis")
)

display(image_analysis_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### テキスト分析の保存
# MAGIC
# MAGIC テキスト分析部分でも同じプロセスを繰り返します。

# COMMAND ----------

# parquetファイルを読み込む
text_analysis_df = spark.read.parquet("/Volumes/takaakiyayoi_catalog/item_onboarding/interim_data/results")

# Deltaテーブルを保存する
(
    text_analysis_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.text_analysis")
)

display(text_analysis_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 結果インターフェース
# MAGIC
# MAGIC プロセスを簡単に監視できるインターフェースを構築しましょう。製品IDを選択し、提供されたデータ、画像モデルが見たもの、テキストモデルが構築したものを理解できるようにしたいと考えています。

# COMMAND ----------

text_analysis_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.text_analysis")

# 利用可能なすべてのIDを取得する
available_ids = [x[0] for x in text_analysis_df.select("item_id").distinct().collect()]

# 1つ選択する
index = 2
selected_id = available_ids[index]

# アイテムデータを取得する
item_data = text_analysis_df.filter(text_analysis_df.item_id == selected_id).collect()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 商品をチェック
# MAGIC
# MAGIC アイテムの画像を見ることから始めます。

# COMMAND ----------

from PIL import Image

print(f">>> 商品ID: {item_data['item_id']} の分析を開始します <<<\n")
print("商品の画像は以下の通りです:")
img = Image.open(item_data["real_path"])
display(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 仕入先からの情報
# MAGIC
# MAGIC この部分は、アイテムに関してデータサプライヤーが提供する情報を示しています。

# COMMAND ----------

print(f">>> 商品の説明: \n\n{item_data['bullet_point']}")
print(f"\n\n>>> 商品の色: \n\n{item_data['color']}")
print(f"\n\n>>> 商品のキーワード: \n\n{item_data['item_keywords']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 画像モデル分析
# MAGIC
# MAGIC 画像モデルが見たものは？

# COMMAND ----------

print(f">>> 画像に何が見えますか？: \n\n{item_data['gen_description']}")
print(f"\n\n>>> 製品の色は何色ですか？: \n\n{item_data['gen_color']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### テキストモデル分析
# MAGIC
# MAGIC テキストモデルが全データポイントを考慮して提案した内容は？

# COMMAND ----------

print(f">>> 推奨説明: \n\n{item_data['suggested_description'].strip()}")
print(f"\n\n>>> 推奨色: \n\n{item_data['suggested_color'].strip()}")
print(f"\n\n>>> 推奨キーワード: \n\n{item_data['suggested_keywords'].strip()}")
print(f"\n>>> 推奨カテゴリ: \n\n{item_data['suggested_category'].strip()}")

# COMMAND ----------


