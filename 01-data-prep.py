# Databricks notebook source
# MAGIC %md
# MAGIC # データ準備
# MAGIC
# MAGIC プロジェクトの開始にあたり、データの準備を行います。アイテムのオンボーディングシナリオでは、企業はさまざまな形式のデータを受け取ることがよくあります。最も一般的なケースの一部は、アイテムの色、説明、素材などのプロパティに関するデータを含むテキスト満載のCSVや、アイテムの写真です。
# MAGIC
# MAGIC 同様のシナリオをシミュレートできるように、類似のデータセットを探しました。[Amazon's Berkley Objects Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html)は、まさに私たちが探していたものです。これは、現実のシナリオで期待されるように、100%一貫していないアイテムに関するデータを特徴としています。また、製品に関する情報を抽出するためにビジョンモデルで使用できる画像も特徴としています。
# MAGIC
# MAGIC このノートブックでは、環境を準備し、データをダウンロードして解凍し、後の段階で使用できるように保存します。
# MAGIC
# MAGIC ここで推奨されるコンピュートは、最新のランタイムを備えたシングルノードのシンプルなマシンです。私たちは、`4 CPUコア`、`32 GBのRAMメモリ`、および`Runtime 15.4 LTS`を備えたマシンを使用しました。このノートブックではクラスターやGPUは必要ありません。

# COMMAND ----------

# MAGIC %md
# MAGIC ### コンテナの準備
# MAGIC
# MAGIC ここでは、[Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html)を活用して、カタログとしてのコンテナと、テーブルを保存するためのスキーマ（データベース）を作成します。
# MAGIC
# MAGIC また、このスキーマ内にファイルを保存するための[ボリューム](https://docs.databricks.com/en/sql/language-manual/sql-ref-volumes.html)を作成します。ボリュームは、CSVや画像のような実際のファイルを保存するのに適したハードドライブのようなストレージ場所と考えることができます。

# COMMAND ----------

# MAGIC %sql
# MAGIC -- このノートブックでは既存のカタログをデフォルトで使用します
# MAGIC USE CATALOG takaakiyayoi_catalog;
# MAGIC -- 新しいカタログが必要な場合: CREATE CATALOG IF NOT EXISTS xyz;
# MAGIC
# MAGIC -- テーブルを保持するためにそのカタログ内にスキーマを作成します
# MAGIC CREATE SCHEMA IF NOT EXISTS item_onboarding;
# MAGIC
# MAGIC -- このノートブックのすべての操作にデフォルトでこのスキーマを使用します
# MAGIC USE SCHEMA item_onboarding;
# MAGIC
# MAGIC -- ファイルを保持するためのボリュームを作成します
# MAGIC CREATE VOLUME IF NOT EXISTS landing_zone;

# COMMAND ----------

# MAGIC %md
# MAGIC ### データのダウンロード
# MAGIC
# MAGIC このセクションでは、シェルスクリプトを使用してデータをダウンロードし、先ほど作成したボリュームに保存します。

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 表形式データのダウンロード
# MAGIC
# MAGIC まず、表形式データから始めます。シェルスクリプトには、ダウンロードしたデータを解凍する部分も含まれています。これは、Sparkでデータを読み取る前に必要です。

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # ターゲットボリュームディレクトリに移動
# MAGIC cd /Volumes/takaakiyayoi_catalog/item_onboarding/landing_zone
# MAGIC
# MAGIC # リスティングファイルをダウンロード
# MAGIC echo "リスティングをダウンロード中"
# MAGIC wget -q https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
# MAGIC
# MAGIC # リスティングファイルを解凍
# MAGIC echo "リスティングを解凍中"
# MAGIC tar -xf ./abo-listings.tar --no-same-owner
# MAGIC gunzip ./listings/metadata/*.gz
# MAGIC
# MAGIC echo "完了"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 画像のダウンロード
# MAGIC
# MAGIC 画像のダウンロードは少し異なります。同じ手順の一部に従いますが、ボリュームへの移動部分が異なります。また、データを直接ボリュームにダウンロードするのではなく、ここではSparkドライバーの一時メモリを使用して操作を実行します。
# MAGIC
# MAGIC それは、多くの小さなファイル（画像など）がある場合、ボリュームの場所よりもメモリ内で解凍する方が速いためです。

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # 一時ディレクトリを作成
# MAGIC mkdir /tmp_landing_zone
# MAGIC
# MAGIC # ターゲットディレクトリに移動
# MAGIC cd /tmp_landing_zone
# MAGIC
# MAGIC # 画像ファイルをダウンロード
# MAGIC echo "画像をダウンロード中"
# MAGIC wget -q https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
# MAGIC
# MAGIC # 画像ファイルを解凍
# MAGIC # (imagesというフォルダに解凍される)
# MAGIC echo "画像を解凍中"
# MAGIC tar -xf ./abo-images-small.tar --no-same-owner
# MAGIC gzip -df ./images/metadata/images.csv.gz
# MAGIC
# MAGIC echo "完了"

# COMMAND ----------

# MAGIC %md
# MAGIC **画像コピーのトリック**
# MAGIC
# MAGIC 少数の大きなファイルを扱う場合、通常のDatabricksユーティリティを使用してファイルをコピーするのは非常に便利ですが、ここでのように多数の小さなファイルを扱う場合にはそれほど速くありません。これは、画像を扱うシナリオで発生することがあります。そのため、スレッド化されたコピーを行う小さなユーティリティを作成しました。
# MAGIC
# MAGIC このユーティリティは、ドライバのメモリから指定したボリュームパスに解凍した画像をコピーするために使用されます。通常のバージョンを使用する場合に比べて約150倍速く動作します。

# COMMAND ----------

# Standard Imports
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# External Imports
from tqdm import tqdm


# TODO: 最適なスレッド数を確認する
def threaded_dbutils_copy(source_directory, target_directory, n_threads=10):
  """
  スレッドを使用してソースディレクトリをターゲットディレクトリにコピーします。
  
  この関数はスレッドを使用して複数のコピーコマンドを実行し、コピー処理を高速化します。
  特に画像のような小さなファイルが多数ある場合に便利です。
  
  :param source_directory: ファイルがコピーされる元のディレクトリ
  :param target_directory: ファイルがコピーされる先のディレクトリ
  :param n_threads: 使用するスレッド数。数が多いほどプロセスが速くなります。
  
  注意事項
    - パスの末尾にバックスラッシュを含めないでください。
    - n_threadsを増やすとドライバに負荷がかかるため、メトリクスを監視してドライバが過負荷にならないようにしてください。
    - 100スレッドは適切なドライバに適度な負荷をかけます。
  """
  
  print("すべてのパスをリストしています")
  
  # すべてのファイルのための空のリストを作成
  all_files = []
  
  # すべてのファイルを発見するための再帰的な検索関数
  # TODO: これをジェネレータに変える
  def recursive_search(_path):
    file_paths = dbutils.fs.ls(_path)
    for file_path in file_paths:
      if file_path.isFile():
        all_files.append(file_path.path)
      else:
        recursive_search(file_path.path)
  
  # ソースディレクトリに再帰的な検索を適用
  recursive_search(source_directory)
  
  # パス文字列のフォーマット
  all_files = [path.split(source_directory)[-1][1:] for path in all_files]
  
  n_files = len(all_files)
  print(f"{n_files} ファイルが見つかりました")
  print(f"{n_threads} スレッドでコピーを開始します")
  
  # 進行状況バーを作成するためのスレッドロックを使用してTQDMを初期化
  p_bar = tqdm(total=n_files, unit=" コピー")
  bar_lock = Lock()
  
  # 単一スレッドで実行される作業を定義
  def single_thread_copy(file_sub_path):
    dbutils.fs.cp(f"{source_directory}/{file_sub_path}", f"{target_directory}/{file_sub_path}")
    with bar_lock:
      p_bar.update(1)
  
  # すべてのパスにスレッド作業をマッピング
  with ThreadPoolExecutor(max_workers=n_threads, thread_name_prefix="copy_thread") as ex:
    ex.map(single_thread_copy, all_files)
  
  # 進行状況バーを閉じる
  p_bar.close()
  
  print("コピー完了")
  return

# COMMAND ----------

# パスを指定
source_dir = "file:/tmp_landing_zone"
target_dir = "/Volumes/takaakiyayoi_catalog/item_onboarding/landing_zone/"

# コピーを実行
threaded_dbutils_copy(
  source_directory=source_dir, 
  target_directory=target_dir, 
  n_threads=150 # 同時に実行するスレッド数はどれくらいにしますか？ 数を増やすことを恐れないでください。
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 生データの読み込みと保存
# MAGIC
# MAGIC 生データをボリュームの場所に移動したので、それを読み込んでDeltaテーブルとして保存できます。

# COMMAND ----------

# データを読み込む
products_df = (
    spark.read.json("/Volumes/takaakiyayoi_catalog/item_onboarding/landing_zone/listings/metadata")
)

# スキーマを上書きして生データを保存
(
    products_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.products_raw")
)

# display(products_df)

# COMMAND ----------

# インポート
from pyspark.sql import functions as SF

# データを読み込む
image_meta_df = (
  spark
      .read
      .csv(
        path="/Volumes/takaakiyayoi_catalog/item_onboarding/landing_zone/images/metadata",
        sep=',',
        header=True
    ) 
)

# 画像データを保存
(
    image_meta_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.image_meta_raw")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 基本的なクリーニング
# MAGIC
# MAGIC テキストベースのデータにはいくつかのネストされた部分があります。基本的なクリーニングと抽出を行い、使用可能な形式に変換します。

# COMMAND ----------

# インポート
from pyspark.sql import functions as SF

# データを読み込む
products_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.products_raw")


# 標準カラムから値を抽出する関数を作成
def value_extractor(df, target_col, sep=""):
    df = (
        df
        .withColumn(
            target_col,
            SF.expr(
                f"""concat_ws('{sep} ', filter({target_col}, x -> x.language_tag in ("en_US")).value)"""
            ),
        )
    )
    return df


# US製品に焦点を当てた変換データフレームを作成
products_clean_df = products_df.filter(SF.col("country").isin(["US"]))

# 変換を適用
transformation_columns = [
    ("brand", ""),
    ("bullet_point", ""),
    ("color", ""),
    ("item_keywords", " |"),
    ("item_name", ""),
    ("material", " |"),
    ("model_name", ""),
    ("product_description", ""),
    ("style", ""),
    ("fabric_type", ""),
    ("finish_type", ""),
]

for row in transformation_columns:
    products_clean_df = value_extractor(products_clean_df, row[0], row[1])

# メタカラムを指定
meta_columns = [
    ### メタ
    "item_id",
    "country",
    "main_image_id",
]

transformed_columns = []
for row in transformation_columns:
    transformed_columns.append(row[0])

in_place_transformed_columns = [
    ### インプレース変換
    "product_type.value[0] AS product_type",
    "node.node_name[0] AS node_name",
]


# カラム変換と選択を適用
products_clean_df = products_clean_df.selectExpr(
    meta_columns + transformed_columns + in_place_transformed_columns
)

# item_idに基づいて重複を削除
products_clean_df = products_clean_df.dropDuplicates(["item_id"])

# クリーンな製品データを保存
(
    products_clean_df.write.mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.products_clean")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 画像メタデータの補強
# MAGIC
# MAGIC 次に、画像のメタデータを画像のパスで補強します。これにより、後で製品とメイン画像IDのパスを簡単に一致させることができます。

# COMMAND ----------

from pyspark.sql import functions as SF

# データフレームを読み込む
products_clean_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.products_clean")
image_meta_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.image_meta_raw")

# メイン画像IDで強化
image_meta_enriched_df = image_meta_df.join(
    products_clean_df.selectExpr("main_image_id AS image_id", "item_id"),
    on="image_id",
    how="left",
)

# 実際のパスを構築
real_path_prefix = "/Volumes/takaakiyayoi_catalog/item_onboarding/landing_zone/images/small/"
image_meta_enriched_df = image_meta_enriched_df.withColumn(
    "real_path", 
    SF.concat(
        SF.lit(real_path_prefix),  # 文字列をリテラルに変換
        SF.col('path')
    )
)

# 保存
(
    image_meta_enriched_df.write.mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.image_meta_enriched")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### サンプルとテストデータの作成
# MAGIC
# MAGIC 速度と再現性のために、100アイテムに焦点を当てることにします。これにより、データのバッチをタイムリーに処理し、結果を再現するのにも役立ちます。ただし、プロジェクトを大規模に実行したい場合は、制限数を100からより大きな数に変更するか、制限文をコメントアウトしてフルスケールで実行してください。

# COMMAND ----------

# テスト用に限定された数の製品を取得
sample_df = (
    spark.read.table("takaakiyayoi_catalog.item_onboarding.products_clean")
    .select("item_id")
    .limit(100)  # 必要に応じて増やすかコメントアウトしてください。
)

# テスト用に限定された数の製品を保存
(
    sample_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.sample")
)

# COMMAND ----------

# クリーンな製品サンプル
products_clean_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.products_clean")
sample_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.sample")
sampled_products_clean_df = sample_df.join(products_clean_df, on="item_id", how="left")

# 保存
(
    sampled_products_clean_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.products_clean_sampled")
)

# COMMAND ----------

# サンプル画像
image_meta_enriched_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.image_meta_enriched")
sample_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.sample")
sampled_image_meta_enriched_df = sample_df.join(image_meta_enriched_df, on="item_id", how="left")

# 保存
(
    sampled_image_meta_enriched_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("takaakiyayoi_catalog.item_onboarding.image_meta_enriched_sampled")
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC これでデータ準備ノートブックが完了です。次のノートブックでは、サンプル化されたテーブルと、Volumeに保存した画像を使用して情報抽出を開始します。
