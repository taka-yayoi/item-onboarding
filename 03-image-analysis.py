# Databricks notebook source
# MAGIC %md
# MAGIC # 画像分析
# MAGIC
# MAGIC データとモデルの準備が整ったので、画像分析を開始できます。このセクションでは、製品画像から有用な情報を抽出することが主な目標です。データからわかることは、商品の説明が明確でない場合や、サプライヤーが製品の色などの情報を提供し忘れることがあるということです。
# MAGIC
# MAGIC 製品の画像があるので、前のノートブックでダウンロードしたビジュアルモデルを使用して、アイテムの画像からこの情報を抽出するフローを構築することに焦点を当てます。
# MAGIC
# MAGIC ビジュアルモデルはシンプルな方法で動作します。画像と「画像内のアイテムを説明してください」といったプロンプトを提供すると、テキストが返されます。
# MAGIC
# MAGIC このノートブックでは、GPUが搭載されたマシンを使用します。`NVIDIA A100` GPUは、使用するモデルに十分なGPUメモリ（約80 GB）があるため、非常に適しています。Azureでは、`NC24_ads`がコンピュートの良い選択肢となります。
# MAGIC
# MAGIC また、必要なパッケージがインストールされている`15.4 ML GPU`ランタイムを使用します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### セットアップ
# MAGIC
# MAGIC モデルを実行するために必要なtransformersライブラリをアップグレードする基本的なセットアッププロセスから始めます。

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade transformers -q

# COMMAND ----------

# インストール後に Python を再起動することが重要であり、このコードはインストール後に別のセルで実行する必要があります
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC -- このノートブックでは既存のカタログをデフォルトで使用します
# MAGIC USE CATALOG takaakiyayoi_catalog;
# MAGIC
# MAGIC -- このノートブックのすべての操作にデフォルトでこのスキーマを使用します
# MAGIC USE SCHEMA item_onboarding;
# MAGIC
# MAGIC -- ファイルを保持するためのボリュームを作成します
# MAGIC CREATE VOLUME IF NOT EXISTS interim_data;

# COMMAND ----------

# MAGIC %md
# MAGIC ### 画像の読み込み
# MAGIC
# MAGIC 画像パスがデータフレームにリストされています。これを使用して実際の画像を読み込むことができます。以下のコードは、画像パスを持つデータフレームを読み込み、それをリストに変換します。後でこのリストを使用して画像を読み込みます。

# COMMAND ----------

# 画像パスのテーブルを取得し、すべての画像のリストを作成
image_meta_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.image_meta_enriched_sampled")
image_meta_df = image_meta_df.select("real_path")

# 収集してリストを作成
image_paths = image_meta_df.collect()
image_paths = [x.real_path for x in image_paths if x.real_path]

# COMMAND ----------

# MAGIC %md
# MAGIC 単一の画像がどのように見えるか確認してみましょう。

# COMMAND ----------

from PIL import Image
img = Image.open(image_paths[0])
img

# COMMAND ----------

# MAGIC %md
# MAGIC ### インタラクティブプログラミング
# MAGIC
# MAGIC 次に、モデルを使用してインタラクティブなプログラミングとプロンプトを行うためのインターフェースを設計します。この部分では、GPU上のワークフローをより良く管理するのに役立つ[RAY](https://www.ray.io/)フレームワークを使用し始めます。RayはGPUベースのワークフローを実行するのに優れており、Databricksプラットフォーム上で非常にスムーズに動作します。
# MAGIC
# MAGIC まず、RayのActor機能を使用して、モデルをGPUにActorとしてロードします。Actorの良い点は、好きなときに呼び出せることで、インタラクティブなプログラミングに役立ちます。指定しない限り、GPUからアンロードされません。
# MAGIC
# MAGIC ここでの推奨事項として、コンピュートのWebターミナル（Databricks経由）にアクセスできる場合、このセクションを進める際にGPUのメモリと利用状況を確認すると非常に興味深いかもしれません。Webターミナルを開き、次のシェルコマンドを入力することで確認できます：
# MAGIC
# MAGIC ```sh
# MAGIC apt-get update
# MAGIC apt install nvtop
# MAGIC nvtop
# MAGIC ```
# MAGIC
# MAGIC これは、リアルタイムでお使いのGPUを監視する助けとなるユーティリティを実行します。

# COMMAND ----------

# 必要なライブラリをインポート
from PIL import Image

from transformers import MllamaForConditionalGeneration, MllamaProcessor
import transformers

import ray
import torch

# Rayを初期化
ray.init(ignore_reinit_error=True)

# モデルが保存されているパスを指定（Volumeディレクトリ）
model_path = "/Volumes/takaakiyayoi_catalog/item_onboarding/models/llama-32-11b-vision-instruct"


# RAYアクターを定義

@ray.remote(num_gpus=1)
class LlamaVisionActor:
    def __init__(self, model_path: str):
        # モデルパスを登録
        self.model_path = model_path

        # コンフィグとモデルをロード
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        self.processor = MllamaProcessor.from_pretrained(model_path)

        # モデルをデバイスに移動
        self.model.to("cuda:0")
        self.model.eval()

    def generate(self, prompt, batch, max_new_tokens=128):

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prmpt=True)
        outputs = []
        for item in batch["image"]:
            image = Image.fromarray(item)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output = self.processor.decode(output[0])
            outputs.append(output)

        return outputs


vision_actor = LlamaVisionActor.remote(model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 説明プロンプト
# MAGIC GPUがメモリにロードされたので、いくつかのプロンプトを試してみましょう。まず、RAYを使用していくつかの画像をロードする必要があります。その後、基本的な説明プロンプトを使用して、モデルに画像に何が写っているかを説明させます。プロンプトに変更を加えたい場合、ほぼプロンプトエンジニアリングに似た方法で、このインターフェースを使用してテストすることができます。

# COMMAND ----------

# 画像を読み込む
images_df = ray.data.read_images(
    image_paths,
    include_paths=True,
    mode="RGB"
)

# 10枚の画像のバッチを作成
test_batch = images_df.take_batch(10)

# COMMAND ----------

# 説明プロンプトを書く
prompt = "画像の中の製品を説明してください"

# アクターを使用して結果を生成
results = ray.get(
    vision_actor.generate.remote(
        prompt=prompt,
        batch=test_batch,
        max_new_tokens=256,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC この時点で結果が出たので、確認してみましょう。

# COMMAND ----------

# 最初の結果を表示し、ヘッダーとターン終了トークンを削除
display(results[0].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())

# 画像も表示
img = Image.fromarray(test_batch["image"][0])
display(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ### カラープロンプト
# MAGIC
# MAGIC 同様のフローを試すことができますが、今回はデータセット内の一部の製品に色フィールドが欠けているため、製品の色を抽出することを目指します。
# MAGIC
# MAGIC コードフローは同じで、プロンプトの変更のみです。

# COMMAND ----------

# 説明プロンプトを書く
prompt = "製品の色は何ですか？"

# アクターを使用して結果を生成
color_results = ray.get(
    vision_actor.generate.remote(
        prompt=prompt,
        batch=test_batch,
        max_new_tokens=32,
    )
)

# COMMAND ----------

# 最初の結果を表示し、ヘッダーとターン終了トークンを削除
display(color_results[1].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())

# 画像も表示
img = Image.fromarray(test_batch["image"][1])
display(img)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 製品画像についてさらに質問したり、情報を抽出したりしたい場合は、ここでテストすることができます。インタラクティブな作業が完了したので、RayをシャットダウンしてGPUとアクターをアンロードできます。次のセクションでは、バッチ推論に焦点を当て始めます。

# COMMAND ----------

ray.shutdown()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 推論ロジックの定義
# MAGIC
# MAGIC プロンプトが画像に対してどのように機能するかをほぼ理解したので、バッチ推論ロジックを定義することができます。アクターを使用することも適用可能な解決策でしたが、Rayの`map_batches` APIを使用することで、スケールで実行したい場合にバッチ推論をよりよく制御できます。
# MAGIC
# MAGIC 推論ロジックのコードは部分的にアクターのコードと非常に似ていますが、バッチモードで実行するために設計するクラスには、`__init__`と`__call__`の2つのメソッドを構築する必要があります。
# MAGIC
# MAGIC `__init__`メソッドは、ここでのアクターのものとほぼ同じになります。バッチ推論を開始するときに一度呼び出されます。
# MAGIC
# MAGIC `__call__`メソッドは、クラスがインスタンス化されたときに呼び出されるものです。
# MAGIC
# MAGIC 以下の設計はこれらのルールに従い、バッチ推論のためのモデルを準備します。

# COMMAND ----------

# Imports
from transformers import MllamaForConditionalGeneration, MllamaProcessor
import transformers
from PIL import Image
import torch
import ray


class LlamaVisionPredictor:
    def __init__(self, model_path: str):
        # モデルパスを登録
        self.model_path = model_path

        # コンフィグとモデルをロード
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        self.processor = MllamaProcessor.from_pretrained(model_path)

        # モデルをデバイスに移動
        self.model.to("cuda:0")
        self.model.eval()

    def __call__(self, batch):
        # すべての推論ロジックはここに記述
        batch["description"] = self.generate(
            prompt="画像内の製品を100文字以内で説明してください。",
            batch=batch,
            max_new_tokens=256,
        )

        batch["color"] = self.generate(
            prompt="画像内の製品の色を10文字以内で返してください。",
            batch=batch,
            max_new_tokens=128,
        )

        return batch

    def generate(self, prompt, batch, max_new_tokens=128):

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prmpt=True)
        outputs = []
        for item in batch["image"]:
            image = Image.fromarray(item)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output = self.processor.decode(output[0])
            outputs.append(output)

        return outputs

# COMMAND ----------

# MAGIC %md
# MAGIC ### 推論の実行
# MAGIC
# MAGIC クラスの準備が整ったので、バッチ推論を開始できます。
# MAGIC
# MAGIC 画像を同じ方法でロードしますが、上記のように、ロードされたデータセットで`map_batches`機能を使用します。

# COMMAND ----------

# Rayをインポート
import ray

# Rayを初期化
ray.init(ignore_reinit_error=True)

# モデルパスを指定
model_path = "/Volumes/takaakiyayoi_catalog/item_onboarding/models/llama-32-11b-vision-instruct"

# 画像を読み込む
image_analysis_df = ray.data.read_images(
    image_paths,
    include_paths=True,
    mode="RGB"
)

# バッチをマップ
image_analysis_df = image_analysis_df.map_batches(
        LlamaVisionPredictor,
        concurrency=1,  # LLMインスタンスの数
        num_gpus=1, # LLMインスタンスごとのGPU数
        batch_size=10, # バッチサイズ
        fn_constructor_kwargs={"model_path": model_path,},
)

# 評価
image_analysis_df = image_analysis_df.materialize()

# 画像解析の保存先を決定
save_path = "/Volumes/takaakiyayoi_catalog/item_onboarding/interim_data/image_analysis"

# ディレクトリをクリア
#dbutils.fs.rm(save_path, recurse=True)

# 保存
image_analysis_df.write_parquet(save_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 例を一つ表示
# MAGIC
# MAGIC バッチ予測の結果も確認してみましょう。

# COMMAND ----------

# 例を表示
single_example = image_analysis_df.take(1)[0]

print(single_example["description"].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())
print(single_example["color"].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())

image = Image.fromarray(single_example["image"])
image

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rayのシャットダウン
# MAGIC
# MAGIC 画像分析が完了したので、RAYをシャットダウンしても構いません。

# COMMAND ----------

# Shutdown Ray
ray.shutdown()
