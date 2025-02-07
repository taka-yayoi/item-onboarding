# Databricks notebook source
# MAGIC %md
# MAGIC # テキスト分析
# MAGIC
# MAGIC 前のセクションで画像分析を完了し、観察された説明や観察された色などのデータポイントを得ました。次に、テキストベースのLLMモデルを使用して、すべてのテキストを整理します。
# MAGIC
# MAGIC このノートブックの目標は、サプライヤーから提供された情報やワークフローを通じて収集されたすべての情報を考慮して、最終的な説明や最終的な色などのテキストポイントを作成することです。
# MAGIC
# MAGIC コードに関しては、[vLLM](https://docs.vllm.ai/en/latest/)を除いて非常に似たフローに従います。VLLMは、実行時にモデルを最適化するのに役立つ非常に人気のあるライブラリです。ほとんどすべてのSOTAオープンソースモデルと連携します。実際、ビジョンモデル用の実験的なアプリケーションもありますが、まだ本番環境には対応していないため、前のノートブックでは使用しませんでした。
# MAGIC
# MAGIC vLLMを使用する場合、特にモデルを呼び出すポイントでコードの設計が若干異なりますが、Rayを中心とした他の部分はほぼ同じです。
# MAGIC
# MAGIC ここでも同様のフローに従い、最初にプロンプトを使ったインタラクティブなテストを行い、その後バッチ推論のための必要なフローを設計します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### ライブラリのインストール
# MAGIC
# MAGIC 必要なライブラリ、transformersとvllmをインストールします

# COMMAND ----------

# 必要なライブラリをインストールするためのコード
%pip install --upgrade transformers -q
%pip install vllm -q

# COMMAND ----------

# この操作は、上記のライブラリインストールセルとは別のセルで行う必要があります
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### デフォルトの設定
# MAGIC
# MAGIC デフォルトのUnity Catalogとスキーマを指定し、中間データを保存するためのボリュームと、オンボーディングdfを保持するパスを作成します。

# COMMAND ----------

# MAGIC %sql
# MAGIC -- デフォルトの定義
# MAGIC USE CATALOG takaakiyayoi_catalog;
# MAGIC USE SCHEMA item_onboarding;
# MAGIC -- 一時データの保存場所を作成
# MAGIC CREATE VOLUME IF NOT EXISTS interim_data;

# COMMAND ----------

# ターゲットパスを指定
onboarding_df_path = "/Volumes/takaakiyayoi_catalog/item_onboarding/interim_data/onboarding"

# COMMAND ----------

# MAGIC %md
# MAGIC ### 中間データの構築
# MAGIC
# MAGIC サプライヤーから取得したデータポイントと、ビジュアルモデルから取得したデータポイントをすべて取り込み、それらを結合してテキストベースのワークフローで使用できるようにする必要があります。
# MAGIC
# MAGIC RayがDatabricksのボリュームからParquetファイルを取得する方が簡単なので、セルの最後に、最終的な中間データフレームをボリュームにParquet形式で保存します。

# COMMAND ----------

from pyspark.sql import functions as SF

# 処理対象のテーブルをparquet形式で構築
products_clean_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.products_clean_sampled")
image_meta_df = spark.read.table("takaakiyayoi_catalog.item_onboarding.image_meta_enriched_sampled")
image_analysis_df = spark.read.parquet("/Volumes/takaakiyayoi_catalog/item_onboarding/interim_data/image_analysis")

# 基本的な変換
image_analysis_df = (
    image_analysis_df
    .drop("image")
    .selectExpr([
        "path AS real_path", 
        "description AS gen_description", 
        "color AS gen_color",
    ])
)

# 生成された説明と色のテキストをクリーンアップ
pattern = r"assistant<\|end_header_id\|>\s*([\s\S]*?)<\|eot_id\|>"
image_analysis_df = (
    image_analysis_df
    .withColumn("gen_description", SF.regexp_extract("gen_description", pattern, 1))
    .withColumn("gen_color", SF.regexp_extract("gen_color", pattern, 1))
)

# 後で使用するために画像説明のエントリを準備
onboarding_df = (
    products_clean_df
    .join(image_meta_df, on="item_id", how="left")
    .join(image_analysis_df, on="real_path", how="left")
)

# 指定された場所にパーケット形式で保存
(
    onboarding_df
    .write
    .mode("overwrite")
    .parquet(onboarding_df_path)
)
# display(onboarding_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### ターゲット製品分類の構築
# MAGIC
# MAGIC このセクションでは、小売業者がカタログのために事前定義された分類法を持っているシナリオをシミュレートしたいと考えました。後のタスクは、この分類法の中にアイテムを配置することです。これは通常、小売業者が製品を分類するのに役立ちます。そこで、実際のような分類法を生成し、モデルがそれでどのように機能するかを確認しようと考えました。

# COMMAND ----------

product_taxonomy = """- 家具・家庭用品 - 椅子
- 家具・家庭用品 - テーブル
- 家具・家庭用品 - ソファ・カウチ
- 家具・家庭用品 - キャビネット、ドレッサー、ワードローブ
- 家具・家庭用品 - ランプ・照明器具
- 家具・家庭用品 - 棚・本棚
- フットウェア・アパレル - 靴
- フットウェア・アパレル - 衣類
- フットウェア・アパレル - アクセサリー
- キッチン・ダイニング - 調理器具
- キッチン・ダイニング - 食器
- キッチン・ダイニング - カトラリー・調理器具
- キッチン・ダイニング - 収納・整理
- ホームデコ・アクセサリー - 花瓶・装飾用ボウル
- ホームデコ・アクセサリー - 写真立て・壁掛けアート
- ホームデコ・アクセサリー - 装飾用クッション・スロー
- ホームデコ・アクセサリー - ラグ・マット
- 家電 - ヘッドホン・イヤホン
- 家電 - ポータブルスピーカー
- 家電 - キーボード、マウス、その他周辺機器
- 家電 - 携帯電話ケース・スタンド
- オフィス・文房具 - デスクオーガナイザー・ペンホルダー
- オフィス・文房具 - ノート・ジャーナル
- オフィス・文房具 - ペン、鉛筆、マーカー
- オフィス・文房具 - フォルダー、バインダー、ファイルオーガナイザー
- パーソナルケア・アクセサリー - ウォーターボトル・タンブラー
- パーソナルケア・アクセサリー - メイクブラシ・ヘアアクセサリー
- パーソナルケア・アクセサリー - パーソナルグルーミングツール
- おもちゃ・レジャー - アクションフィギュア・人形
- おもちゃ・レジャー - ブロック・建設セット
- おもちゃ・レジャー - ボードゲーム・パズル
- おもちゃ・レジャー - ぬいぐるみ"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### インタラクティブモデルとプロンプトの構成
# MAGIC
# MAGIC これからテキストモデルを使ったインタラクティブな部分を始めます。ここでの目標は、モデルの動作をテストし、テキスト分析のためのプロンプトエンジニアリングを行うことです。
# MAGIC
# MAGIC 画像モデルで行った方法と同様に、アクターを作成します。ここでの違いは、vLLMライブラリを使用してモデルをロードすることです。vLLMは最適化されているため、モデルのロード（ボリュームからGPUメモリへの）や推論が高速化されることが期待できます。

# COMMAND ----------

# インポート
from vllm import LLM, SamplingParams
import ray

# Rayの初期化
ray.init(ignore_reinit_error=True)

# モデルパスの指定
model_path = "/Volumes/takaakiyayoi_catalog/item_onboarding/models/llama-31-8b-instruct/"

# LLMをGPUにロード
@ray.remote(num_gpus=1)
class LLMActor:
    def __init__(self, model_path: str):
        # モデルの初期化
        self.model = LLM(model=model_path, max_model_len=2048)

    def generate(self, prompt, sampling_params):
        raw_output = self.model.generate(
            prompt, 
            sampling_params=sampling_params
        )
        return raw_output

# LLMアクターの作成 - この部分はモデルを非同期でGPUにロードします。
llm_actor = LLMActor.remote(model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### プロンプト技術
# MAGIC
# MAGIC 私たちはLLAMA 3.1 8B instructモデルを使用しています。このモデルは、ベースモデルとは少し異なる特定の方法で呼び出されることを期待しています。この特別な方法では、プロンプトや指示を特別なトークンと事前設定された構造でフォーマットする必要があります。この構造では、システムプロンプトを受け取ることを期待しており、これはモデルに「あなたは役に立つアシスタントです」といったことを伝えます。指示についても同様です。これらのテキストは特別なトークンの前後に配置され、トークンは次のように見えます: `<|eot_id|>`。この技術に関する詳細は、[Metaのモデルドキュメント](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md)を参照してください。
# MAGIC
# MAGIC 以下のセルでは、システムテキストと指示テキストを与えられた場合に、正しい形式でプロンプトを構築できる基本的な関数を作成します。

# COMMAND ----------

# Llamaプロンプト形式
def produce_prompt(system, instruction):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return prompt

test_prompt = produce_prompt("あなたは役に立つアシスタントです", "1週間は何日ですか")
print(test_prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC 次に、簡単なテストを行いましょう：

# COMMAND ----------

# アクターを呼び出して、上記で作成した生成リクエストを実行
result = ray.get(llm_actor.generate.remote(test_prompt, SamplingParams(temperature=0.1)))

# 結果のフォーマット表示
print(result)

# 実際の結果オブジェクトは出力のリストなので、最初のものにアクセスする必要があります
print("\n")
print(result[0].prompt)
print("\n")
print(" ".join([o.text for o in result[0].outputs]).strip())
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC 私たちのモデルは動作しています。いくつかの実際の例でテストを開始しましょう。以下のセルで実際のデータセットを読み込みます。

# COMMAND ----------

# データセットを読み込んでいくつかの例を取得
onboarding_ds = ray.data.read_parquet(onboarding_df_path)

# スキーマを表示
display(onboarding_ds.schema())

# COMMAND ----------

# MAGIC %md
# MAGIC このデータセットから単一のレコードがどのように見えるかを確認しましょう

# COMMAND ----------

# レコードを1つ取得
single_record = onboarding_ds.take(2)[1]
print(single_record)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 一般的なサンプリングパラメータ
# MAGIC
# MAGIC サンプリングパラメータを使用してモデルの出力を調整できます。ここで調整できる引数は多数あります。例えば、温度の設定によって、モデルをより「創造的」にするか、より「指示に従う」ようにするかを決定できます。top_pやtop_kパラメータを変更することでトークン選択プロセスを調整したり、max_tokensを変更することでモデルが返す回答の長さを決定したりできます。詳細は[vLLMサンプリングパラメータ](https://docs.vllm.ai/en/stable/dev/sampling_params.html)をご覧ください。

# COMMAND ----------

sampling_params = SamplingParams(
    n=1, # 与えられたプロンプトに対して返される出力シーケンスの数
    temperature=0.1, # サンプリングのランダム性。値が低いほどモデルは決定論的になり、値が高いほどモデルはランダムになります。ゼロは貪欲サンプリングを意味します。
    top_p=0.9, # 考慮するトップトークンの累積確率
    top_k=50, # 考慮するトップトークンの数
    max_tokens=256,  # 特定のタスクに基づいてこの値を調整します
    stop_token_ids=[128009], # 生成が行われたときに生成を停止する
    presence_penalty=0.1, # 生成されたテキストに既に登場しているかどうかに基づいて新しいトークンをペナルティ化します
    frequency_penalty=0.1, # 生成されたテキストにおける頻度に基づいて新しいトークンをペナルティ化します
    ignore_eos=False, # EOSトークンを無視して、EOSトークンが生成された後もトークンの生成を続けるかどうか
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 説明プロンプト
# MAGIC
# MAGIC 説明プロンプトを始めましょう。画像モデルによって生成された視覚的な説明と、サプライヤーから受け取った情報を元に、新しい説明を生成するようモデルに依頼します。

# COMMAND ----------

# 推奨される説明 - システムプロンプト
description_system_prompt = "あなたは小売製品の専門ライターです。"

# 推奨される説明 - 指示プロンプト
description_instruction = """
以下に、製品の2つの説明があります。重要な詳細を捉えた自然で明確な説明（50語未満）を作成してください。

説明1: {bullet_point}
説明2: {gen_description}

新しい説明のみを出力してください。引用符や追加のテキストは不要です。
"""

# プロンプトに値を埋め込む
description_instruction = description_instruction.format(
    bullet_point=single_record["bullet_point"],
    gen_description=single_record["gen_description"],
)

# プロンプトをフォーマット
description_prompt = produce_prompt(
    system = description_system_prompt,
    instruction=description_instruction
    )

print(description_prompt)

result = ray.get(llm_actor.generate.remote(description_prompt, sampling_params))
suggested_description = " ".join([o.text for o in result[0].outputs]).strip()
print(suggested_description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### カラープロンプト
# MAGIC
# MAGIC 説明が準備できたので、モデルに製品の最適な色を生成するよう依頼しましょう。サプライヤーからのデータの一部には色のフィールドが欠けているため、視覚モデルからの入力が重要になります。

# COMMAND ----------

# 推奨される色 - システムプロンプト
color_system_prompt = "あなたは色の専門アナリストです。"

# 推奨される色 - 指示プロンプト
color_instruction = """
以下を考慮してください:
- 製品の色: {color}
- ビジョンモデルの色: {gen_color}

色を返してください。追加のテキストは不要です。
"""


# プロンプトに値を埋め込む
color_instruction = color_instruction.format(
    color=single_record["color"],
    gen_color=single_record["gen_color"],
)

# プロンプトをフォーマット
color_prompt = produce_prompt(
    system = color_system_prompt,
    instruction=color_instruction
)

print(color_prompt)

result = ray.get(llm_actor.generate.remote(color_prompt, sampling_params))
suggested_color = " ".join([o.text for o in result[0].outputs]).strip()
print(suggested_color)

# COMMAND ----------

# MAGIC %md
# MAGIC これは素晴らしい例です。サプライヤーが提供した色「ハンター」は実際の色ではありません。視覚モデルは実際の色が「緑」であることを確認し、テキストモデルもそれを決定します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### キーワードプロンプト
# MAGIC
# MAGIC サプライヤーは検索最適化のために多数のキーワードも提供してくれますが、ここからもキーワードが複数回繰り返されたり、実際のアイテムと一致しなかったりする問題のあるデータポイントが出てきます。
# MAGIC
# MAGIC この部分では、同じ形式を維持しながらキーワードを最適化することを目指します。

# COMMAND ----------

# 推奨されるキーワード - システムプロンプト
keyword_system_prompt = "あなたはSEOと製品キーワードの専門家です。"

# 推奨されるキーワード - 指示プロンプト
keyword_instruction = """
入力:
- 現在のキーワード: {item_keywords}
- 製品説明: {suggested_description}
- 製品の色: {suggested_color}

新しいキーワードを|で区切って返してください。他のテキストは不要です。説明しないでください。
"""


# プロンプトをフォーマット
keyword_prompt = produce_prompt(
    system = keyword_system_prompt,
    instruction=keyword_instruction
)

# プロンプトに値を埋め込む
keyword_prompt = keyword_prompt.format(
    item_keywords=single_record["item_keywords"],
    suggested_description=suggested_description,
    suggested_color=suggested_color,
)


print(keyword_prompt)

result = ray.get(llm_actor.generate.remote(keyword_prompt, sampling_params))
suggested_keywords = " ".join([o.text for o in result[0].outputs]).strip()
print("\n")
print(suggested_keywords)

# COMMAND ----------

# MAGIC %md
# MAGIC そして、モデルはそれも成功裏に行うことができます！

# COMMAND ----------

# MAGIC %md
# MAGIC ### カテゴリープロンプト
# MAGIC
# MAGIC 最後に、これまで生成および修正したすべての情報をもとに、アイテムをノートブックの上部で作成したカテゴリのいずれかにモデルが配置します。
# MAGIC
# MAGIC この部分では、モデルは前のセルから生成したテキストも使用します。

# COMMAND ----------

# 推奨される分類 - システムプロンプト
taxonomy_system_prompt = "あなたは専門のマーチャンダイズ分類スペシャリストです。"

# 推奨される分類 - 指示プロンプト
taxonomy_instruction = """
製品説明を確認し、提供された分類から最も適切なカテゴリを選択してください。
製品説明: 
{suggested_description}

製品分類: 
{target_taxonomy}

最も適したカテゴリを1つだけ返してください。他のテキストは不要です。
"""

# プロンプトをフォーマット
taxonomy_prompt = produce_prompt(
    system = taxonomy_system_prompt,
    instruction=taxonomy_instruction
)

# プロンプトに値を埋め込む
taxonomy_prompt = taxonomy_prompt.format(
    suggested_description=suggested_description,
    target_taxonomy=product_taxonomy,
)

print(taxonomy_prompt)

result = ray.get(llm_actor.generate.remote(taxonomy_prompt, sampling_params))
suggested_category = " ".join([o.text for o in result[0].outputs]).strip()

print("\n")
print(suggested_category)

# COMMAND ----------

# MAGIC %md
# MAGIC モデルはアイテムを正しいカテゴリに正常に配置します。

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPUアンロード
# MAGIC
# MAGIC バッチ推論の準備が整ったので、続行する前にRayをシャットダウンしてGPUをアンロードします。

# COMMAND ----------

ray.shutdown()

# COMMAND ----------

# MAGIC %md
# MAGIC ### バッチ推論
# MAGIC
# MAGIC モデルと対話的に作業し、プロンプトがモデルとどのように機能するかを理解したので、バッチ推論のフローを設定する時が来ました。

# COMMAND ----------

# MAGIC %md
# MAGIC ### Rayの初期化とデータの取得
# MAGIC
# MAGIC Rayを再初期化し、バッチ推論のためのデータセットを取得します。

# COMMAND ----------

# Imports
import ray

# Init ray
ray.init()

# データを取得
onboarding_ds = ray.data.read_parquet(onboarding_df_path)

# スキーマを確認
onboarding_ds.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ### 推論ロジック
# MAGIC
# MAGIC 推論の設計方法は、画像モデルで行った方法と非常に似ていますが、ここではvLLMを使用する点が異なります。
# MAGIC
# MAGIC `__init__`メソッドと`__call__`メソッドを持つクラスを使用し、`__call__`メソッドが推論のフローを保持します。フローは重要で、最初のステップで生成された回答が後の段階で使用されるため、順序が必要です。
# MAGIC
# MAGIC また、プロンプトのフォーマットなどを標準化するためのヘルパー関数も作成します。

# COMMAND ----------

# Imports
from vllm import LLM, SamplingParams
import numpy as np


class OnboardingLLM:
    # クラスの構築
    def __init__(self, model_path: str, target_taxonomy: str):
        # モデルの初期化
        self.model = LLM(model=model_path, max_model_len=2048)
        self.target_taxonomy = target_taxonomy

    def __call__(self, batch):
        """各バッチで実行されるロジックを定義"""
        # すべての推論ロジックはここに入る
        batch = self.generate_suggested_description(batch)
        batch = self.generate_suggested_color(batch)
        batch = self.generate_suggested_keywords(batch)
        batch = self.generate_suggested_product_category(batch)
        return batch

    @staticmethod
    def format_prompt(system, instruction):
        """プロンプトのフォーマットを支援"""
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return prompt

    @staticmethod
    def standardise_output(raw_output):
        """各推論後の標準化された出力を返す"""
        generated_outputs = []
        for _ro in raw_output:
            generated_outputs.append(" ".join([o.text for o in _ro.outputs]))
        return generated_outputs

    @staticmethod
    def build_sampling_params(max_tokens=256):
        """推論のためのサンプリングパラメータを構築"""
        sampling_params = SamplingParams(
            n=1,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=max_tokens,  # 特定のタスクに基づいてこの値を調整
            stop_token_ids=[128009], # LLAMA 3.1 <|eot_id|>に特有
            presence_penalty=0.1,
            frequency_penalty=0.1,
            ignore_eos=False,
        )
        return sampling_params

    def generate_suggested_description(self, batch):
        # 推奨される説明 - システムプロンプト
        system_prompt = "あなたは小売製品の専門ライターです。"

        # 推奨される説明 - 指示プロンプト
        instruction = """
        以下は製品の2つの説明です。主要な詳細を捉えた自然で明確な説明（50語以内）を作成してください。

        説明1: {bullet_point}
        説明2: {gen_description}

        新しい説明のみを出力してください。引用符や追加のテキストは不要です。
        """

        # プロンプトを構築
        prompt_template = produce_prompt(system=system_prompt, instruction=instruction)
        prompts = np.vectorize(prompt_template.format)(
            bullet_point=batch["bullet_point"], gen_description=batch["gen_description"]
        )

        # サンプリングパラメータを構築
        sampling_params = self.build_sampling_params(max_tokens=256)

        # 推論
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # バッチに戻す
        batch["suggested_description"] = self.standardise_output(raw_output)

        return batch

    def generate_suggested_color(self, batch):
        # 推奨される色 - システムプロンプト
        system_prompt = "あなたは色の専門アナリストです。"

        # 推奨される色 - 指示プロンプト
        instruction = """
        製品の:
        - 説明された色: {color}
        - 観察された色: {gen_color}

        色を返してください。追加のテキストは不要です。
        """

        # プロンプトをフォーマット
        prompt_template = produce_prompt(system=system_prompt, instruction=instruction)
        prompts = np.vectorize(prompt_template.format)(
            color=batch["color"], gen_color=batch["gen_color"]
        )

        # サンプリングパラメータを構築
        sampling_params = self.build_sampling_params(max_tokens=16)

        # 推論
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # バッチに戻す
        batch["suggested_color"] = self.standardise_output(raw_output)

        return batch

    def generate_suggested_keywords(self, batch):
        # 推奨されるキーワード - システムプロンプト
        system_prompt = "あなたはSEOと製品キーワードの専門家です。"

        # 推奨されるキーワード - 指示プロンプト
        instruction = """
        入力:
        - 現在のキーワード: {item_keywords}
        - 製品説明: {suggested_description}
        - 製品の色: {suggested_color}

        新しいキーワードを|で区切って返してください。その他のテキストは不要です。説明しないでください。
        """

        # プロンプトをフォーマット
        prompt_template = produce_prompt(system=system_prompt, instruction=instruction)
        prompts = np.vectorize(prompt_template.format)(
            item_keywords=batch["item_keywords"],
            suggested_description=batch["suggested_description"],
            suggested_color=batch["suggested_color"],
        )

        # サンプリングパラメータを構築
        sampling_params = self.build_sampling_params(max_tokens=256)
        
        # 推論
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # バッチに戻す
        batch["suggested_keywords"] = self.standardise_output(raw_output)

        return batch
    
    def generate_suggested_product_category(self, batch):

        # 推奨されるカテゴリ - システムプロンプト
        system_prompt = "あなたは商品分類の専門家です。"

        # 推奨されるカテゴリ - 指示プロンプト
        instruction = """
        製品説明を確認し、提供された分類から最も適切なカテゴリを選択してください。
        製品説明: 
        {suggested_description}

        製品分類: 
        {target_taxonomy}

        最も適したカテゴリを1つだけ返してください。その他のテキストは不要です。
        """

        # プロンプトをフォーマット
        prompt_template = produce_prompt(system=system_prompt, instruction=instruction)
        prompts = np.vectorize(prompt_template.format)(
            suggested_description=batch["suggested_description"],
            target_taxonomy=self.target_taxonomy
        )

        # サンプリングパラメータを構築
        sampling_params = self.build_sampling_params(max_tokens=256)
        
        # 推論
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # バッチに戻す
        batch["suggested_category"] = self.standardise_output(raw_output)

        return batch

# COMMAND ----------

# MAGIC %md
# MAGIC ### 推論の実行
# MAGIC
# MAGIC クラスの準備が整ったので、推論ロジックを実行しましょう！
# MAGIC
# MAGIC 結果を再びボリュームにParquetファイルとして保存し、次のノートブックで結果を確認します。

# COMMAND ----------

# モデルパスを指定
model_path = "/Volumes/takaakiyayoi_catalog/item_onboarding/models/llama-31-8b-instruct/"

# モデルの重みを保存するフォルダを指定
#model_weights_folder = "takaakiyayoi_catalog.review_summarisation.model_weights"

# データを取得
onboarding_ds = ray.data.read_parquet(onboarding_df_path)

# フローを実行
ft_onboarding_ds = onboarding_ds.map_batches(
    OnboardingLLM,
    concurrency=1,  # LLMインスタンスの数
    num_gpus=1,  # LLMインスタンスごとのGPU数
    batch_size=32,  # OOMになるまで最大化、OOMになったらbatch_sizeを減らす
    fn_constructor_kwargs={
        "model_path": model_path,
        "target_taxonomy": product_taxonomy,
    },
)

# 評価
ft_onboarding_ds = ft_onboarding_ds.materialize()

# 結果を保存する場所を指定
save_path = "/Volumes/takaakiyayoi_catalog/item_onboarding/interim_data/results"

# フォルダをクリア
dbutils.fs.rm(save_path, recurse=True)

# 保存
ft_onboarding_ds.write_parquet(save_path)

# COMMAND ----------

ray.shutdown()

# COMMAND ----------


