# 🧩 Streamlit 感情分析アプリ — コード説明

このフォルダには、実際に動作するStreamlitアプリのソースコードが入っています。  
以下では `app.py` の構成と動作を簡単に説明します。


## 💻 ファイル概要

### `app.py`

このファイルは、**感情分析アプリ**のメインスクリプトです。

```python
import streamlit as st
from transformers import pipeline
```
まず必要なライブラリをimportします。

### 🔍 アプリの仕組み
#### 1️⃣ タイトルと説明を表示
```python
st.title("🔍 感情分析アプリ")
st.write("テキストを入力して感情を分析します。")
```
→ブラウザ上にタイトルと説明文を表示します。

#### 2️⃣ モデルをロード
```python
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")

analyzer = load_model()
```
- `pipeline` は、Hugging Faceモデルを簡単に使うための関数です。
- `@st.cache_resource` は、Streamlitのキャッシュ機能で、同じモデルを何度も再読み込みしないようにします（高速化のため）。

#### 3️⃣ ユーザー入力
```python
text = st.text_area("テキストを入力", "今日はとても良い気分です！")
```
→ テキスト入力欄を表示し、初期値を設定します。

#### 4️⃣ ボタンで分析を実行
```python
if st.button("分析する"):
    with st.spinner("分析中..."):
        result = analyzer(text)[0]
        st.success(f"ラベル: {result['label']}, スコア: {result['score']:.3f}")
```
- 「分析する」ボタンが押されたとき、感情分析を実行します。
- `result['label']` に感情ラベル（例: POSITIVE, NEGATIVE）
- `result['score']` に確信度スコアが含まれます。

## 🌈 応用ヒント
- `pipeline("translation", model="Helsinki-NLP/opus-mt-ja-en")` に変えると翻訳アプリに！
- `pipeline("summarization")` で要約アプリにもできます。

## 📚 参考リンク
- [Streamlit公式ドキュメント](https://docs.streamlit.io/)
- [Hugging Face Transformers Pipeline](https://huggingface.co/docs/transformers/v4.57.1/en/pipeline_tutorial)
- [使用しているモデル](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)