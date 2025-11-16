# グループワーク第1回　✉メール文変換・評価

テンプレートを元に作成しました。

## 📦 セットアップ手順
### 1. リポジトリをクローン

GitHubからコードを取得します。

```bash
git clone https://github.com/toda235/gw1-demo-aiweb-app.git
cd demo-gw1-aiwebapp
```


### 2. 依存パッケージをインストール

uvがインストールされている状態で以下を実行します。

```bash
uv sync
```

💡 ここでインストールされる主なパッケージ
| パッケージ名              | 役割                                     |
| ------------------- | -------------------------------------- |
| **streamlit**       | Pythonだけでブラウザアプリを作成できるフレームワーク          |
| **transformers**    | Hugging FaceのNLPモデルを簡単に扱うためのライブラリ      |
| **torch (PyTorch)** | モデルの実行を支える数値計算エンジン（Hugging Faceモデルの基盤） |
| **accelerate**      | モデルの処理を自動的にCPU/GPUに最適化してくれる補助ライブラリ     |
| **huggingface_hub** | huggingfaceに関する機能を提供しているライブラリ（今回はAPIでLLMモデルを使うために利用）

### 3 Huggingface API
**app.py**の13行目にHuggingfaceのReadTokenを入力することで、APIを使った機能を利用できます。
```python app.py
api_key = "YourToken"
```

### 4. アプリを起動
以下のコマンドでWebアプリを起動します。
```bash
uv run streamlit run src/app.py --server.port 8000
```

## 📁 ディレクトリ構成
```txt
.
├─ src
│   ├─ app.py               # Streamlitアプリのメインコード
│   └─ particles.min.js     # 背景パーティクルのためのJSコード
├─ .gitignore
├─ .python-version
├─ pyproject.toml   # パッケージ管理
├─ README.md        # このファイル
└─ uv.lock          # パッケージ管理

```

## 😊使用モデル
- メール文変換　Qwen/Qwen2.5-7B-Instruct 
- メール文変換・評価　openai/gpt-oss-120b
- 丁寧レベル評価　microsoft/deberta-v3-base ("[Kei-Corpus ](https://github.com/Liumx2020/KeiCO-corpus)© Ochanomizu University Kobayashi Lab, Liu Muxuan" を用いたファインチューニング )