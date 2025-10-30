# グループワーク第1回 AIWebアプリ開発
このリポジトリは、**Streamlit** と **Hugging Face Transformers** を使って簡単にNLP（自然言語処理）アプリを作るためのスターターキット（Boilerplate）です。  
ハッカソンや学習用プロジェクトの土台として活用できます！

## 🚀 できること

- ブラウザ上で動く感情分析アプリをすぐに実行できます。
- 入力したテキストに対して、ポジティブ／ネガティブなどの感情を判定します。
- Hugging Faceのモデルを直接利用します（GPUサーバを利用します）。

---

## 📦 セットアップ手順

以下の手順で実行環境を整えましょう！

### 1. リポジトリをクローン

GitHubからコードを取得します。

```bash
git clone https://github.com/fuji029/demo-gw1-aiwebapp
cd demo-gw1-aiwebapp
```

> 💡 GitHubとは？
> プログラムのソースコードを共有・管理するためのサービスです。
> git clone は、リポジトリ（プロジェクト全体）を自分のPCにコピーするコマンドです。

2. 依存パッケージをインストール

Pythonがインストールされている状態で以下を実行します。

pip install -r requirements.txt


> 💡 ここでインストールされる主なパッケージ
> streamlit：WebアプリをPythonで簡単に作るためのフレームワーク
> transformers：Hugging FaceのNLPモデルを利用するためのライブラリ