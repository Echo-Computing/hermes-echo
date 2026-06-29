# Hermes

> **hermes-echo** — an extended fork of [yasutoshi-lab/Hermes](https://github.com/yasutoshi-lab/Hermes) maintained by [Echo-Computing](https://github.com/Echo-Computing). It adds the **Echo agent**: an interactive LangGraph chat agent with file/shell/search/memory/web tools, learning (correction reflection, auto-memory, idea capture, session summaries), and a collaborative multi-agent research mode. All upstream features are preserved. MIT license; upstream copyright retained.

[English Documentation](README.md)

> ローカルLLMベースの高度な情報収集エージェント

**Hermes** は、研究者や技術者向けのローカル実行可能な CLI 情報収集エージェントです。Web検索、コンテンツ分析、レポート生成までを自動化し、高品質なリサーチレポートを作成します。

## 概要

Hermesは、ローカルLLM（Large Language Model）を活用した次世代のリサーチエージェントです。プライバシーを保護しながら、包括的なWeb検索と高度な分析を実行し、引用付きの高品質なレポートを自動生成します。

主な用途:
- 技術調査レポートの自動生成
- 市場分析・競合分析
- 学術研究の情報収集
- トレンド分析
- 複数ソースからの情報統合

## 特徴

- 🔒 **完全ローカル実行**: 外部API課金なし、プライバシー完全保護
- 🔍 **インテリジェント検索**: SearxNG経由で複数検索エンジンを統合
- 🤖 **自動検証ループ**: 情報の矛盾や不足を自動検出・修正
- 📝 **高品質レポート**: Markdown形式で引用付きレポートを自動生成
- 🎯 **CLI特化**: シェルスクリプト・自動化との統合が容易
- 📊 **トレーサビリティ**: Langfuseによる実行トレース記録（オプション）
- **マルチステージワークフロー**: LangGraphによる柔軟なエージェントフロー
- **並列検索処理**: 複数クエリの並列実行で高速な情報収集
- **インテリジェントキャッシング**: Redis による検索結果のキャッシング
- **品質保証**: 複数回の検証ループによる高精度な出力
- **拡張性**: モジュラー設計による機能追加の容易さ

## ドキュメント

- **[セットアップガイド](./doc/setup/setup_ja.md)**: インストールと環境構築の詳細手順
- **コマンドリファレンス**:
    - [`hermes init`](./doc/command/init_cmd_ja.md)
    - [`hermes task`](./doc/command/task_cmd_ja.md)
    - [`hermes run`](./doc/command/run_cmd_ja.md)
    - [`hermes log`](./doc/command/log_cmd_ja.md)
    - [`hermes history`](./doc/command/history_cmd_ja.md)
- **[設定ファイル (`config.yaml`)](./doc/config/config_ja.md)**: `config.yaml` の詳細な設定方法
- **[テスト戦略](./doc/test/tests_ja.md)**: プロジェクトのテストに関する方針
- **[トラブルシューティング](./doc/troubleshooting/troubleshooting_ja.md)**: よくある問題とその解決策

## アーキテクチャ

```
┌─────────────────┐
│   User Input    │
│   (Prompt)      │
└────────┬────────┘
         │
         v
┌─────────────────────────────────────────┐
│         Prompt Normalizer               │
│  (プロンプトの正規化・前処理)            │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│       Query Generator                   │
│  (LLMによる検索クエリ生成)               │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│       Web Researcher                    │
│  (SearxNGによる並列Web検索)              │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│    Container Processor                  │
│  (LLMによるコンテンツ分析・要約)          │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│      Draft Aggregator                   │
│  (ドラフトレポート作成)                  │
└────────┬────────────────────────────────┘
         │
         v
┌─────────────────────────────────────────┐
│         Validator                       │
│  (レポート検証・改善提案)                │
└────────┬────────────────────────────────┘
         │
         v
    ┌───┴───┐
    │ OK?   │
    └───┬───┘
  NO │       │ YES
     │       │
     v       v
┌─────────┐ ┌─────────────────┐
│ Query   │ │ Final Reporter  │
│Generator│ │ (最終レポート)   │
└─────────┘ └─────────────────┘
```

## 前提条件

- **OS**: Ubuntu 22.04 以上
- **Python**: 3.10 以上
- **Docker**: docker および docker-compose
- **GPU**: VRAM 16GB 推奨（Ollama用）

## インストール

詳細な手順は `doc/setup` を参照してください。

1.  **リポジトリクローン**:
    ```bash
    git clone https://github.com/yasutoshi-lab/Hermes.git
    cd Hermes
    ```
2.  **依存関係インストール**:
    ```bash
    uv sync
    uv pip install -e .
    ```
3.  **Ollama と Hermes のセットアップ**:
    `doc/setup` のガイドに従って、OllamaのインストールとHermesの初期設定を行ってください。

## 基本的な使用例

コマンドの詳細は `doc/command` を参照してください。

```bash
# 即時実行
hermes run --prompt "量子コンピュータの暗号化への影響を調査"

# タスク登録
hermes task --add "AI倫理の最新動向"

# タスク一覧表示
hermes task --list

# タスク実行
hermes run --task-id 2025-0001
```

## ディレクトリ構造

```
~/.hermes/
├── config.yaml              # 設定ファイル
├── docker-compose.yaml      # Docker設定
├── cache/                   # キャッシュ
├── task/                    # タスク定義
├── log/                     # 通常ログ
├── debug_log/               # デバッグログ
├── history/                 # 実行履歴とレポート
└── searxng/                 # SearxNG設定
```

## 設定

設定ファイルの詳細は `doc/config/config.md` を参照してください。

## ライセンス

MIT License

## 貢献

Issue や Pull Request を歓迎します！
詳細は `CONTRIBUTING.md` を参照してください。