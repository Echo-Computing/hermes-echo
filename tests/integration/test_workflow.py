"""Integration tests for full workflow

これらのテストは実際の依存サービス(Redis, Ollama, SearxNG)を必要とします。
Ollama must be running AND have the required model pulled.
"""

import pytest
from pathlib import Path
import httpx

from hermes_cli.models.config import HermesConfig
from hermes_cli.services.run_service import RunService
from hermes_cli.services.task_service import TaskService
from hermes_cli.persistence.config_repository import ConfigRepository


pytestmark = pytest.mark.integration


def _ollama_ready() -> bool:
    """Check if Ollama is reachable and has the default model available."""
    try:
        # Check Ollama is running
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code != 200:
            return False
        # Check the configured model is pulled
        config = ConfigRepository().load()
        model = config.ollama.model
        models = response.json().get("models", [])
        model_names = [m.get("name", "") for m in models]
        return any(name.startswith(model) for name in model_names)
    except Exception:
        return False


ollama_required = pytest.mark.skipif(
    not _ollama_ready(),
    reason="Ollama is not running or the required model is not pulled",
)


class TestFullWorkflow:
    """完全なワークフロー統合テスト"""

    @pytest.fixture
    def config(self) -> HermesConfig:
        """実際の設定を読み込み"""
        config_repo = ConfigRepository()
        return config_repo.load()

    @pytest.fixture
    def run_service(self, config: HermesConfig) -> RunService:
        """RunServiceインスタンス"""
        return RunService(config)

    @pytest.fixture
    def task_service(self, config: HermesConfig) -> TaskService:
        """TaskServiceインスタンス"""
        return TaskService(config.work_dir)

    @pytest.mark.asyncio
    @pytest.mark.slow
    @ollama_required
    async def test_simple_prompt_execution(self, run_service: RunService):
        """シンプルなプロンプトの実行テスト"""
        prompt = "Pythonの基本的な特徴を3つ挙げてください"

        result = await run_service.execute(
            prompt=prompt,
        )

        assert result is not None
        assert result["status"] == "success"
        assert "task_id" in result
        assert "report_path" in result
        assert Path(result["report_path"]).exists()

    @pytest.mark.asyncio
    @pytest.mark.slow
    @ollama_required
    async def test_task_based_execution(
        self,
        run_service: RunService,
        task_service: TaskService,
    ):
        """タスクベースの実行テスト"""
        prompt = "機械学習の主な種類について説明してください"
        task = task_service.create_task(prompt)

        result = await run_service.execute(task_id=task.id)

        assert result is not None
        assert result["status"] == "success"
        assert result["task_id"] == task.id

        updated_task = task_service.get_task(task.id)
        assert updated_task.status == "completed"

    @pytest.mark.asyncio
    @pytest.mark.slow
    @ollama_required
    async def test_workflow_with_validation(self, run_service: RunService):
        """検証ループを含むワークフローテスト"""
        prompt = "量子コンピューティングの基本原理と応用分野について詳しく説明してください"

        result = await run_service.execute(prompt=prompt)

        assert result is not None
        assert result["status"] == "success"

        report_path = Path(result["report_path"])
        assert report_path.exists()

        report_content = report_path.read_text(encoding="utf-8")
        assert len(report_content) > 0
        assert "参考文献" in report_content or "Citations" in report_content
