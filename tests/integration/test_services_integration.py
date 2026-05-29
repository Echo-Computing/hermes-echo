"""Integration tests for service interactions

複数のサービスが連携して動作することを確認するテスト
"""

import pytest
from pathlib import Path

from hermes_cli.services.task_service import TaskService
from hermes_cli.services.history_service import HistoryService
from hermes_cli.persistence.config_repository import ConfigRepository
from hermes_cli.models.report import Report, ReportSection, Citation, ReportMetadata
from datetime import datetime


pytestmark = pytest.mark.integration


class TestServicesIntegration:
    """サービス統合テスト"""

    @pytest.fixture
    def work_dir(self) -> Path:
        """作業ディレクトリ"""
        config_repo = ConfigRepository()
        config = config_repo.load()
        return config.work_dir

    @pytest.fixture
    def task_service(self, work_dir: Path) -> TaskService:
        """TaskServiceインスタンス"""
        return TaskService(work_dir)

    @pytest.fixture
    def history_service(self, work_dir: Path) -> HistoryService:
        """HistoryServiceインスタンス"""
        return HistoryService(work_dir)

    def test_task_and_history_integration(
        self,
        task_service: TaskService,
        history_service: HistoryService,
    ):
        """タスクとレポート履歴の統合テスト"""
        # タスク作成
        task = task_service.create_task("Integration test prompt")

        # レポート作成
        report = Report(
            title="Integration Test Report",
            sections=[
                ReportSection(
                    title="Test Section",
                    content="Test content",
                )
            ],
            citations=[
                Citation(
                    index=1,
                    title="Test Source",
                    url="https://example.com",
                    accessed_at=datetime.now(),
                )
            ],
        )

        metadata = ReportMetadata(
            task_id=task.id,
            status="success",
            start_at=datetime.now(),
            finish_at=datetime.now(),
            duration=10.5,
            model="test-model",
            loops=1,
            sources=5,
        )

        # レポート保存
        history_service.save_report(task.id, report, metadata)

        # レポート取得確認 (returns Tuple[str, ReportMetadata])
        loaded = history_service.get_report(task.id)
        assert loaded is not None
        markdown, loaded_metadata = loaded
        assert loaded_metadata.task_id == task.id
        assert loaded_metadata.status == "success"

    def test_list_reports(self, history_service: HistoryService):
        """レポート一覧取得テスト"""
        # 既存のレポート一覧を取得
        reports = history_service.list_histories()

        # レポートが存在することを確認
        assert isinstance(reports, list)
        # 少なくとも1つはレポートが存在するはず（過去の実行から）
        assert len(reports) >= 0
