"""Unit tests for TaskRepository"""

import pytest
from datetime import datetime
from pathlib import Path

from hermes_cli.persistence.task_repository import TaskRepository
from hermes_cli.models.task import Task


class TestTaskRepository:
    """TaskRepositoryのテスト"""

    @pytest.fixture
    def task_repo(self, temp_work_dir):
        """TaskRepositoryインスタンス"""
        return TaskRepository(temp_work_dir)

    @pytest.fixture
    def sample_task(self):
        """サンプルタスク"""
        return Task(
            id="2024-0001",
            prompt="Test prompt",
            status="scheduled",
            created_at=datetime(2024, 1, 1, 0, 0, 0),
            updated_at=datetime(2024, 1, 1, 0, 0, 0),
        )

    def test_init(self, task_repo, temp_work_dir):
        """初期化テスト"""
        assert task_repo.file_paths.work_dir == temp_work_dir

    def test_generate_task_id(self, task_repo):
        """タスクID生成テスト"""
        task_id = task_repo.generate_task_id()

        # フォーマット確認 (YYYY-NNNN)
        assert len(task_id) == 9
        assert task_id[4] == "-"
        assert task_id[:4].isdigit()
        assert task_id[5:].isdigit()

    def test_save_and_load_task(self, task_repo, sample_task):
        """タスク保存・読み込みテスト"""
        # 保存
        task_repo.save(sample_task)

        # 読み込み
        loaded_task = task_repo.load(sample_task.id)

        # 検証
        assert loaded_task is not None
        assert loaded_task.id == sample_task.id
        assert loaded_task.prompt == sample_task.prompt
        assert loaded_task.status == sample_task.status

    def test_load_nonexistent_task(self, task_repo):
        """存在しないタスクの読み込みテスト"""
        result = task_repo.load("nonexistent-id")
        assert result is None

    def test_list_tasks(self, task_repo):
        """タスク一覧取得テスト"""
        # 複数タスク保存
        tasks = [
            Task(
                id=f"2024-000{i}",
                prompt=f"Test prompt {i}",
                status="scheduled",
                created_at=datetime(2024, 1, i, 0, 0, 0),
                updated_at=datetime(2024, 1, i, 0, 0, 0),
            )
            for i in range(1, 4)
        ]

        for task in tasks:
            task_repo.save(task)

        # 一覧取得
        loaded_tasks = task_repo.list_all()

        # 検証
        assert len(loaded_tasks) == 3
        assert all(isinstance(t, Task) for t in loaded_tasks)

    def test_list_tasks_by_status(self, task_repo):
        """ステータス別タスク一覧取得テスト"""
        # 異なるステータスのタスクを保存
        tasks = [
            Task(
                id="2024-0001",
                prompt="Scheduled task",
                status="scheduled",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            Task(
                id="2024-0002",
                prompt="Running task",
                status="running",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
            Task(
                id="2024-0003",
                prompt="Completed task",
                status="completed",
                created_at=datetime.now(),
                updated_at=datetime.now(),
            ),
        ]

        for task in tasks:
            task_repo.save(task)

        # ステータスフィルタで取得 (list_all returns all, filter manually)
        all_tasks = task_repo.list_all()
        scheduled_tasks = [t for t in all_tasks if t.status == "scheduled"]
        running_tasks = [t for t in all_tasks if t.status == "running"]
        completed_tasks = [t for t in all_tasks if t.status == "completed"]

        # 検証
        assert len(scheduled_tasks) == 1
        assert len(running_tasks) == 1
        assert len(completed_tasks) == 1
        assert scheduled_tasks[0].status == "scheduled"
        assert running_tasks[0].status == "running"
        assert completed_tasks[0].status == "completed"

    def test_delete_task(self, task_repo, sample_task):
        """タスク削除テスト"""
        # 保存
        task_repo.save(sample_task)
        assert task_repo.load(sample_task.id) is not None

        # 削除
        task_repo.delete(sample_task.id)

        # 検証
        assert task_repo.load(sample_task.id) is None
