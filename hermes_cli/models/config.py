"""Configuration models for Hermes"""

from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional, Literal
from pathlib import Path


class OllamaConfig(BaseModel):
    """Ollama API設定"""

    api_url: str = Field(
        default="http://localhost:11434/api/chat", description="Ollama APIエンドポイント"
    )
    model: str = Field(default="gpt-oss:20b", description="使用するLLMモデル")
    timeout: int = Field(default=120, ge=10, le=600, description="タイムアウト秒")
    retry: int = Field(default=3, ge=0, le=10, description="リトライ回数")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=128, le=32768)


class SearchConfig(BaseModel):
    """検索設定"""

    searxng_base_url: str = Field(
        default="http://localhost:8080", description="SearxNGベースURL"
    )
    redis_url: str = Field(default="redis://localhost:6379/0", description="RedisベースURL")
    min_search: int = Field(default=3, ge=1, le=20, description="最小ソース数")
    max_search: int = Field(default=8, ge=1, le=50, description="最大ソース数")
    query_count: int = Field(default=3, ge=1, le=10, description="クエリ生成数")
    cache_ttl: int = Field(default=3600, description="キャッシュTTL(秒)")


class ValidationConfig(BaseModel):
    """検証設定"""

    min_validation: int = Field(default=1, ge=0, le=10)
    max_validation: int = Field(default=3, ge=0, le=10)
    strictness: Literal["strict", "moderate", "lenient"] = Field(default="moderate")
    max_additional_queries: int = Field(default=3, ge=1, le=10, description="1回の検証で生成する追加クエリの最大数")

    @field_validator("max_validation")
    @classmethod
    def validate_max(cls, v: int, info) -> int:
        if "min_validation" in info.data and v < info.data["min_validation"]:
            raise ValueError("max_validation must be >= min_validation")
        return v


class LoggingConfig(BaseModel):
    """ロギング設定"""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: str = "{time:YYYY-MM-DDTHH:mm:ss.SSSSSSZ} [{level}] [{extra[category]}] {message}"
    rotation: str = "1 day"
    retention: str = "30 days"


class LangfuseConfig(BaseModel):
    """Langfuse設定 (オプション)"""

    enabled: bool = False
    host: Optional[str] = "http://127.0.0.1:3000"
    public_key: Optional[str] = None
    secret_key: Optional[str] = None


class LearningConfig(BaseModel):
    """Echo agent self-improvement settings"""

    enabled: bool = True
    auto_memory: bool = True
    auto_memory_max_per_session: int = Field(default=2, ge=0, le=10)
    correction_reflection: bool = True
    session_summary: bool = True
    history_search: bool = True
    history_search_limit: int = Field(default=10, ge=1, le=50)


class ResearchConfig(BaseModel):
    """Collaborative multi-agent research settings (Co-Scientist + Robin inspired)"""

    max_rounds: int = Field(default=3, ge=0, le=100, description="Max research iteration rounds (0=unlimited)")
    debates_per_round: int = Field(default=10, ge=0, le=50, description="Max ELO debates per round")
    hypotheses_per_round: int = Field(default=5, ge=1, le=20, description="Max hypotheses generated per round")
    parallel_instances: int = Field(default=3, ge=1, le=5, description="Finch-style parallel code execution instances")
    code_timeout: int = Field(default=30, ge=5, le=120, description="Code execution sandbox timeout (seconds)")
    search_results_per_query: int = Field(default=5, ge=1, le=10, description="Search results per sub-question")


class EchoConfig(BaseModel):
    """Echo agent settings"""

    model: str = Field(default="kimi-k2.6:cloud", description="Default model for Echo agent")
    max_tool_calls: int = Field(default=10, ge=1, le=50, description="Max tool iterations per turn")
    context_messages: int = Field(default=50, ge=5, le=200, description="Chat history messages to keep")
    shell_timeout: int = Field(default=120, ge=1, le=600, description="Shell command timeout (seconds)")
    confirm_destructive: bool = Field(default=True, description="Confirm before destructive commands")
    auto_memory: bool = Field(default=True, description="Auto-save corrections and decisions")
    memory_dir: Path = Field(default=Path.home() / ".hermes" / "memory", description="Memory directory")
    history_dir: Path = Field(default=Path.home() / ".hermes" / "history" / "echo", description="History directory")
    learning: LearningConfig = Field(default_factory=LearningConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)


class HermesConfig(BaseModel):
    """Hermes全体設定"""

    work_dir: Path = Field(default=Path.home() / ".hermes")
    language: Literal["ja", "en"] = "ja"

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    echo: EchoConfig = Field(default_factory=EchoConfig)

    class Config:
        use_enum_values = True

    def save_to_yaml(self, path: Path) -> None:
        """YAMLファイルに保存"""
        import yaml

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.model_dump(mode="json", exclude_none=True),
                f,
                allow_unicode=True,
                default_flow_style=False,
            )

    @classmethod
    def load_from_yaml(cls, path: Path) -> "HermesConfig":
        """YAMLファイルから読み込み"""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
