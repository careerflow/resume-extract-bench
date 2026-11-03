"""Benchmark configuration via environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    data_dir: Path = Path.home() / ".cache" / "resume-bench"
    output_dir: Path = Path("./output")
    concurrency: int = 4
    retries: int = 2

    llama_cloud_api_key: str = ""
    reducto_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    openrouter_api_key: str = ""

    careerflow_api_key: str = ""
    careerflow_api_url: str = ""

    model_config = {
        "env_prefix": "RESUME_BENCH_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "env_nested_delimiter": "__",
    }


settings = Settings()
