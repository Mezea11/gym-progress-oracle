import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Laddar .env om den finns. Miljövariabler från systemet prioriteras fortfarande.
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
	value = os.getenv(name)

	if value is None:
		return default

	try:
		return int(value)
	except ValueError:
		return default


def _env_bool(name: str, default: bool) -> bool:
	value = os.getenv(name)

	if value is None:
		return default

	return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
	model_name: str
	model_max_new_tokens: int
	model_do_sample: bool
	model_return_full_text: bool
	database_path: Path
	database_table_name: str
	huggingface_token: str | None

	@classmethod
	def from_env(cls) -> "Settings":
		return cls(
			model_name=os.getenv(
				"MODEL_NAME",
				"HuggingFaceTB/SmolLM2-135M-Instruct",
			),
			model_max_new_tokens=_env_int("MODEL_MAX_NEW_TOKENS", 160),
			model_do_sample=_env_bool("MODEL_DO_SAMPLE", False),
			model_return_full_text=_env_bool("MODEL_RETURN_FULL_TEXT", False),
			database_path=Path(
				os.getenv(
					"DATABASE_PATH",
					str(PROJECT_ROOT / "app" / "gym_progress.db"),
				)
			),
			database_table_name=os.getenv("DATABASE_TABLE_NAME", "uploaded_dataset"),
			huggingface_token=os.getenv("HUGGINGFACE_TOKEN"),
		)


settings = Settings.from_env()

