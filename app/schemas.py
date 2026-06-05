from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):  # svaret från /data/upload
    rows: int
    columns: list[str]
    dtypes: dict[str, str]


class AskRequest(BaseModel):  # frågan användaren skickar till /ai/ask
    question: str = Field(
        min_length=3,
        max_length=500,
        description="A natural language question about the uploaded gym dataset.",
    )


class AskResponse(BaseModel):  # färdigt svar från /ai/ask
    question: str
    answer: str
    model: str


class PromptBuilderInput(BaseModel):  # fråga + statistik in i kedjan
    question: str
    stats: dict[str, Any]


class PromptBuilderOutput(BaseModel):  # färdig prompt ut från första steget
    question: str
    prompt: str
    facts_summary: str
    stats: dict[str, Any]


class LLMRunnerOutput(BaseModel):  # råtext från modellen
    question: str
    raw_output: str
    model: str
    facts_summary: str
    stats: dict[str, Any]


class ResponseParserOutput(BaseModel):  # städat slutresultat
    question: str
    answer: str
    model: str
