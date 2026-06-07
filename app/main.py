from pathlib import Path
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from app.chain.pipeline import gym_oracle_chain
from app.schemas import AskRequest, AskResponse, PromptBuilderInput, UploadResponse
from app.data import clear_dataset, get_dataset_insights, get_dataset_stats, upload_dataset

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
logger = logging.getLogger("uvicorn.error")

app = FastAPI(
    title="Gym Progress Oracle",
    description="An API that analyzes gym progress data from CSV files and answers questions using a typed LLM chain.",
    version="0.1.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")

# Health check för att se till att vår server är status 200 / ok


@app.get("/health", response_model=dict[str, str])
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# Första version av vår upload, ska testa att vår CSV går att laddas upp och valideras korrekt
@app.post("/data/upload", response_model=UploadResponse)
def upload_data(file: UploadFile = File(...)) -> UploadResponse:
    logger.info("Dataset upload started filename=%s", file.filename)
    result = upload_dataset(file)
    logger.info(
        "Dataset upload completed filename=%s rows=%s column_count=%s",
        file.filename,
        result.get("rows"),
        len(result.get("columns", [])),
    )
    return result


# Hämta ut statistik från vår CSV, gör kalkulationer på vår data
@app.get("/data/stats")
def data_stats() -> dict:
    logger.info("Dataset stats requested")
    return get_dataset_stats()


@app.get("/data/insights")
def data_insights() -> dict:
    logger.info("Dataset insights requested")
    return get_dataset_insights()


@app.delete("/data/clear")
def clear_data() -> dict[str, int | str]:
    return clear_dataset()

# Använd vår runnable engine för att få fram AI genererade svar, få in input, generera output


@app.post("/ai/ask", response_model=AskResponse)
def ask_ai(request: AskRequest) -> AskResponse:
    logger.info("AI question received question_length=%s",
                len(request.question))
    try:
        stats = get_dataset_stats()
        insights = get_dataset_insights()
    except HTTPException as error:
        if (
            error.status_code == 404
            and str(error.detail) == "No dataset has been uploaded yet."
        ):
            raise HTTPException(
                status_code=400,
                detail="Upload a dataset before asking AI questions.",
            ) from error
        raise

    chain_stats = {
        **stats,
        "insights": insights,
    }

    chain_input = PromptBuilderInput(
        question=request.question,
        stats=chain_stats,
    )

    result = gym_oracle_chain.invoke(chain_input)

    return AskResponse(
        question=result.question,
        answer=result.answer,
        model=result.model,
    )
