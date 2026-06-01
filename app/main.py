from fastapi import FastAPI, File, UploadFile

from app.data import get_dataset_stats, upload_dataset

app = FastAPI(
    title="Gym Progress Oracle",
    description="An API that analyzes gym progress data from CSV files and answers questions using a typed LLM chain.",
    version="0.1.0",
)

# Health check för att se till att vår server är status 200 / ok


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


# Första version av vår upload, ska testa att vår CSV går att laddas upp och valideras korrekt
@app.post("/data/upload")
def upload_data(file: UploadFile = File(...)) -> dict:
    return upload_dataset(file)


@app.get("/data/stats")
def data_stats() -> dict:
    return get_dataset_stats()
