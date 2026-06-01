from fastapi import FastAPI

app = FastAPI(
    title="Gym Progress Oracle",
    description="An API that analyzes gym progress data from CSV files and answers questions using a typed LLM chain.",
    version="0.1.0",
)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
