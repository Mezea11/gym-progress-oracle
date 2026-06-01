from fastapi import FastAPI, File, UploadFile
from app.chain.pipeline import gym_oracle_chain
from app.schemas import AskRequest, AskResponse, PromptBuilderInput
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


# Hämta ut statistik från vår CSV, gör kalkulationer på vår data
@app.get("/data/stats")
def data_stats() -> dict:
    return get_dataset_stats()

# Använd vår runnable engine för att få fram AI genererade svar, få in input, generera output


@app.post("/ai/ask", response_model=AskResponse)
def ask_ai(request: AskRequest) -> AskResponse:
    stats = get_dataset_stats()

    chain_input = PromptBuilderInput(
        question=request.question,
        stats=stats,
    )

    result = gym_oracle_chain.invoke(chain_input)

    return AskResponse(
        question=result.question,
        answer=result.answer,
        model=result.model,
    )
