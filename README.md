# Gym Progress Oracle

Gym Progress Oracle är en FastAPI-app som analyserar träningsdata från CSV och svarar på frågor med en typad AI-kedja.

Projektets huvudprincip:

- Pandas räknar statistik
- LLM formulerar svaret

## Nuvarande status

Detta är implementerat i koden idag:

- SQLite-persistens för uppladdad dataset (save/load/clear).
- CSV-upload med validering.
- Filstorleksgräns för upload: max 2 MB.
- Endpoint-logging av anrop och fel med metadata.
- `/ai/ask` returnerar 400 om dataset saknas.
- Runnable-kedja med `PromptBuilder | LLMRunner | ResponseParser`.
- Parser-guardrails och fallback mot dålig modelloutput och prompt-läckage.
- Edge case-tester för upload, endpoints och fallback-beteenden.

## Teknikstack

- Python
- FastAPI
- Pandas
- SQLite (`sqlite3`)
- Transformers (Hugging Face pipeline)
- Uvicorn
- Pytest

## Krav

- Python 3.12+
- `uv`

## Installation

```powershell
uv sync
```

## Starta applikationen

```powershell
uv run uvicorn app.main:app --reload
```

URL:er:

- API: http://127.0.0.1:8000
- Docs: http://127.0.0.1:8000/docs
- Frontend: http://127.0.0.1:8000/

## Konfiguration

Konfiguration läses i `app/config.py` via miljö variabler (och `.env` om filen finns).

Vanliga variabler:

- `MODEL_NAME`
- `MODEL_MAX_NEW_TOKENS`
- `MODEL_DO_SAMPLE`
- `MODEL_RETURN_FULL_TEXT`
- `HUGGINGFACE_TOKEN`
- `DATABASE_PATH`
- `DATABASE_TABLE_NAME`

## API-endpoints

### `GET /health`

Returnerar:

```json
{
  "status": "ok"
}
```

### `POST /data/upload`

Laddar upp CSV och sparar dataset i SQLite.

Validering inkluderar:

- Filen måste vara `.csv`
- Max filstorlek 2 MB
- Filen får inte vara tom
- Obligatoriska kolumner: `date`, `exercise`, `weight`, `reps`, `sets`
- Numeriska fält måste vara giltiga och positiva
- Datum måste vara giltiga
- Exercise får inte vara tom efter trim

Response innehåller:

- antal rader
- kolumnlista
- dtypes

### `GET /data/stats`

Returnerar beräknad statistik från dataset.

Om dataset saknas: `404`.

### `GET /data/insights`

Returnerar fördjupade insikter från dataset.

### `DELETE /data/clear`

Rensar sparat dataset i SQLite.

### `POST /ai/ask`

Tar emot en fråga och returnerar:

- `question`
- `answer`
- `model`

Viktigt beteende:

- Om dataset saknas returnerar endpointen `400` med meddelandet:
  `Upload a dataset before asking AI questions.`

## AI-kedjan

Kedjan i `app/chain/pipeline.py`:

```python
gym_oracle_chain = PromptBuilder() | LLMRunner() | ResponseParser()
```

Steg:

1. `PromptBuilder`: bygger prompt från verifierade stats + fråga.
2. `LLMRunner`: kör modellen och returnerar raw output.
3. `ResponseParser`: extraherar/städar output och fallbackar till datadrivet svar vid modellfel, prompt-eko eller suspekt output.

## Logging

Applikationen loggar metadata (inte hela CSV-innehåll):

- upload start/slut
- stats/insights-anrop
- AI-frågor (frågelängd)
- modellfel
- parser fallback
- upload-valideringsfel (inkl. 400/413)

## Tester

Kör alla tester:

```powershell
uv run pytest
```

Kör endpointtester:

```powershell
uv run pytest app/tests/test_endpoints.py -q
```

Kör kedjetester:

```powershell
uv run pytest app/tests/test_chain.py -q
```

Kör prompt-regression:

```powershell
uv run pytest app/tests/test_prompt_regression.py -v
```

## Testtäckning för robusthet

Projektet har tester för bland annat:

- ogiltig filtyp, tom fil, för stor fil, saknade kolumner
- ogiltiga numeriska värden och datum
- `/ai/ask` utan dataset
- parser fallback vid prompt-läckage/suspekt output
- LLMRunner exception-path och fortsatt hantering i parser
