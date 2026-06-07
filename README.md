# Gym Progress Oracle

Gym Progress Oracle är ett FastAPI-projekt som:

- laddar upp och validerar träningsdata från CSV,
- räknar ut statistik med Pandas,
- svarar på frågor på svenska via en typad LLM-kedja.

Projektet följer principen: **Pandas räknar, modellen formulerar**.

## Funktioner

- CSV-upload med validering av obligatoriska kolumner.
- Persistens i lokal SQLite-databas så uppladdad data lever kvar mellan omstarter.
- Statistik-API för volym, tyngsta lyft och estimerad 1RM.
- `/ai/ask`-endpoint som bygger prompt från verifierade fakta och returnerar AI-svar.
- Guardrails i parser-steget som fångar trasiga eller läckande modellsvar.

## Teknikstack

- Python
- FastAPI
- Pandas
- SQLite (via Python stdlib `sqlite3`)
- Transformers (Hugging Face pipeline)
- Uvicorn
- Pytest

## Projektstruktur

```text
app/
	main.py              # FastAPI-app och endpoints
  database/
    __init__.py        # publika databasfunktioner
    sqlite_store.py    # SQLite-lager (save/load/clear)
	data.py              # Upload, validering och statistik
	schemas.py           # Pydantic-modeller för API och chain
	chain/
		runnable.py        # Enkel typed runnable engine
		steps.py           # PromptBuilder, LLMRunner, ResponseParser
		pipeline.py        # Kedjekoppling med | operatorn
sample_data/
  gym_progress.csv                         # Mindre exempeldata
  christian_mock_gym_progression_2023_2026.csv  # Större testdataset
```

## Krav

- Python 3.14+ (enligt `pyproject.toml`)
- `uv` installerat (rekommenderat för detta projekt)

## Installation

1. Klona projektet.
2. Installera beroenden:

```powershell
uv sync
```

Detta skapar/uppdaterar miljön baserat på `pyproject.toml` och `uv.lock`.

## Konfiguration (.env)

Projektet använder central konfiguration i `app/config.py`.
Inställningar läses via `os.getenv()` och `.env` (med `python-dotenv`).

1. Kopiera `.env.example` till `.env`.
2. Justera värden vid behov.

Stödda variabler:

- `HUGGINGFACE_TOKEN` (valfri, behövs normalt inte för lokal modell)
- `MODEL_NAME`
- `MODEL_MAX_NEW_TOKENS`
- `MODEL_DO_SAMPLE`
- `MODEL_RETURN_FULL_TEXT`
- `DATABASE_PATH`
- `DATABASE_TABLE_NAME`

## Starta API:t

```powershell
uv run uvicorn app.main:app --reload
```

Servern kör normalt på:

- `http://127.0.0.1:8000`

Interaktiv dokumentation:

- `http://127.0.0.1:8000/docs`

Frontend:

- `http://127.0.0.1:8000/`

## Använd frontend

Efter att servern startat kan du använda webbgränssnittet för hela flödet:

1. Öppna `http://127.0.0.1:8000/`.
2. Ladda upp en CSV i sektionen **CSV-uppladdning**.
3. Hämta statistik i sektionen **Statistik**.
4. Fråga AI:n i sektionen **AI-chat**.

Tips: chatpanelen ligger bredvid statistik på större skärmar för enklare analys + prompting.

## Testdataset

I `sample_data/` finns två dataset:

- `gym_progress.csv` (mindre och snabb att testa med)
- `christian_mock_gym_progression_2023_2026.csv` (större, bra för att testa långa statistiklistor och AI-frågor)

Du kan använda båda i frontend (eller via `POST /data/upload`) för att verifiera olika scenarier.

## API-endpoints

### `GET /health`

Hälsokontroll.

Exempel svar:

```json
{
  "status": "ok"
}
```

### `POST /data/upload`

Laddar upp en CSV-fil och sätter den som aktuell dataset i minnet.

Obligatoriska kolumner:

- `date`
- `exercise`
- `weight`
- `reps`
- `sets`

Observera: extra kolumner är tillåtna (exempeldata innehåller även `bodyweight`).

Exempel med PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/data/upload" `
	-H "accept: application/json" `
	-H "Content-Type: multipart/form-data" `
	-F "file=@sample_data/gym_progress.csv"
```

### `GET /data/stats`

Returnerar statistik beräknad från uppladdad dataset, bl.a.:

- antal rader,
- antal unika övningar,
- tyngsta lyft,
- total volym per övning,
- estimerad 1RM per övning,
- `describe` från Pandas.

Om ingen fil är uppladdad returneras `404`.

### `DELETE /data/clear`

Rensar sparad dataset i SQLite.

Exempel svar:

```json
{
  "status": "cleared",
  "rows_removed": 7
}
```

### `POST /ai/ask`

Tar emot en fråga om uppladdad data och returnerar ett svar från kedjan.

Request body:

```json
{
  "question": "Vilken övning har högst total volym?"
}
```

Validering:

- `question` måste vara 3-500 tecken.

Exempel svar:

```json
{
  "question": "Vilken övning har högst total volym?",
  "answer": "Deadlift har högst total träningsvolym i den uppladdade datan.",
  "model": "HuggingFaceTB/SmolLM2-135M-Instruct"
}
```

## Typiskt flöde

1. Starta servern.
2. Ladda upp CSV via frontend på `http://127.0.0.1:8000/` eller anropa `POST /data/upload`.
3. Verifiera statistik i frontend eller med `GET /data/stats`.
4. Fråga modellen via frontend-chatten eller `POST /ai/ask`.

Notering: datan sparas i `app/gym_progress.db` och finns kvar tills du laddar upp ny CSV eller kör `DELETE /data/clear`.

## Exempelprompter till AI

Använd dessa frågor i chatten för att snabbt komma igång.

Basfrågor:

- `Vilken övning har högst estimerad 1RM?`
- `Vilken övning har lägst estimerad 1RM?`
- `Vilken övning har högst total volym?`
- `Vilket är mitt tyngsta enskilda lyft?`

Övningsspecifika frågor:

- `Vad är estimerad 1RM i squat?`
- `Vad är estimerad 1RM i deadlift?`
- `Vad är total volym i bench press?`

Jämförelsefrågor:

- `Jämför estimerad 1RM mellan squat och deadlift.`
- `Hur stor är skillnaden i estimerad 1RM mellan deadlift och squat?`
- `Rangordna övningarna efter estimerad 1RM.`

Frågor utanför datat (för att testa fallback):

- `Vad är min VO2max?`
- `Hur många kalorier bränner jag per pass?`
- `Kan du förutspå min 1RM om tre månader?`

## Hur AI-kedjan fungerar

Kedjan definieras i `app/chain/pipeline.py`:

1. `PromptBuilder`
2. `LLMRunner`
3. `ResponseParser`

Pipeline:

```python
gym_oracle_chain = PromptBuilder() | LLMRunner() | ResponseParser()
```

Kort om stegen:

- `PromptBuilder` skapar en svensk prompt från fråga + verifierade stats.
- `LLMRunner` kör textgenerering med modellen `HuggingFaceTB/SmolLM2-135M-Instruct`.
- `ResponseParser` städar output och fallbackar till fakta om svaret ser trasigt ut.

## Felhantering

- Upload accepterar endast `.csv`.
- Tom eller ogiltig CSV ger `400`.
- Saknade obligatoriska kolumner ger `400`.
- Ingen uppladdad dataset ger `404` för stats/AI-flöde.

## Utveckling

Kör server i utvecklingsläge:

```powershell
uv run uvicorn app.main:app --reload
```

Kör tester:

```powershell
uv run pytest
```

Kör endast appens tester med verbose output:

```powershell
uv run pytest app/tests/ -v
```

Kör prompt-regression med kategori-trösklar (CI-gate):

```powershell
uv run pytest app/tests/test_prompt_regression.py -v
```

Kör snabb prompt-rapport i tabellformat:

```powershell
uv run python -m app.tests.prompt_eval_cli --enforce-thresholds
```

## Testtäckning (KK2)

Projektet innehåller tester för flera olika aspekter enligt KK2:

- Kedjesteg i isolation i `app/tests/test_chain.py`:
  - promptkontrakt och svarmarkörer,
  - parser-extraktion mellan markörer,
  - robust fallback vid prompt-läckage,
  - fallback-logik för lägst/specifik/skillnad i estimerad 1RM.
- Endpoints via FastAPI TestClient i `app/tests/test_endpoints.py`:
  - `GET /health` returnerar 200,
  - `POST /data/upload` med giltig CSV returnerar metadata,
  - `POST /data/upload` med ogiltig fil returnerar 400,
  - `GET /data/stats` utan dataset returnerar 404,
  - `GET /data/stats` efter upload returnerar statistik,
  - `DELETE /data/clear` rensar dataset och `GET /data/stats` ger därefter 404.
- Mockad modell/kedja i endpointtest:
  - `/ai/ask` testas utan riktig modellnedladdning genom mockad chain-invoke,
  - `/ai/ask` utan uppladdad data returnerar 404.

Senaste körning lokalt:

- `uv run pytest app/tests/ -v` -> 13 passed
- `uv run pytest app/tests/ -v` -> 16 passed

## Kända begränsningar

- Första AI-anropet kan vara långsamt om modellen behöver laddas ner.
- Projektet kräver relativt ny Python-version.

## Nästa förbättringar (förslag)

- Persistens i databas eller filcache.
- Fler automatiserade tester för endpoints och chain-steg.
- Konfigurerbar modell via miljövariabler.
- Asynkrona endpoints vid behov.
